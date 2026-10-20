"""
Training script for Convolutional Autoencoder (AE) and Variational Autoencoder (VAE) baselines.
Supports GPU acceleration, tqdm progress monitoring, seamless training resume, and TensorBoard logging.

Usage examples:
    # Train Standard Autoencoder
    python scripts/train_autoencoder.py --model ae --category bottle --epochs 50 --batch_size 16

    # Train Variational Autoencoder
    python scripts/train_autoencoder.py --model vae --category bottle --epochs 50 --batch_size 16

    # Resume training from last checkpoint
    python scripts/train_autoencoder.py --model ae --category bottle --resume auto

    # Train with TensorBoard logging
    python scripts/train_autoencoder.py --model ae --category bottle --tensorboard
    # Then visualize: tensorboard --logdir runs
"""

import argparse
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import get_mvtec_dataloader
from src.models import (
    AutoencoderLoss,
    ConvAutoencoder,
    ConvVAE,
    VAELoss,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder / VAE baseline for Anomaly Detection")
    parser.add_argument("--model", type=str, default="ae", choices=["ae", "vae"], help="Model type: 'ae' or 'vae'")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category name (e.g. bottle, cable, tile)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to MVTec dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of latent bottleneck vector")
    parser.add_argument("--alpha", type=float, default=0.85, help="Weight for L1 loss vs SSIM loss")
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="KL divergence weight for VAE")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=10, help="Save periodic checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from, or 'auto'")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--log_dir", type=str, default="./runs", help="Root directory for TensorBoard logs")
    return parser.parse_args()


def get_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    return device


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: float,
    best_loss: float,
    model_type: str,
    category: str,
    latent_dim: int,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "model_type": model_type,
        "category": category,
        "latent_dim": latent_dim,
    }
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(f"Resumed from epoch {checkpoint['epoch']} with recorded loss {checkpoint.get('loss', 0.0):.4f}")
    return start_epoch, best_loss


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    model_type: str,
    epoch: int,
    total_epochs: int,
) -> dict:
    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_kl = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}]", leave=True)

    for batch in pbar:
        images = batch["image"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if model_type == "ae":
            recon, _ = model(images)
            loss, loss_dict = loss_fn(recon, images)
        else:  # vae
            recon, mu, logvar = model(images)
            loss, loss_dict = loss_fn(recon, images, mu, logvar)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_l1 += loss_dict.get("l1", 0.0) * batch_size
        running_ssim += loss_dict.get("ssim_loss", 0.0) * batch_size
        running_kl += loss_dict.get("kl_div", 0.0) * batch_size
        total_samples += batch_size

        postfix = {
            "loss": f"{loss.item():.4f}",
            "l1": f"{loss_dict.get('l1', 0.0):.4f}",
            "ssim": f"{loss_dict.get('ssim_loss', 0.0):.4f}",
        }
        if model_type == "vae":
            postfix["kl"] = f"{loss_dict.get('kl_div', 0.0):.2f}"
        pbar.set_postfix(postfix)

    epoch_metrics = {
        "loss": running_loss / total_samples,
        "l1": running_l1 / total_samples,
        "ssim": running_ssim / total_samples,
        "kl": running_kl / total_samples,
    }
    return epoch_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print("=" * 70)
    print(f"Initializing Training for Model: {args.model.upper()} | Category: {args.category.upper()}")
    print(f"Device: {device} | Total Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    if args.tensorboard:
        print(f"TensorBoard logging: ENABLED → {args.log_dir}/{args.category}_{args.model}")
    print("=" * 70)

    # 1. DataLoader: Autoencoder is trained strictly on normal images
    # normalize_imagenet=False to keep pixel values bounded in [0, 1] for Sigmoid decoder
    train_loader = get_mvtec_dataloader(
        root_dir=args.data_dir,
        category=args.category,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        normalize_imagenet=False,
    )
    print(f"Training dataset loaded: {len(train_loader.dataset)} normal samples.")

    # 2. Model instantiation
    if args.model == "ae":
        model = ConvAutoencoder(in_channels=3, base_channels=32, latent_dim=args.latent_dim)
        loss_fn = AutoencoderLoss(alpha=args.alpha, in_channels=3)
    else:
        model = ConvVAE(in_channels=3, base_channels=32, latent_dim=args.latent_dim)
        loss_fn = VAELoss(alpha=args.alpha, kl_weight=args.kl_weight, in_channels=3)

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 4. Checkpoint directory management
    checkpoint_dir = Path(args.checkpoint_dir) / f"{args.category}_{args.model}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = checkpoint_dir / "last.pth"
    best_ckpt_path = checkpoint_dir / "best.pth"

    # 4b. TensorBoard writer
    writer = None
    if args.tensorboard:
        log_path = Path(args.log_dir) / f"{args.category}_{args.model}"
        writer = SummaryWriter(log_dir=str(log_path))
        print(f"TensorBoard log directory: {log_path.resolve()}")

    start_epoch = 1
    best_loss = float("inf")

    # 5. Handle training resume
    if args.resume:
        resume_path = None
        if args.resume.lower() == "auto":
            if last_ckpt_path.exists():
                resume_path = last_ckpt_path
            else:
                print(f"No previous checkpoint found at {last_ckpt_path}. Starting fresh training.")
        else:
            resume_path = Path(args.resume)

        if resume_path is not None and resume_path.exists():
            start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, scheduler)

    # 6. Training loop
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...\n")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            model_type=args.model,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        epoch_loss = epoch_metrics["loss"]

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/total", epoch_loss, epoch)
            writer.add_scalar("Loss/l1", epoch_metrics["l1"], epoch)
            writer.add_scalar("Loss/ssim", epoch_metrics["ssim"], epoch)
            if args.model == "vae":
                writer.add_scalar("Loss/kl", epoch_metrics["kl"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            save_checkpoint(
                checkpoint_path=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=epoch_loss,
                best_loss=best_loss,
                model_type=args.model,
                category=args.category,
                latent_dim=args.latent_dim,
            )

        # Save latest checkpoint
        save_checkpoint(
            checkpoint_path=last_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=epoch_loss,
            best_loss=best_loss,
            model_type=args.model,
            category=args.category,
            latent_dim=args.latent_dim,
        )

        # Periodic checkpoint
        if epoch % args.save_freq == 0:
            periodic_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                checkpoint_path=periodic_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=epoch_loss,
                best_loss=best_loss,
                model_type=args.model,
                category=args.category,
                latent_dim=args.latent_dim,
            )

        status_msg = f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f} (Best: {best_loss:.4f}, LR: {current_lr:.2e})"
        if is_best:
            status_msg += " [Best Checkpoint Saved]"
        print(status_msg)

    if writer is not None:
        writer.close()
        print(f"\nTensorBoard logs saved. View with: tensorboard --logdir {Path(args.log_dir).resolve()}")

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Loss: {best_loss:.4f}")
    print(f"Best checkpoint: {best_ckpt_path.resolve()}")
    print(f"Last checkpoint: {last_ckpt_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

