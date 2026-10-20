"""
Evaluation script for Convolutional Autoencoder (AE) and Variational Autoencoder (VAE).
Evaluates Image AUROC, Pixel AUROC, and PRO scores on MVTec AD test sets,
and exports side-by-side reconstruction & anomaly localization figures.

Usage:
    # Evaluate both AE and VAE on bottle
    python scripts/evaluate_autoencoder.py --category bottle --model both

    # Evaluate only Autoencoder
    python scripts/evaluate_autoencoder.py --category bottle --model ae

    # Evaluate only Variational Autoencoder
    python scripts/evaluate_autoencoder.py --category bottle --model vae
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data import get_mvtec_dataloader
from src.models import ConvAutoencoder, ConvVAE
from src.utils import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_score,
    overlay_mask_on_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AE / VAE on MVTec AD test set")
    parser.add_argument("--category", type=str, default="bottle", help="Category name (e.g. bottle, cable)")
    parser.add_argument("--model", type=str, default="both", choices=["ae", "vae", "both"], help="Model to evaluate")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to MVTec dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to checkpoints folder")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation plots")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of qualitative samples to visualize")
    return parser.parse_args()


def load_model(model_cls, ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    latent_dim = checkpoint.get("latent_dim", 128)
    model = model_cls(in_channels=3, base_channels=32, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_single_model(model, loader, device: torch.device, model_name: str):
    all_scores, all_labels = [], []
    all_maps, all_masks = [], []
    sample_items = []

    print(f"\nEvaluating {model_name} on test set...")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference ({model_name})"):
            imgs = batch["image"].to(device)
            masks = batch["mask"]
            labels = batch["label"]

            anom_maps = model.get_anomaly_map(imgs)
            anom_scores = model.get_anomaly_score(imgs)

            all_scores.extend(anom_scores.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_maps.append(anom_maps.cpu())
            all_masks.append(masks)

            # Collect a few representative anomaly samples for visualization
            if len(sample_items) < 8:
                recons = model(imgs)[0] if isinstance(model, (ConvAutoencoder, ConvVAE)) else model(imgs)
                for i in range(imgs.size(0)):
                    if labels[i].item() == 1 and len(sample_items) < 8:
                        sample_items.append({
                            "image": imgs[i].cpu().numpy().transpose(1, 2, 0),
                            "recon": recons[i].cpu().numpy().transpose(1, 2, 0),
                            "mask": masks[i].squeeze().numpy(),
                            "map": anom_maps[i].squeeze().cpu().numpy(),
                            "defect_type": batch["defect_type"][i],
                            "score": anom_scores[i].item(),
                        })

    all_maps = torch.cat(all_maps, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    print("Computing evaluation metrics...")
    img_auc = compute_image_auroc(all_labels, all_scores)
    pix_auc = compute_pixel_auroc(all_masks, all_maps)
    pro = compute_pro_score(all_masks, all_maps)

    metrics = {
        "Model": model_name,
        "Image AUROC": img_auc,
        "Pixel AUROC": pix_auc,
        "PRO Score": pro,
    }
    return metrics, sample_items


def visualize_reconstruction_grid(sample_items, save_path: Path, model_name: str):
    num_samples = min(len(sample_items), 5)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 5, figsize=(16, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Test Image",
        "Reconstruction",
        "Ground-Truth Mask",
        "Anomaly Residual Map",
        "Heatmap Overlay",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i in range(num_samples):
        item = sample_items[i]
        img = np.clip(item["image"], 0.0, 1.0)
        recon = np.clip(item["recon"], 0.0, 1.0)
        mask = item["mask"]
        anom_map = item["map"]

        # Normalize heatmap to [0, 1] for visualization
        anom_norm = (anom_map - anom_map.min()) / (anom_map.max() - anom_map.min() + 1e-8)
        overlay = overlay_mask_on_image(img, anom_norm, alpha=0.5, color=(1.0, 0.0, 0.0))

        # Col 0: Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(f"{item['defect_type']}\nScore: {item['score']:.3f}", fontsize=10, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Col 1: Reconstruction
        axes[i, 1].imshow(recon)
        axes[i, 1].axis("off")

        # Col 2: GT Mask
        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].axis("off")

        # Col 3: Anomaly Residual Map
        im3 = axes[i, 3].imshow(anom_norm, cmap="hot")
        axes[i, 3].axis("off")

        # Col 4: Overlay
        axes[i, 4].imshow(overlay)
        axes[i, 4].axis("off")

    plt.suptitle(f"Reconstruction & Anomaly Localization: {model_name.upper()}", fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization figure to: {save_path.resolve()}")
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Evaluation Pipeline | Category: '{args.category.upper()}' | Device: {device}")
    print("=" * 70)

    # Load test DataLoader (normalize_imagenet=False for Sigmoid decoder outputs in [0, 1])
    test_loader = get_mvtec_dataloader(
        root_dir=args.data_dir,
        category=args.category,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        normalize_imagenet=False,
    )
    print(f"Loaded test dataset: {len(test_loader.dataset)} samples.")

    models_to_eval = []
    if args.model in ["ae", "both"]:
        ae_path = Path(args.checkpoint_dir) / f"{args.category}_ae" / "best.pth"
        models_to_eval.append(("Autoencoder (AE)", ConvAutoencoder, ae_path, "ae"))
    if args.model in ["vae", "both"]:
        vae_path = Path(args.checkpoint_dir) / f"{args.category}_vae" / "best.pth"
        models_to_eval.append(("Variational Autoencoder (VAE)", ConvVAE, vae_path, "vae"))

    results_list = []

    for name, model_cls, ckpt_path, short_name in models_to_eval:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found for {name} at {ckpt_path}. Skipping.")
            continue

        model = load_model(model_cls, ckpt_path, device)
        metrics, sample_items = evaluate_single_model(model, test_loader, device, name)
        results_list.append(metrics)

        # Export qualitative visualization plot
        fig_path = output_dir / f"{args.category}_{short_name}_localization.png"
        visualize_reconstruction_grid(sample_items, fig_path, name)

    if results_list:
        df_res = pd.DataFrame(results_list)
        print("\n" + "=" * 70)
        print(f"BENCHMARK RESULTS ON MVTec AD: '{args.category.upper()}'")
        print("=" * 70)
        print(df_res.to_string(index=False))
        print("=" * 70)

        csv_path = output_dir / f"{args.category}_baseline_benchmark.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"Benchmark results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()

