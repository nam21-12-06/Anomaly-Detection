"""
Milestone 2.2: Build Normal Feature Memory Bank.

Offline pipeline to extract deep spatial patch representations from all normal
training samples of a specified MVTec AD category, saving the bank to disk.

Usage:
    python scripts/build_feature_bank.py --category bottle --backbone resnet18
    python scripts/build_feature_bank.py --category bottle --backbone resnet50 --batch_size 32
"""

import argparse
import datetime
from pathlib import Path
import sys
import time
from typing import List, Tuple

import torch
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_mvtec_dataloader
from src.models import FeatureMemoryBank, ResNetExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and save normal feature memory bank.")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="MVTec AD category name (e.g. bottle, carpet, hazelnut).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to MVTec AD dataset directory.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Torchvision CNN backbone architecture.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        help="Intermediate residual layers to extract features from.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for feature extraction dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/feature_bank",
        help="Directory to save the serialized memory bank artifact.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Local directory to download and cache pretrained backbone weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for extraction ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def build_feature_bank(
    category: str,
    data_dir: str = "data",
    backbone: str = "resnet18",
    layers: Tuple[str, ...] = ("layer2", "layer3"),
    batch_size: int = 16,
    output_dir: str = "checkpoints/feature_bank",
    weights_dir: str = "weights",
    device: str = "cuda",
) -> Tuple[Path, int, Tuple[int, int]]:
    """
    Extracts deep patch features from all normal training samples of a category
    and serializes the resulting FeatureMemoryBank to disk.

    Returns:
        Tuple of (saved_file_path, total_patches, spatial_shape).
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    data_path = PROJECT_ROOT / data_dir
    save_dir = PROJECT_ROOT / output_dir
    local_weights_dir = PROJECT_ROOT / weights_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    local_weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  BUILDING FEATURE MEMORY BANK: '{category.upper()}'")
    print("=" * 70)
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU Model:        {torch.cuda.get_device_name(0)}")
    print(f"Dataset root:     {data_path}")
    print(f"Backbone:         {backbone} (ImageNet-1K pretrained)")
    print(f"Weights cache:    {local_weights_dir}")
    print(f"Target layers:    {list(layers)}")
    print(f"Batch size:       {batch_size}")
    print(f"Output directory: {save_dir}")
    print("-" * 70)

    # 1. Initialize DataLoader (Normal training samples only)
    # Important: normalize_imagenet=True matches ImageNet pretrained backbone expectations
    train_loader = get_mvtec_dataloader(
        root_dir=data_path,
        category=category,
        split="train",
        batch_size=batch_size,
        shuffle=False,
        normalize_imagenet=True,
    )
    num_samples = len(train_loader.dataset)
    print(f"Loaded normal training samples: {num_samples}")

    # 2. Initialize Extractor
    extractor = ResNetExtractor(
        backbone=backbone,
        pretrained=True,
        layers=tuple(layers),
        weights_dir=local_weights_dir,
        device=device,
    )
    print(f"Extractor ready: fused embedding dimension = {extractor.embedding_dim}")

    # 3. Extract patch embeddings across train set
    patch_features_list = []
    spatial_shape = None
    start_time = time.time()
    total_patches_extracted = 0

    pbar = tqdm(
        train_loader,
        desc=f"Extracting '{category}' features",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            # patches: (B, N_patches, embedding_dim), spatial_shape: (H_out, W_out)
            patches, shape = extractor.get_patch_embeddings(images)
            spatial_shape = shape
            # Move to CPU to keep VRAM clean during extraction loop
            patch_cpu = patches.cpu()
            patch_features_list.append(patch_cpu)
            total_patches_extracted += patch_cpu.shape[0] * patch_cpu.shape[1]
            pbar.set_postfix({"vectors": f"{total_patches_extracted:,}"})

    elapsed_time = time.time() - start_time
    extractor.remove_hooks()

    # 4. Concatenate and fit Memory Bank
    all_patches = torch.cat(patch_features_list, dim=0)  # (N_images, N_patches, D)
    b_total, n_patches, d = all_patches.shape
    total_vectors = b_total * n_patches

    bank = FeatureMemoryBank(device=device)
    bank.fit(all_patches)

    # 5. Metadata packaging
    metadata = {
        "category": category,
        "backbone": backbone,
        "layers": list(layers),
        "num_train_samples": num_samples,
        "spatial_shape": list(spatial_shape),
        "patches_per_image": n_patches,
        "total_vectors": total_vectors,
        "embedding_dim": d,
        "created_at": datetime.datetime.now().isoformat(),
    }

    # 6. Save to disk
    save_path = save_dir / f"{category}_{backbone}_bank.pt"
    bank.save(save_path, metadata=metadata)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)

    print("-" * 70)
    print("  EXTRACTION COMPLETE & SAVED")
    print("-" * 70)
    print(f"Feature grid shape per image: {spatial_shape[0]}x{spatial_shape[1]} ({n_patches} patches/image)")
    print(f"Total patch vectors in bank:  {total_vectors:,}")
    print(f"Feature vector dimension (D): {d}")
    print(f"Extraction time:              {elapsed_time:.2f} seconds ({num_samples / elapsed_time:.1f} img/s)")
    print(f"Saved memory bank:            {save_path} ({file_size_mb:.2f} MB)")
    print("=" * 70)

    return save_path, total_vectors, spatial_shape


def main():
    args = parse_args()
    build_feature_bank(
        category=args.category,
        data_dir=args.data_dir,
        backbone=args.backbone,
        layers=tuple(args.layers),
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
