"""
Milestone 2.3: Evaluate Feature Memory Bank with k-NN Distance Anomaly Detection.

Online inference pipeline:
1. Loads test samples (normal and defective) for a category.
2. Extracts multi-scale patch embeddings using ResNetExtractor.
3. Computes nearest-neighbor distances against FeatureMemoryBank.
4. Generates pixel-level anomaly heatmaps (bilinear upsampling + Gaussian smoothing)
   and image-level anomaly scores (max patch score).
5. Calculates Image-AUROC, Pixel-AUROC, and PRO score.
6. Exports benchmark metrics to CSV and qualitative anomaly localization plots.

Usage:
    python scripts/evaluate_feature_knn.py --category bottle --backbone resnet18
    python scripts/evaluate_feature_knn.py --category bottle --n_neighbors 3 --batch_size 16
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

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
from src.models import FeatureMemoryBank, ResNetExtractor
from src.utils import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_score,
    denormalize,
    overlay_mask_on_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate feature k-NN anomaly detection.")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="MVTec AD category name (e.g. bottle, cable, hazelnut).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Torchvision CNN backbone architecture.",
    )
    parser.add_argument(
        "--bank_path",
        type=str,
        default=None,
        help="Path to serialized memory bank (.pt). If None, defaults to checkpoints/feature_bank/{category}_{backbone}_bank.pt.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to MVTec AD dataset directory.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=1,
        help="Number of nearest neighbors to average for anomaly scoring.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for test evaluation dataloader.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128,
        help="Batch size for distance calculation chunks against memory bank.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian filter sigma for anomaly heatmap spatial smoothing.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/feature_knn",
        help="Directory to save evaluation results and visualizations.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Local directory containing cached pretrained backbone weights.",
    )
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=5,
        help="Number of qualitative defective samples to visualize in grid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def visualize_feature_knn_grid(
    sample_items: List[Dict],
    save_path: Path,
    category: str,
    backbone: str,
    n_neighbors: int,
):
    """
    Renders a qualitative inspection grid displaying:
    Original Test Image | Ground-Truth Mask | Anomaly Heatmap | Heatmap Overlay.
    """
    num_samples = len(sample_items)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Test Image",
        "Ground-Truth Mask",
        "Predicted Anomaly Heatmap",
        "Heatmap Overlay",
    ]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i in range(num_samples):
        item = sample_items[i]
        img = item["image"]
        mask = item["mask"]
        anom_map = item["map"]

        # Normalize heatmap to [0.0, 1.0] for clean colormap rendering
        min_v, max_v = float(anom_map.min()), float(anom_map.max())
        norm_map = (anom_map - min_v) / (max_v - min_v + 1e-8)
        overlay = overlay_mask_on_image(img, norm_map, alpha=0.45, color=(1.0, 0.0, 0.0))

        # Col 0: Original Test Image
        axes[i, 0].imshow(img)
        defect_label = item.get("defect_type", "defect")
        score_val = item.get("score", 0.0)
        axes[i, 0].set_ylabel(
            f"{defect_label}\nScore: {score_val:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Col 1: Ground-Truth Binary Mask
        axes[i, 1].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].axis("off")

        # Col 2: Continuous Anomaly Heatmap
        axes[i, 2].imshow(norm_map, cmap="jet")
        axes[i, 2].axis("off")

        # Col 3: Heatmap Overlay
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis("off")

    plt.suptitle(
        f"Feature Memory Bank k-NN Localization: {category.upper()} ({backbone.upper()}, k={n_neighbors})",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved qualitative visualization figure to: {save_path.resolve()}")
    plt.close()


def evaluate_feature_knn(
    category: str = "bottle",
    backbone: str = "resnet18",
    bank_path: Optional[str] = None,
    data_dir: str = "data",
    n_neighbors: int = 1,
    batch_size: int = 16,
    chunk_size: int = 128,
    sigma: float = 4.0,
    output_dir: str = "results/feature_knn",
    weights_dir: str = "weights",
    num_vis_samples: int = 5,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluates test set samples against a trained FeatureMemoryBank using patch-level
    nearest-neighbor distance calculations.

    Returns:
        Dictionary of quantitative evaluation metrics (Image AUROC, Pixel AUROC, PRO Score).
    """
    device_obj = torch.device(device)
    resolved_bank_path = (
        Path(bank_path)
        if bank_path is not None
        else Path("checkpoints/feature_bank") / f"{category}_{backbone}_bank.pt"
    )

    if not resolved_bank_path.exists():
        raise FileNotFoundError(
            f"Feature memory bank not found at '{resolved_bank_path}'. "
            f"Please run scripts/build_feature_bank.py --category {category} --backbone {backbone} first."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Feature Memory Bank k-NN Evaluation | Category: '{category.upper()}'")
    print("=" * 70)
    print(f"Backbone:         {backbone} (ImageNet-1K pretrained)")
    print(f"Bank path:        {resolved_bank_path}")
    print(f"Nearest neighbors: k={n_neighbors}")
    print(f"Device:           {device_obj}")
    print(f"Batch size:       {batch_size}")
    print(f"Smoothing sigma:  {sigma}")
    print(f"Output directory: {out_path}")
    print("-" * 70)

    # 1. Load Feature Memory Bank
    bank = FeatureMemoryBank(device=device_obj)
    bank_metadata = bank.load(resolved_bank_path)
    print(
        f"Loaded memory bank with {len(bank):,} normal patch vectors "
        f"(fused dim = {bank.bank.shape[-1]})."
    )

    # 2. Initialize Pretrained Feature Extractor
    extractor = ResNetExtractor(
        backbone=backbone,
        pretrained=True,
        weights_dir=weights_dir,
        device=device_obj,
    )

    # 3. Load Test DataLoader
    test_loader = get_mvtec_dataloader(
        root_dir=data_dir,
        category=category,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        normalize_imagenet=True,
    )
    total_test_samples = len(test_loader.dataset)
    print(f"Loaded test dataset: {total_test_samples} samples.")

    # 4. Inference loop over test batches
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_maps: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    sample_items: List[Dict] = []

    start_time = time.time()
    pbar = tqdm(
        test_loader,
        desc=f"Inference k-NN ({category})",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    with torch.no_grad():
        for batch in pbar:
            imgs = batch["image"].to(device_obj)
            masks = batch["mask"]  # (B, 1, H, W)
            labels = batch["label"]  # (B,)
            defect_types = batch["defect_type"]

            # Extract multi-scale patch embeddings: (B, 784, 384)
            patches, spatial_shape = extractor.get_patch_embeddings(imgs)

            # Predict smoothed anomaly heatmaps: (B, 1, 224, 224) and image scores: (B,)
            anom_maps, anom_scores = bank.predict_anomaly_maps(
                patches,
                spatial_shape=spatial_shape,
                image_size=(imgs.shape[-2], imgs.shape[-1]),
                n_neighbors=n_neighbors,
                sigma=sigma,
                batch_size=chunk_size,
            )

            all_scores.extend(anom_scores.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_maps.append(anom_maps.cpu())
            all_masks.append(masks.cpu())

            # Collect representative defect samples for visualization
            if len(sample_items) < num_vis_samples:
                for idx in range(imgs.size(0)):
                    if labels[idx].item() == 1 and len(sample_items) < num_vis_samples:
                        sample_items.append({
                            "image": denormalize(imgs[idx], is_imagenet_normalized=True),
                            "mask": masks[idx].squeeze().numpy(),
                            "map": anom_maps[idx].squeeze().cpu().numpy(),
                            "defect_type": defect_types[idx],
                            "score": float(anom_scores[idx].item()),
                        })

            current_rate = len(all_scores) / max(0.001, time.time() - start_time)
            pbar.set_postfix({"samples_done": len(all_scores), "rate": f"{current_rate:.1f} img/s"})

    elapsed_time = time.time() - start_time
    print(f"\nInference completed in {elapsed_time:.2f}s ({len(all_scores) / elapsed_time:.1f} img/s).")

    # 5. Concatenate predictions across all batches
    all_maps_tensor = torch.cat(all_maps, dim=0)
    all_masks_tensor = torch.cat(all_masks, dim=0)

    # 6. Compute Quantitative Metrics
    print("Computing evaluation metrics (Image AUROC, Pixel AUROC, PRO Score)...")
    img_auroc = compute_image_auroc(all_labels, all_scores)
    pix_auroc = compute_pixel_auroc(all_masks_tensor, all_maps_tensor)
    pro = compute_pro_score(all_masks_tensor, all_maps_tensor)

    results_dict = {
        "Category": category,
        "Model": f"kNN-{backbone}",
        "k_Neighbors": n_neighbors,
        "Image AUROC": img_auroc,
        "Pixel AUROC": pix_auroc,
        "PRO Score": pro,
    }

    print("-" * 70)
    print("EVALUATION BENCHMARK SUMMARY:")
    df_metrics = pd.DataFrame([results_dict])
    print(df_metrics.to_string(index=False))
    print("-" * 70)

    # 7. Export Benchmark CSV
    csv_filename = f"{category}_{backbone}_k{n_neighbors}_benchmark.csv"
    csv_save_path = out_path / csv_filename
    df_metrics.to_csv(csv_save_path, index=False)
    print(f"Saved benchmark CSV to: {csv_save_path.resolve()}")

    # 8. Export Qualitative Anomaly Localization Grid
    vis_filename = f"{category}_{backbone}_k{n_neighbors}_localization.png"
    vis_save_path = out_path / vis_filename
    visualize_feature_knn_grid(
        sample_items=sample_items,
        save_path=vis_save_path,
        category=category,
        backbone=backbone,
        n_neighbors=n_neighbors,
    )

    return results_dict


def main():
    args = parse_args()
    print(f"=== [Phase 3] Starting Feature k-NN Evaluation for '{args.category}' ===")
    evaluate_feature_knn(
        category=args.category,
        backbone=args.backbone,
        bank_path=args.bank_path,
        data_dir=args.data_dir,
        n_neighbors=args.n_neighbors,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        sigma=args.sigma,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        num_vis_samples=args.num_vis_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
