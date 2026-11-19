"""
Full Evaluation Pipeline: Multi-Category, Multi-Model Visual Anomaly Detection Benchmark.

Executes a unified, automated benchmark across multiple MVTec AD categories and models:
1. Convolutional Autoencoder (ConvAE) - Reconstruction baseline (L1 + SSIM)
2. Convolutional Variational Autoencoder (ConvVAE) - Probabilistic baseline
3. PatchCore-ResNet18 - SOTA memory bank with 3x3 patch pooling and 1% Coreset
4. PatchCore-ViT - Vision Transformer (vit_base_patch16_224) with intermediate blocks and 1% Coreset

Computes comprehensive quantitative metrics:
- Image AUROC (%)
- Pixel AUROC (%)
- PRO Score (%) at FPR <= 0.3
- Optimal Thresholds: Percentile (95%), Youden's J statistic, and F1-optimal
- Inference Latency (ms/image)
- Memory Footprint (MB)

Exports:
- Benchmark metrics CSV: results/full_evaluation/full_benchmark_summary.csv
- Markdown comparison table: results/full_evaluation/master_comparison_table.md
- Serialized predictions (.pt) per category & method for rapid notebook loading
- Qualitative localization inspection figures (.png) for representative defect types

Usage:
    # Run full benchmark on standard 5 representative categories
    python scripts/full_evaluation.py --categories bottle capsule carpet hazelnut leather

    # Run only PatchCore models across all 15 MVTec AD categories
    python scripts/full_evaluation.py --categories all --methods patchcore_resnet18 patchcore_vit

    # Quick test run on a single category
    python scripts/full_evaluation.py --categories bottle --methods patchcore_resnet18
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import (
    MVTEC_ALL_CATEGORIES,
    MVTEC_OBJECT_CATEGORIES,
    MVTEC_TEXTURE_CATEGORIES,
    get_mvtec_dataloader,
)
from src.data.transforms import get_transforms
from src.models import ConvAutoencoder, ConvVAE, PatchCore
from src.utils import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_score,
    overlay_mask_on_image,
)

SUPPORTED_METHODS = [
    "patchcore_resnet18",
    "patchcore_vit",
    "knn_resnet18",
    "ae",
    "vae",
]

DEFAULT_BENCHMARK_CATEGORIES = [
    "bottle",
    "capsule",
    "carpet",
    "hazelnut",
    "leather",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive Multi-Category Anomaly Detection Benchmark."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_BENCHMARK_CATEGORIES,
        help="List of categories to evaluate, or 'all', 'textures', 'objects'.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["patchcore_resnet18", "patchcore_vit"],
        choices=SUPPORTED_METHODS + ["all"],
        help="Model architectures to evaluate.",
    )
    parser.add_argument(
        "--coreset_ratio",
        type=float,
        default=0.01,
        help="PatchCore coreset subsampling ratio (default: 0.01 for 1%).",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=9,
        help="Number of nearest neighbors for PatchCore density re-weighting.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for dataloaders and inference.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian spatial smoothing sigma for PatchCore heatmaps.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to MVTec AD dataset directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing or caching model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/full_evaluation",
        help="Directory to save summary CSV, markdown reports, and visualizations.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Directory caching pretrained backbone weights.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save raw test scores, maps, and masks to .pt for fast notebook inspection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def resolve_categories(categories_arg: List[str]) -> List[str]:
    """Resolves CLI category arguments into an explicit list of valid MVTec categories."""
    if len(categories_arg) == 1 and categories_arg[0].lower() == "all":
        return list(MVTEC_ALL_CATEGORIES)
    elif len(categories_arg) == 1 and categories_arg[0].lower() == "textures":
        return list(MVTEC_TEXTURE_CATEGORIES)
    elif len(categories_arg) == 1 and categories_arg[0].lower() == "objects":
        return list(MVTEC_OBJECT_CATEGORIES)

    resolved = []
    for cat in categories_arg:
        cat_clean = cat.strip().lower()
        if cat_clean in MVTEC_ALL_CATEGORIES:
            resolved.append(cat_clean)
        else:
            print(f"[Warning] Category '{cat}' is not in MVTec AD dataset. Skipping.")
    return resolved


def resolve_methods(methods_arg: List[str]) -> List[str]:
    """Resolves CLI methods arguments into an explicit list of supported model methods."""
    if len(methods_arg) == 1 and methods_arg[0].lower() == "all":
        return list(SUPPORTED_METHODS)
    return [m for m in methods_arg if m in SUPPORTED_METHODS]


def compute_optimal_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Computes optimal anomaly decision thresholds using 3 distinct strategies:
    1. Percentile-based: 95th percentile on normal test samples.
    2. Youden's J statistic: argmax(TPR - FPR) along the ROC curve.
    3. F1-optimal: threshold maximizing F1-score on precision-recall curve.
    """
    y_true = np.asarray(y_true).astype(int).flatten()
    y_scores = np.asarray(y_scores).astype(float).flatten()

    # Strategy 1: 95th percentile on normal test samples
    normal_scores = y_scores[y_true == 0]
    th_percentile = float(np.percentile(normal_scores, 95)) if len(normal_scores) > 0 else float(np.median(y_scores))

    # Strategy 2: Youden's J statistic (ROC curve)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    youden_index = np.argmax(tpr - fpr)
    th_youden = float(roc_thresholds[youden_index])

    # Strategy 3: F1-optimal threshold (Precision-Recall curve)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = int(np.argmax(f1_scores))
    if best_f1_idx < len(pr_thresholds):
        th_f1 = float(pr_thresholds[best_f1_idx])
        best_f1 = float(f1_scores[best_f1_idx])
    else:
        th_f1 = float(pr_thresholds[-1]) if len(pr_thresholds) > 0 else th_youden
        best_f1 = float(f1_scores[-1]) if len(f1_scores) > 0 else 0.0

    return {
        "th_percentile_95": th_percentile,
        "th_youden_j": th_youden,
        "th_f1_optimal": th_f1,
        "best_f1_score": best_f1,
    }


def export_qualitative_gallery(
    sample_items: List[Dict[str, Any]],
    category: str,
    method: str,
    save_path: Path,
):
    """Generates and exports qualitative localization inspection figures."""
    num_samples = len(sample_items)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    column_headers = [
        "Test Sample",
        "Ground Truth Mask",
        "Predicted Heatmap",
        "Anomaly Overlay",
    ]
    for col_idx, title in enumerate(column_headers):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i in range(num_samples):
        item = sample_items[i]
        img = item["image"]
        mask = item["mask"]
        anom_map = item["map"]

        min_val, max_val = float(anom_map.min()), float(anom_map.max())
        norm_map = (anom_map - min_val) / (max_val - min_val + 1e-8)
        overlay = overlay_mask_on_image(img, norm_map, alpha=0.45, color=(1.0, 0.0, 0.0))

        # Col 0: Image
        axes[i, 0].imshow(img)
        defect_type = item.get("defect_type", "defect")
        score_val = item.get("score", 0.0)
        axes[i, 0].set_ylabel(
            f"{defect_type}\nScore: {score_val:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Col 1: Mask
        axes[i, 1].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].axis("off")

        # Col 2: Heatmap
        axes[i, 2].imshow(norm_map, cmap="jet")
        axes[i, 2].axis("off")

        # Col 3: Overlay
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis("off")

    plt.suptitle(
        f"Localization Benchmark: {category.upper()} | Method: {method.upper()}",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_autoencoder_model(
    category: str,
    model_type: str,
    data_dir: str,
    checkpoint_dir: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Evaluates ConvAE or ConvVAE on the given category if checkpoint exists."""
    ckpt_path = Path(checkpoint_dir) / f"{category}_{model_type}" / "best.pth"
    if not ckpt_path.exists():
        print(f"  [Info] Checkpoint not found for {category} {model_type} at: {ckpt_path}. Skipping.")
        return None, None

    model_cls = ConvAutoencoder if model_type == "ae" else ConvVAE
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    latent_dim = checkpoint.get("latent_dim", 128)
    model = model_cls(in_channels=3, base_channels=32, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Calculate model parameter size in MB
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    test_loader = get_mvtec_dataloader(
        root_dir=data_dir,
        category=category,
        split="test",
        image_size=(256, 256),
        crop_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        normalize_imagenet=False,
    )

    all_scores, all_labels = [], []
    all_maps, all_masks = [], []
    all_defect_types, all_paths = [], []
    sample_items = []

    # Latency timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Inference {model_type.upper()} ({category})", leave=False):
            imgs = batch["image"].to(device)
            masks = batch["mask"]
            labels = batch["label"]

            anom_maps = model.get_anomaly_map(imgs)
            anom_scores = model.get_anomaly_score(imgs)

            all_scores.extend(anom_scores.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_maps.append(anom_maps.cpu())
            all_masks.append(masks)
            all_defect_types.extend(batch["defect_type"])
            all_paths.extend(batch["image_path"])

            # Collect representative samples for qualitative figure
            if len(sample_items) < 6:
                for i in range(imgs.size(0)):
                    if labels[i].item() == 1 and len(sample_items) < 6:
                        sample_items.append({
                            "image": imgs[i].cpu().numpy().transpose(1, 2, 0),
                            "mask": masks[i].squeeze().numpy(),
                            "map": anom_maps[i].squeeze().cpu().numpy(),
                            "defect_type": batch["defect_type"][i],
                            "score": anom_scores[i].item(),
                        })

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    total_test_imgs = len(all_labels)
    latency_ms = (total_time / max(1, total_test_imgs)) * 1000.0

    all_maps_tensor = torch.cat(all_maps, dim=0)
    all_masks_tensor = torch.cat(all_masks, dim=0)

    # Compute metrics
    image_auroc = compute_image_auroc(all_labels, all_scores)
    pixel_auroc = compute_pixel_auroc(all_masks_tensor, all_maps_tensor)
    pro_score = compute_pro_score(all_masks_tensor, all_maps_tensor, max_fpr=0.3, num_thresholds=100)
    thresholds = compute_optimal_thresholds(np.array(all_labels), np.array(all_scores))

    metrics = {
        "category": category,
        "category_type": "texture" if category in MVTEC_TEXTURE_CATEGORIES else "object",
        "method": model_type,
        "backbone": "conv_encoder",
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "pro_score": pro_score,
        "latency_ms": latency_ms,
        "memory_mb": param_size_mb,
        "num_train": 0,
        "num_test": total_test_imgs,
        "num_anomalies": sum(all_labels),
    }

    predictions = {
        "category": category,
        "method": model_type,
        "backbone": "conv_encoder",
        "image_scores": np.array(all_scores),
        "image_labels": np.array(all_labels),
        "defect_types": all_defect_types,
        "image_paths": all_paths,
        "anomaly_maps": all_maps_tensor.squeeze().numpy(),
        "gt_masks": all_masks_tensor.squeeze().numpy(),
        "sample_items": sample_items,
        "metrics": metrics,
        "thresholds": thresholds,
    }

    return metrics, predictions


def evaluate_patchcore_model(
    category: str,
    method: str,
    backbone: str,
    coreset_ratio: float,
    n_neighbors: int,
    sigma: float,
    batch_size: int,
    data_dir: str,
    checkpoint_dir: str,
    weights_dir: str,
    device: torch.device,
    patch_pooling: bool = True,
    reweight: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fits (or loads) PatchCore or Feature k-NN model and evaluates performance on the MVTec test set."""
    is_knn = method.startswith("knn")
    ckpt_dir = Path(checkpoint_dir) / ("feature_bank" if is_knn else "patchcore")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if is_knn:
        ckpt_file = ckpt_dir / f"{category}_{backbone}_bank.pt"
    else:
        ckpt_file = ckpt_dir / f"{category}_{backbone}_c{coreset_ratio:.2f}.pt"

    # 1. Initialize PatchCore / Feature k-NN model
    model = PatchCore(
        backbone=backbone,
        pretrained=True,
        layers=None,
        patch_pooling=patch_pooling,
        pooling_kernel=3,
        coreset_sampling_ratio=coreset_ratio,
        n_neighbors=n_neighbors,
        dimension_reduction=None if is_knn else 128,
        image_size=(224, 224),
        sigma=sigma,
        weights_dir=weights_dir,
        device=device,
    )

    # 2. Setup Dataloaders
    train_loader = get_mvtec_dataloader(
        root_dir=data_dir,
        category=category,
        split="train",
        image_size=(256, 256),
        crop_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        normalize_imagenet=True,
    )
    test_loader = get_mvtec_dataloader(
        root_dir=data_dir,
        category=category,
        split="test",
        image_size=(256, 256),
        crop_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        normalize_imagenet=True,
    )

    # 3. Fit or load memory bank
    if ckpt_file.exists():
        if is_knn:
            print(f"  [Info] Loading existing Feature Memory Bank: {ckpt_file.name}")
            payload = torch.load(ckpt_file, map_location=device, weights_only=False)
            model.memory_bank.bank = payload["bank"].to(device)
            model.metadata = payload.get("metadata", {})
        else:
            print(f"  [Info] Loading existing PatchCore artifact: {ckpt_file.name}")
            model.load(ckpt_file)
    else:
        print(f"  [Info] Fitting memory bank on {len(train_loader.dataset)} normal train images...")
        model.fit(train_loader, verbose=False)
        if not is_knn:
            model.save(ckpt_file)
            print(f"  [Info] Saved trained artifact to: {ckpt_file.name}")

    # Memory footprint: Coreset patch bank size in MB
    coreset_tensor = model.memory_bank.bank
    memory_mb = (coreset_tensor.element_size() * coreset_tensor.nelement()) / (1024 * 1024)

    # 4. Run test set evaluation
    all_scores, all_labels = [], []
    all_maps, all_masks = [], []
    all_defect_types, all_paths = [], []
    sample_items = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Inference {method.upper()} ({category})", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"]
            labels = batch["label"]

            batch_maps, batch_scores = model.predict(images, reweight=reweight)

            all_scores.extend(batch_scores.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_maps.append(batch_maps.cpu())
            all_masks.append(masks)
            all_defect_types.extend(batch["defect_type"])
            all_paths.extend(batch["image_path"])

            # Collect representative samples for qualitative figure
            if len(sample_items) < 6:
                for i in range(images.size(0)):
                    if labels[i].item() == 1 and len(sample_items) < 6:
                        # Denormalize image for visualization
                        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        denorm_img = np.clip(img_np * std + mean, 0.0, 1.0)

                        sample_items.append({
                            "image": denorm_img,
                            "mask": masks[i].squeeze().numpy(),
                            "map": batch_maps[i].squeeze().cpu().numpy(),
                            "defect_type": batch["defect_type"][i],
                            "score": batch_scores[i].item(),
                        })

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    total_test_imgs = len(all_labels)
    latency_ms = (total_time / max(1, total_test_imgs)) * 1000.0

    all_maps_tensor = torch.cat(all_maps, dim=0)
    all_masks_tensor = torch.cat(all_masks, dim=0)

    # 5. Compute metrics
    image_auroc = compute_image_auroc(all_labels, all_scores)
    pixel_auroc = compute_pixel_auroc(all_masks_tensor, all_maps_tensor)
    pro_score = compute_pro_score(all_masks_tensor, all_maps_tensor, max_fpr=0.3, num_thresholds=100)
    thresholds = compute_optimal_thresholds(np.array(all_labels), np.array(all_scores))

    metrics = {
        "category": category,
        "category_type": "texture" if category in MVTEC_TEXTURE_CATEGORIES else "object",
        "method": method,
        "backbone": backbone,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "pro_score": pro_score,
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "num_train": len(train_loader.dataset),
        "num_test": total_test_imgs,
        "num_anomalies": sum(all_labels),
    }

    predictions = {
        "category": category,
        "method": method,
        "backbone": backbone,
        "image_scores": np.array(all_scores),
        "image_labels": np.array(all_labels),
        "defect_types": all_defect_types,
        "image_paths": all_paths,
        "anomaly_maps": all_maps_tensor.squeeze().numpy(),
        "gt_masks": all_masks_tensor.squeeze().numpy(),
        "sample_items": sample_items,
        "metrics": metrics,
        "thresholds": thresholds,
    }

    return metrics, predictions


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generates a clean markdown table string from benchmark metrics DataFrame."""
    lines = [
        "| Category | Type | Method | Backbone | Image AUROC | Pixel AUROC | PRO Score | Latency (ms) | Memory (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| `{row['category']}` | {row['category_type']} | {row['method']} | `{row['backbone']}` | "
            f"**{row['image_auroc'] * 100:.2f}%** | {row['pixel_auroc'] * 100:.2f}% | "
            f"{row['pro_score'] * 100:.2f}% | {row['latency_ms']:.1f} ms | {row['memory_mb']:.2f} MB |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    device = torch.device(args.device)
    categories = resolve_categories(args.categories)
    methods = resolve_methods(args.methods)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Multi-Category Visual Anomaly Detection Benchmark (Week 5 Full Evaluation)")
    print("=" * 78)
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"Methods    ({len(methods)}): {', '.join(methods)}")
    print(f"Device:          {device}")
    print(f"Coreset ratio:   {args.coreset_ratio:.2%}")
    print(f"Output dir:      {out_dir}")
    print("=" * 78)

    benchmark_rows = []

    # Outer progress bar over Categories
    cat_pbar = tqdm(categories, desc="Overall Progress (Categories)", unit="category", position=0)

    for category in cat_pbar:
        cat_pbar.set_postfix({"Current": category})

        # Inner loop over Methods
        for method in methods:
            print(f"\n---> Benchmarking Category: '{category.upper()}' | Method: '{method}'")

            metrics = None
            predictions = None

            if method in ["ae", "vae"]:
                metrics, predictions = evaluate_autoencoder_model(
                    category=category,
                    model_type=method,
                    data_dir=args.data_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    batch_size=args.batch_size,
                    device=device,
                )
            elif method == "knn_resnet18":
                metrics, predictions = evaluate_patchcore_model(
                    category=category,
                    method=method,
                    backbone="resnet18",
                    coreset_ratio=1.0,
                    n_neighbors=3,
                    sigma=args.sigma,
                    batch_size=args.batch_size,
                    data_dir=args.data_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    weights_dir=args.weights_dir,
                    device=device,
                    patch_pooling=False,
                    reweight=False,
                )
            elif method == "patchcore_resnet18":
                metrics, predictions = evaluate_patchcore_model(
                    category=category,
                    method=method,
                    backbone="resnet18",
                    coreset_ratio=args.coreset_ratio,
                    n_neighbors=args.n_neighbors,
                    sigma=args.sigma,
                    batch_size=args.batch_size,
                    data_dir=args.data_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    weights_dir=args.weights_dir,
                    device=device,
                )
            elif method == "patchcore_vit":
                metrics, predictions = evaluate_patchcore_model(
                    category=category,
                    method=method,
                    backbone="vit_base_patch16_224",
                    coreset_ratio=args.coreset_ratio,
                    n_neighbors=args.n_neighbors,
                    sigma=args.sigma,
                    batch_size=args.batch_size,
                    data_dir=args.data_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    weights_dir=args.weights_dir,
                    device=device,
                )

            if metrics is not None and predictions is not None:
                benchmark_rows.append(metrics)

                # Save raw predictions (.pt) for notebook analysis
                if args.save_predictions:
                    pred_file = preds_dir / f"pred_{category}_{method}.pt"
                    torch.save(predictions, pred_file)

                # Export qualitative localization gallery
                if len(predictions.get("sample_items", [])) > 0:
                    vis_file = vis_dir / f"{category}_{method}_localization.png"
                    export_qualitative_gallery(
                        sample_items=predictions["sample_items"],
                        category=category,
                        method=method,
                        save_path=vis_file,
                    )

                # Print summary line
                print(
                    f"     [Result] Image AUROC: {metrics['image_auroc'] * 100:.2f}% | "
                    f"Pixel AUROC: {metrics['pixel_auroc'] * 100:.2f}% | "
                    f"PRO: {metrics['pro_score'] * 100:.2f}% | "
                    f"Latency: {metrics['latency_ms']:.1f}ms"
                )

    # Compile and export summary results
    if len(benchmark_rows) > 0:
        df = pd.DataFrame(benchmark_rows)
        csv_path = out_dir / "full_benchmark_summary.csv"
        df.to_csv(csv_path, index=False)

        md_content = generate_markdown_table(df)
        md_path = out_dir / "master_comparison_table.md"
        md_path.write_text(md_content, encoding="utf-8")

        print("\n" + "=" * 78)
        print("FULL BENCHMARK EXECUTION SUMMARY")
        print("=" * 78)
        print(df[["category", "method", "backbone", "image_auroc", "pixel_auroc", "pro_score", "latency_ms"]].to_string(index=False))
        print("=" * 78)
        print(f"Summary CSV exported to:         {csv_path.resolve()}")
        print(f"Markdown report exported to:    {md_path.resolve()}")
        print(f"Predictions saved to:           {preds_dir.resolve()}")
        print(f"Visualizations saved to:        {vis_dir.resolve()}")
        print("=" * 78)
    else:
        print("[Warning] No benchmark evaluations were completed successfully.")


if __name__ == "__main__":
    main()

