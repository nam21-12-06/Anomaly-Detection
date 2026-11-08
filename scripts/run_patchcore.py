"""
Run PatchCore: SOTA Industrial Visual Anomaly Detection & Localization.

Implements the complete PatchCore pipeline (Roth et al., CVPR 2022):
1. Normal training feature extraction with local neighborhood patch pooling (3x3).
2. Greedy Farthest-Point Coreset Subsampling (reducing memory bank to 1-10%).
3. Patch-level nearest-neighbor distance scoring & softmax neighborhood density re-weighting.
4. Evaluation of Image AUROC, Pixel AUROC, and PRO score on MVTec AD test set.
5. Export of benchmark metrics to CSV and qualitative anomaly localization plots.

Usage:
    # Run PatchCore on 'bottle' with ResNet-18 (1% Coreset)
    python scripts/run_patchcore.py --category bottle --backbone resnet18 --coreset_ratio 0.01

    # Run PatchCore with WideResNet-50-2 (SOTA paper configuration)
    python scripts/run_patchcore.py --category bottle --backbone wide_resnet50_2 --coreset_ratio 0.01

    # Evaluate using an existing saved PatchCore model artifact
    python scripts/run_patchcore.py --category bottle --load_checkpoint checkpoints/patchcore/bottle_resnet18_c0.01.pt
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
from src.models import PatchCore
from src.utils import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_score,
    denormalize,
    overlay_mask_on_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PatchCore Visual Anomaly Detection pipeline.")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="MVTec AD category name (e.g. bottle, cable, carpet, tile).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "wide_resnet50_2"],
        help="Torchvision CNN backbone architecture.",
    )
    parser.add_argument(
        "--coreset_ratio",
        type=float,
        default=0.01,
        help="Fraction of memory bank patch embeddings to retain in coreset (e.g. 0.01 for 1%).",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=9,
        help="Number of nearest neighbors for patch anomaly scoring and density re-weighting.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for feature extraction and test evaluation dataloaders.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian filter sigma for anomaly heatmap spatial smoothing.",
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
        default="checkpoints/patchcore",
        help="Directory to save/load trained PatchCore model artifacts.",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to existing saved PatchCore checkpoint (.pt). If provided, skips training fit.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/patchcore",
        help="Directory to save benchmark metrics CSV and localization visualizations.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Directory to cache pretrained backbone weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for execution ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def export_qualitative_figure(
    sample_items: List[Dict],
    category: str,
    backbone: str,
    coreset_ratio: float,
    save_path: Path,
):
    """Exports side-by-side localization figures (Test Image, Ground Truth, PatchCore Heatmap, Overlay)."""
    num_samples = len(sample_items)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Test Sample",
        "Ground Truth Mask",
        "PatchCore Heatmap",
        "Anomaly Overlay",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i in range(num_samples):
        item = sample_items[i]
        img = item["image"]
        mask = item["mask"]
        anom_map = item["map"]

        min_v, max_v = float(anom_map.min()), float(anom_map.max())
        norm_map = (anom_map - min_v) / (max_v - min_v + 1e-8)
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
        f"PatchCore Localization: {category.upper()} ({backbone.upper()}, Coreset={coreset_ratio:.1%})",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved qualitative localization plot to: {save_path.resolve()}")
    plt.close()


def run_patchcore(
    category: str = "bottle",
    backbone: str = "resnet18",
    coreset_ratio: float = 0.01,
    n_neighbors: int = 9,
    batch_size: int = 16,
    sigma: float = 4.0,
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints/patchcore",
    load_checkpoint: Optional[str] = None,
    output_dir: str = "results/patchcore",
    weights_dir: str = "weights",
    device: str = "cuda",
) -> Dict[str, float]:
    """Runs end-to-end PatchCore training and evaluation pipeline."""
    device_obj = torch.device(device)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = (
        Path(load_checkpoint)
        if load_checkpoint is not None
        else ckpt_dir / f"{category}_{backbone}_c{coreset_ratio:.2f}.pt"
    )

    print("=" * 72)
    print(f"PatchCore Anomaly Detection | Category: '{category.upper()}'")
    print("=" * 72)
    print(f"Backbone:          {backbone} (ImageNet-1K pretrained)")
    print(f"Neighborhood pool: 3x3 average pooling (Patch-Level Context)")
    print(f"Coreset ratio:     {coreset_ratio:.2%} (Greedy Farthest-Point)")
    print(f"Nearest neighbors: k={n_neighbors} (Softmax density re-weighting)")
    print(f"Device:            {device_obj}")
    print(f"Checkpoint file:   {ckpt_file}")
    print(f"Output directory:  {out_dir}")
    print("-" * 72)

    # 1. Initialize PatchCore model
    model = PatchCore(
        backbone=backbone,
        pretrained=True,
        layers=("layer2", "layer3"),
        patch_pooling=True,
        pooling_kernel=3,
        coreset_sampling_ratio=coreset_ratio,
        n_neighbors=n_neighbors,
        dimension_reduction=128,
        image_size=(224, 224),
        sigma=sigma,
        weights_dir=weights_dir,
        device=device_obj,
    )

    # 2. Fit model or load existing checkpoint
    if ckpt_file.exists():
        print(f"Loading existing PatchCore checkpoint: {ckpt_file}")
        model.load(ckpt_file)
        print(f"Loaded memory bank with {len(model.memory_bank):,} coreset vectors.")
    else:
        print(f"\nStep 1: Training feature extraction & Coreset Subsampling...")
        train_loader = get_mvtec_dataloader(
            root_dir=data_dir,
            category=category,
            split="train",
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            normalize_imagenet=True,
        )
        fit_meta = model.fit(train_loader, verbose=True, random_state=42)
        model.save(ckpt_file)
        print(f"Saved PatchCore model artifact to: {ckpt_file.resolve()}")

    # 3. Test evaluation
    print(f"\nStep 2: Evaluating on test set...")
    test_loader = get_mvtec_dataloader(
        root_dir=data_dir,
        category=category,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        normalize_imagenet=True,
    )

    all_scores: List[float] = []
    all_labels: List[int] = []
    all_maps: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    sample_items: List[Dict] = []

    t_eval_start = time.time()
    for batch in tqdm(test_loader, desc=f"PatchCore Inference ({category})"):
        imgs = batch["image"]
        masks = batch["mask"]
        labels = batch["label"]
        defect_types = batch["defect_type"]

        anom_maps, img_scores = model.predict(imgs, reweight=True, batch_size=128)

        all_scores.extend(img_scores.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_maps.append(anom_maps.cpu())
        all_masks.append(masks.cpu())

        # Collect qualitative visualization items
        for j in range(imgs.size(0)):
            if labels[j].item() == 1 and len(sample_items) < 8:
                orig_img = denormalize(imgs[j])
                gt_mask = masks[j].squeeze().cpu().numpy()
                anom_map_np = anom_maps[j].squeeze().cpu().numpy()
                sample_items.append({
                    "image": orig_img,
                    "mask": gt_mask,
                    "map": anom_map_np,
                    "defect_type": defect_types[j],
                    "score": img_scores[j].item(),
                })

    t_eval_total = time.time() - t_eval_start
    total_test_images = len(all_labels)
    latency_per_img_ms = (t_eval_total / max(1, total_test_images)) * 1000.0

    all_maps_tensor = torch.cat(all_maps, dim=0)
    all_masks_tensor = torch.cat(all_masks, dim=0)

    # 4. Compute quantitative benchmark metrics
    image_auroc = compute_image_auroc(all_labels, all_scores)
    pixel_auroc = compute_pixel_auroc(all_masks_tensor, all_maps_tensor)
    pro_score = compute_pro_score(all_masks_tensor, all_maps_tensor, max_fpr=0.3)

    print("\n" + "=" * 72)
    print(f"PatchCore Quantitative Results: {category.upper()}")
    print("=" * 72)
    print(f"Image AUROC:        {image_auroc * 100:.2f}%")
    print(f"Pixel AUROC:        {pixel_auroc * 100:.2f}%")
    print(f"PRO Score (FPR<=0.3): {pro_score * 100:.2f}%")
    print(f"Coreset Size:       {len(model.memory_bank):,} patches")
    print(f"Total Test Images:  {total_test_images}")
    print(f"Evaluation Latency: {t_eval_total:.2f}s ({latency_per_img_ms:.1f} ms / image)")
    print("=" * 72)

    # 5. Export benchmark CSV
    csv_path = out_dir / f"{category}_{backbone}_c{coreset_ratio:.2f}_benchmark.csv"
    results_df = pd.DataFrame([{
        "Category": category,
        "Method": "PatchCore",
        "Backbone": backbone,
        "Coreset_Ratio": coreset_ratio,
        "Coreset_Size": len(model.memory_bank),
        "k_Neighbors": n_neighbors,
        "Image_AUROC": image_auroc,
        "Pixel_AUROC": pixel_auroc,
        "PRO_Score": pro_score,
        "Latency_ms_per_image": latency_per_img_ms,
    }])
    results_df.to_csv(csv_path, index=False)
    print(f"Exported benchmark metrics CSV to: {csv_path.resolve()}")

    # 6. Export qualitative localization figure
    fig_path = out_dir / f"{category}_{backbone}_c{coreset_ratio:.2f}_localization.png"
    export_qualitative_figure(
        sample_items=sample_items,
        category=category,
        backbone=backbone,
        coreset_ratio=coreset_ratio,
        save_path=fig_path,
    )

    return {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "pro_score": pro_score,
        "latency_ms": latency_per_img_ms,
    }


if __name__ == "__main__":
    args = parse_args()
    run_patchcore(
        category=args.category,
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        n_neighbors=args.n_neighbors,
        batch_size=args.batch_size,
        sigma=args.sigma,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=args.load_checkpoint,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        device=args.device,
    )
