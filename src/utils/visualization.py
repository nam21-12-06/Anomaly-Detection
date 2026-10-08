from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor, is_imagenet_normalized: bool = True) -> np.ndarray:
    """
    Converts a PyTorch tensor (C, H, W) to a displayable NumPy image (H, W, C) in [0, 1].
    """
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if is_imagenet_normalized:
        img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Overlays a binary mask onto an RGB image.
    Args:
        image: RGB float array in [0, 1], shape (H, W, 3).
        mask: Binary mask in [0, 1], shape (H, W).
        alpha: Transparency of the mask overlay.
        color: RGB color of overlay (default red).
    """
    overlay = image.copy()
    mask_bool = mask > 0.5
    for c in range(3):
        overlay[..., c] = np.where(
            mask_bool,
            (1.0 - alpha) * image[..., c] + alpha * color[c],
            image[..., c],
        )
    return np.clip(overlay, 0.0, 1.0)


def plot_defect_gallery(
    dataset,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots representative examples of each defect type found in a dataset instance,
    displaying Original Image, Ground-Truth Mask, and Overlay side-by-side.
    """
    defect_dict = {}
    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        dtype = sample["defect_type"]
        if dtype not in defect_dict:
            defect_dict[dtype] = idx

    num_types = len(defect_dict)
    fig, axes = plt.subplots(num_types, 3, figsize=(10, 3.5 * num_types))
    if num_types == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (dtype, sample_idx) in enumerate(defect_dict.items()):
        item = dataset[sample_idx]
        img = denormalize(item["image"])
        mask = item["mask"].squeeze().cpu().numpy()
        overlay = overlay_mask_on_image(img, mask)

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"Type: {dtype} (Label={item['label']})", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(mask, cmap="gray")
        axes[row_idx, 1].set_title("Ground-Truth Mask", fontsize=11)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Overlay", fontsize=11)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_dataset_distribution(
    df_summary: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Renders publication-ready horizontal stacked bar charts showing sample distribution
    (Train Normal vs Test Normal vs Test Defect) and defect category diversity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True, gridspec_kw={"width_ratios": [3, 1.2]})

    categories = df_summary["Category"].tolist()
    y_pos = np.arange(len(categories))

    train_vals = df_summary["Train Normal"].values
    test_normal_vals = df_summary["Test Normal"].values
    test_defect_vals = df_summary["Test Defect"].values

    ax1.barh(y_pos, train_vals, color="#2b5c8f", label="Train (Normal)", height=0.65)
    ax1.barh(y_pos, test_normal_vals, left=train_vals, color="#529b5c", label="Test (Normal)", height=0.65)
    ax1.barh(
        y_pos,
        test_defect_vals,
        left=train_vals + test_normal_vals,
        color="#c94c4c",
        label="Test (Defect)",
        height=0.65,
    )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories, fontsize=11)
    ax1.set_xlabel("Image Count", fontsize=12)
    ax1.set_title(
        "Sample Distribution Across Categories (Train Normal vs. Test Splits)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.invert_yaxis()

    for i, total in enumerate(df_summary["Total Images"].values):
        ax1.text(total + 5, i, f"{total}", va="center", fontsize=9, color="#444444")

    defect_classes = df_summary["Defect Classes"].values
    ax2.barh(y_pos, defect_classes, color="#e08244", height=0.65)
    ax2.set_xlabel("Number of Defect Types", fontsize=12)
    ax2.set_title("Defect Class Diversity", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(0, max(defect_classes) + 2)

    for i, count in enumerate(defect_classes):
        ax2.text(count + 0.2, i, f"{count}", va="center", fontsize=9, color="#444444")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_defect_area_distribution(
    df_masks: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Renders a KDE distribution and boxplot illustrating defect surface area coverage
    ratios (% of image pixels) across categories.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(
        data=df_masks,
        x="Area Ratio (%)",
        hue="Type",
        palette={"Object": "#2b5c8f", "Texture": "#e08244"},
        log_scale=True,
        kde=True,
        bins=30,
        ax=ax1,
        element="step",
    )
    ax1.axvline(1.0, color="#c94c4c", linestyle="--", linewidth=1.5, label="1.0% Image Surface Threshold")
    ax1.set_title("Log-Scale Distribution of Defect Area Ratios (%)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Defect Area Coverage (% of Total Image Pixels, Log Scale)", fontsize=11)
    ax1.set_ylabel("Count of Defect Masks", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)

    order = df_masks.groupby("Category")["Area Ratio (%)"].median().sort_values().index
    sns.boxplot(
        data=df_masks,
        x="Category",
        y="Area Ratio (%)",
        hue="Category",
        order=order,
        palette="vlag",
        legend=False,
        ax=ax2,
        showfliers=False,
    )
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_title("Defect Area Percentage by Category (Median Sorted)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Area Ratio (% of Pixels)", fontsize=11)
    ax2.set_xlabel("")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_multi_category_gallery(
    data_dir: Union[str, Path] = "./data",
    categories: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Displays a representative defect sample for a list of categories, showing
    Original Test Image, Ground-Truth Mask, and Translucent Overlay.
    """
    from src.data.mvtec import MVTecDataset

    if categories is None:
        categories = ["bottle", "cable", "metal_nut", "carpet", "tile", "wood"]

    fig, axes = plt.subplots(len(categories), 3, figsize=(11, 3.2 * len(categories)))

    for row_idx, cat in enumerate(categories):
        test_ds = MVTecDataset(root_dir=data_dir, category=cat, split="test")

        # Pick first defect sample
        sample_idx = next(i for i, s in enumerate(test_ds.samples) if s["label"] == 1)
        item = test_ds[sample_idx]

        img = denormalize(item["image"])
        mask = item["mask"].squeeze().cpu().numpy()
        overlay = overlay_mask_on_image(img, mask, alpha=0.45)

        dtype = item["defect_type"]
        area_pct = float((mask > 0.5).mean() * 100.0)

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"{cat.upper()} | Type: {dtype}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(mask, cmap="gray")
        axes[row_idx, 1].set_title(f"GT Mask (Area: {area_pct:.2f}%)", fontsize=11)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Defect Localization Overlay", fontsize=11)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def inspect_category(
    data_dir: Union[str, Path] = "./data",
    category_name: str = "bottle",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Inspects all unique defect types within a specified category alongside normal sample.
    """
    from src.data.mvtec import MVTecDataset

    test_ds = MVTecDataset(root_dir=data_dir, category=category_name, split="test")

    defect_types_map = {}
    for idx, s in enumerate(test_ds.samples):
        dtype = s["defect_type"]
        if dtype not in defect_types_map:
            defect_types_map[dtype] = idx

    n_rows = len(defect_types_map)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (dtype, sample_idx) in enumerate(defect_types_map.items()):
        item = test_ds[sample_idx]
        img = denormalize(item["image"])
        mask = item["mask"].squeeze().cpu().numpy()
        overlay = overlay_mask_on_image(img, mask, alpha=0.45)

        status_str = "Normal" if item["label"] == 0 else "Anomaly"
        area_str = f"{((mask > 0.5).mean() * 100):.2f}%" if item["label"] == 1 else "0.00%"

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"[{status_str}] {dtype}", fontsize=11, fontweight="bold")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(mask, cmap="gray")
        axes[r, 1].set_title(f"Ground-Truth Mask ({area_str})", fontsize=11)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title("Defect Overlay", fontsize=11)
        axes[r, 2].axis("off")

    plt.suptitle(
        f"Defect Taxonomy Inspection: '{category_name.upper()}'",
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
