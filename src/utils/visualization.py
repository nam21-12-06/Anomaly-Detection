from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
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
            (1 - alpha) * image[..., c] + alpha * color[c],
            image[..., c],
        )
    return np.clip(overlay, 0.0, 1.0)


def plot_defect_gallery(
    dataset,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots representative examples of each defect type found in the dataset,
    displaying Original Image, Ground-Truth Mask, and Overlay side-by-side.
    """
    # Group one index per defect type
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

        # Col 1: Original Image
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"Type: {dtype} (Label={item['label']})", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 2: Ground Truth Mask
        axes[row_idx, 1].imshow(mask, cmap="gray")
        axes[row_idx, 1].set_title("Ground-Truth Mask", fontsize=11)
        axes[row_idx, 1].axis("off")

        # Col 3: Overlay
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Overlay", fontsize=11)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved gallery plot to: {save_path}")

    if show:
        plt.show()
    plt.close()
