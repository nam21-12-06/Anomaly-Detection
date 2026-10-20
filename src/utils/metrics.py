from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from sklearn.metrics import roc_auc_score
import torch


def to_numpy(data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
    """
    Converts input data (Tensor, list, or ndarray) to a contiguous NumPy array.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def compute_image_auroc(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    scores: Union[np.ndarray, torch.Tensor, List[float]],
) -> float:
    """
    Computes Image-level Area Under the Receiver Operating Characteristic Curve (AUROC).

    Args:
        labels: 1D array of binary ground-truth labels (0 for normal, 1 for anomaly).
        scores: 1D array of predicted anomaly scores (higher indicates anomaly).

    Returns:
        AUROC score in range [0.0, 1.0].
    """
    y_true = to_numpy(labels).flatten()
    y_scores = to_numpy(scores).flatten()

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        raise ValueError(
            f"AUROC requires both normal and anomaly classes, but only found classes: {unique_labels}"
        )

    return float(roc_auc_score(y_true, y_scores))


def compute_pixel_auroc(
    gt_masks: Union[np.ndarray, torch.Tensor, List],
    anomaly_maps: Union[np.ndarray, torch.Tensor, List],
) -> float:
    """
    Computes Pixel-level AUROC for anomaly localization across test samples.

    Args:
        gt_masks: Binary ground-truth masks of shape (N, H, W) or (N, 1, H, W).
        anomaly_maps: Predicted continuous anomaly heatmaps matching gt_masks dimensions.

    Returns:
        Pixel AUROC score in range [0.0, 1.0].
    """
    y_true = (to_numpy(gt_masks) > 0.5).astype(np.uint8).flatten()
    y_scores = to_numpy(anomaly_maps).astype(np.float32).flatten()

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Pixel AUROC requires both background and defect pixels, but found classes: {unique_labels}"
        )

    return float(roc_auc_score(y_true, y_scores))


def compute_pro_score(
    gt_masks: Union[np.ndarray, torch.Tensor, List],
    anomaly_maps: Union[np.ndarray, torch.Tensor, List],
    max_fpr: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """
    Computes the Per-Region Overlap (PRO) score.

    Evaluates the average proportion of overlapping pixels for each connected defect
    component individually, integrated up to a maximum False Positive Rate (FPR).

    Args:
        gt_masks: Binary ground-truth masks of shape (N, H, W) or (N, 1, H, W).
        anomaly_maps: Continuous anomaly heatmaps of shape (N, H, W) or (N, 1, H, W).
        max_fpr: Maximum False Positive Rate limit for integration (default 0.3).
        num_thresholds: Number of discrete evaluation thresholds.

    Returns:
        Normalized PRO score in range [0.0, 1.0].
    """
    masks = to_numpy(gt_masks)
    maps = to_numpy(anomaly_maps)

    # Normalize shapes to (N, H, W)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps.squeeze(1)

    masks = (masks > 0.5).astype(np.uint8)
    maps = maps.astype(np.float32)

    # Identify connected components in defect ground truth masks
    regions = []
    for i in range(len(masks)):
        if masks[i].max() > 0:
            labeled_mask, num_features = ndimage.label(masks[i])
            for feat_idx in range(1, num_features + 1):
                region_coords = (i, *(labeled_mask == feat_idx).nonzero())
                region_size = np.count_nonzero(labeled_mask == feat_idx)
                regions.append((region_coords, region_size))

    if len(regions) == 0:
        raise ValueError("No defective connected components found in ground truth masks.")

    total_normal_pixels = np.count_nonzero(masks == 0)
    if total_normal_pixels == 0:
        raise ValueError("No normal pixels found for False Positive Rate calculation.")

    # Select threshold range based on anomaly map percentiles
    min_val = float(maps.min())
    max_val = float(maps.max())
    if min_val == max_val:
        return 0.5

    thresholds = np.linspace(min_val, max_val, num_thresholds)

    fpr_list = []
    pro_list = []

    for th in thresholds:
        bin_pred = maps >= th

        # False Positive Rate on normal background pixels
        fp_count = np.count_nonzero(bin_pred & (masks == 0))
        fpr = fp_count / total_normal_pixels

        # Per-region overlap (average recall across connected components)
        region_overlaps = []
        for (img_idx, r_rows, r_cols), r_size in regions:
            overlap = np.count_nonzero(bin_pred[img_idx, r_rows, r_cols])
            region_overlaps.append(overlap / r_size)

        pro = float(np.mean(region_overlaps))

        fpr_list.append(fpr)
        pro_list.append(pro)

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)

    # Filter by max_fpr
    df = pd.DataFrame({"fpr": fpr_arr, "pro": pro_arr})
    df = df[df["fpr"] <= max_fpr]

    if len(df) == 0:
        return 0.0

    # For duplicate FPR values, take the maximum recall (PRO) achieved
    grouped = df.groupby("fpr")["pro"].max().reset_index()

    # Ensure left boundary (fpr=0.0)
    if grouped["fpr"].min() > 0.0:
        grouped = pd.concat([pd.DataFrame([{"fpr": 0.0, "pro": 0.0}]), grouped], ignore_index=True)

    # Ensure right boundary (fpr=max_fpr)
    if grouped["fpr"].max() < max_fpr:
        last_pro = float(grouped["pro"].iloc[-1])
        grouped = pd.concat([grouped, pd.DataFrame([{"fpr": max_fpr, "pro": last_pro}])], ignore_index=True)

    # Sort ascending by FPR
    grouped = grouped.sort_values("fpr").reset_index(drop=True)

    # Trapezoidal integration normalized by max_fpr
    integral = np.trapezoid(grouped["pro"].values, grouped["fpr"].values)
    normalized_pro = float(integral / max_fpr)

    return float(np.clip(normalized_pro, 0.0, 1.0))


def evaluate_predictions(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    scores: Union[np.ndarray, torch.Tensor, List[float]],
    gt_masks: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    anomaly_maps: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    compute_pro: bool = True,
) -> Dict[str, float]:
    """
    Convenience wrapper to compute Image AUROC, Pixel AUROC, and PRO metrics.

    Returns:
        Dictionary containing available metric scores.
    """
    results = {}

    # 1. Image-level AUROC
    results["image_auroc"] = compute_image_auroc(labels, scores)

    # 2. Pixel-level metrics if masks and maps are provided
    if gt_masks is not None and anomaly_maps is not None:
        results["pixel_auroc"] = compute_pixel_auroc(gt_masks, anomaly_maps)

        if compute_pro:
            try:
                results["pro_score"] = compute_pro_score(gt_masks, anomaly_maps)
            except Exception as e:
                results["pro_score"] = 0.0
                results["pro_error"] = str(e)

    return results
