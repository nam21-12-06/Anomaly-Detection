import numpy as np
import pytest
import torch

from src.utils.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_score,
    evaluate_predictions,
)


def test_image_auroc_perfect():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.15, 0.8, 0.9, 0.85]
    auroc = compute_image_auroc(labels, scores)
    assert auroc == 1.0


def test_image_auroc_inverted():
    labels = [0, 0, 1, 1]
    scores = [0.8, 0.9, 0.1, 0.2]
    auroc = compute_image_auroc(labels, scores)
    assert auroc == 0.0


def test_image_auroc_single_class_error():
    labels = [0, 0, 0]
    scores = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="AUROC requires both normal and anomaly classes"):
        compute_image_auroc(labels, scores)


def test_pixel_auroc_perfect_and_partial():
    # 2 images of size 10x10
    gt_masks = np.zeros((2, 10, 10), dtype=np.uint8)
    anomaly_maps = np.zeros((2, 10, 10), dtype=np.float32)

    # Put defect in corner of image 1
    gt_masks[0, :3, :3] = 1
    anomaly_maps[0, :3, :3] = 0.95  # high anomaly values on defect
    anomaly_maps[0, 3:, 3:] = 0.05

    pixel_auc = compute_pixel_auroc(gt_masks, anomaly_maps)
    assert pixel_auc > 0.99


def test_pro_score_calculation():
    # 2 images of size 20x20
    gt_masks = np.zeros((2, 20, 20), dtype=np.uint8)
    anomaly_maps = np.zeros((2, 20, 20), dtype=np.float32)

    # Component 1 on image 0
    gt_masks[0, 2:6, 2:6] = 1
    anomaly_maps[0, 2:6, 2:6] = 0.9

    # Component 2 on image 1
    gt_masks[1, 10:14, 10:14] = 1
    anomaly_maps[1, 10:14, 10:14] = 0.85

    # Background has near zero noise
    anomaly_maps[anomaly_maps == 0] = 0.01

    pro_score = compute_pro_score(gt_masks, anomaly_maps, max_fpr=0.3)
    assert 0.0 <= pro_score <= 1.0
    assert pro_score > 0.8, f"Expected high PRO score, got {pro_score}"


def test_evaluate_predictions_wrapper():
    labels = [0, 1]
    scores = [0.1, 0.9]

    gt_masks = np.zeros((2, 10, 10), dtype=np.uint8)
    gt_masks[1, 2:5, 2:5] = 1

    anomaly_maps = np.zeros((2, 10, 10), dtype=np.float32)
    anomaly_maps[1, 2:5, 2:5] = 0.9

    results = evaluate_predictions(labels, scores, gt_masks, anomaly_maps, compute_pro=True)

    assert "image_auroc" in results
    assert "pixel_auroc" in results
    assert "pro_score" in results
    assert results["image_auroc"] == 1.0
    assert results["pixel_auroc"] > 0.9


def test_torch_tensor_compatibility():
    labels_tensor = torch.tensor([0, 0, 1, 1])
    scores_tensor = torch.tensor([0.1, 0.2, 0.8, 0.9])

    auroc = compute_image_auroc(labels_tensor, scores_tensor)
    assert auroc == 1.0

