from pathlib import Path
import tempfile
import pytest
import torch

from src.models import FeatureMemoryBank, interpolate_and_smooth_anomaly_maps
from src.utils import compute_image_auroc, compute_pixel_auroc, compute_pro_score


def test_knn_synthetic_evaluation_pipeline():
    """
    Validates end-to-end mathematical consistency of the k-NN anomaly scoring
    and evaluation metrics on synthetic representations without heavy computation.
    """
    # 1. Simulate a normal memory bank: 200 normal patch vectors of dim 64
    torch.manual_seed(42)
    bank = FeatureMemoryBank(device="cpu")
    normal_bank_features = torch.randn(200, 64)
    bank.fit(normal_bank_features)
    assert len(bank) == 200

    # 2. Simulate 4 test images (2 normal, 2 defective), each with 4 patches (2x2 grid)
    # Normal test patches sampled from same distribution as bank
    normal_patches = torch.randn(2, 4, 64)
    # Defective test patches have huge positive offset
    defect_patches = torch.randn(2, 4, 64) + 15.0

    test_patches = torch.cat([normal_patches, defect_patches], dim=0)  # (4, 4, 64)
    labels = [0, 0, 1, 1]

    # 3. Predict anomaly maps and image scores
    anomaly_maps, image_scores = bank.predict_anomaly_maps(
        test_patches,
        spatial_shape=(2, 2),
        image_size=(32, 32),
        n_neighbors=1,
        sigma=1.0,
    )

    assert anomaly_maps.shape == (4, 1, 32, 32)
    assert image_scores.shape == (4,)

    # Defect image scores must be substantially higher than normal image scores
    assert image_scores[2] > image_scores[0]
    assert image_scores[3] > image_scores[1]

    # 4. Image AUROC must be perfect (1.0)
    auroc = compute_image_auroc(labels, image_scores.tolist())
    assert auroc == 1.0

    # 5. Create synthetic ground truth masks: normal = 0, defect = square in center
    masks = torch.zeros(4, 1, 32, 32)
    masks[2, 0, 10:22, 10:22] = 1.0
    masks[3, 0, 10:22, 10:22] = 1.0

    pix_auroc = compute_pixel_auroc(masks, anomaly_maps)
    assert 0.0 <= pix_auroc <= 1.0

    pro = compute_pro_score(masks, anomaly_maps)
    assert 0.0 <= pro <= 1.0

