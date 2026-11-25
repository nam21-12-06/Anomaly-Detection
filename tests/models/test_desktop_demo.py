"""
Automated Unit Tests for Industrial AOI Desktop Application (PyQt6).
"""

from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import torch

from app.desktop_demo import (
    CALIBRATED_THRESHOLDS,
    CATEGORIES,
    METHOD_NAMES,
    ModelManager,
    AOIStationWindow,
    DEVICE,
)

from PyQt6.QtWidgets import QApplication
import sys

# Ensure single QApplication instance across tests
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_calibrated_thresholds_coverage():
    """Verifies that all 15 MVTec AD categories have calibrated PatchCore thresholds."""
    for cat in CATEGORIES:
        key_resnet = f"{cat}:patchcore_resnet18"
        key_vit = f"{cat}:patchcore_vit"
        assert key_resnet in CALIBRATED_THRESHOLDS, f"Missing ResNet18 threshold for {cat}"
        assert key_vit in CALIBRATED_THRESHOLDS, f"Missing ViT threshold for {cat}"
        assert CALIBRATED_THRESHOLDS[key_resnet]["youden_j"] > 0
        assert CALIBRATED_THRESHOLDS[key_vit]["youden_j"] > 0


def test_model_manager_inspect():
    """Verifies ModelManager loads checkpoint and inspects sample image."""
    manager = ModelManager(DEVICE)
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    
    result = manager.inspect(
        image_input=img,
        category="bottle",
        method="patchcore_resnet18",
        threshold=1.982,
    )
    
    assert result.category == "bottle"
    assert result.method == "patchcore_resnet18"
    assert isinstance(result.anomaly_score, float)
    assert isinstance(result.is_anomaly, (bool, np.bool_))
    assert result.raw_anomaly_map.shape == (224, 224)
    assert result.display_image.shape == (224, 224, 3)
    assert result.latency_ms > 0


def test_aoi_station_window_init(qapp):
    """Verifies AOIStationWindow initializes, configures widgets, and switches categories."""
    window = AOIStationWindow()
    assert window.combo_category.count() == 15
    assert window.combo_method.count() == len(METHOD_NAMES)
    
    # Test category switch
    window.combo_category.setCurrentText("capsule")
    assert window.combo_category.currentText() == "capsule"
    assert window.spin_thresh.value() > 0
    
    # Test auto threshold
    window.apply_auto_threshold("youden_j")
    youden_val = window.spin_thresh.value()
    assert youden_val == pytest.approx(1.589, rel=1e-2)
    
    window.close()

