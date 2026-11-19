"""
Unit tests for scripts/full_evaluation.py helper functions.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.full_evaluation import (
    MVTEC_ALL_CATEGORIES,
    MVTEC_OBJECT_CATEGORIES,
    MVTEC_TEXTURE_CATEGORIES,
    compute_optimal_thresholds,
    generate_markdown_table,
    resolve_categories,
    resolve_methods,
)


def test_resolve_categories():
    # Test 'all'
    all_cats = resolve_categories(["all"])
    assert len(all_cats) == len(MVTEC_ALL_CATEGORIES)
    assert set(all_cats) == set(MVTEC_ALL_CATEGORIES)

    # Test 'textures'
    textures = resolve_categories(["textures"])
    assert len(textures) == len(MVTEC_TEXTURE_CATEGORIES)
    assert set(textures) == set(MVTEC_TEXTURE_CATEGORIES)

    # Test 'objects'
    objects = resolve_categories(["objects"])
    assert len(objects) == len(MVTEC_OBJECT_CATEGORIES)
    assert set(objects) == set(MVTEC_OBJECT_CATEGORIES)

    # Test explicit list with invalid filter
    custom = resolve_categories(["bottle", "non_existent_category", "carpet"])
    assert custom == ["bottle", "carpet"]


def test_resolve_methods():
    # Test 'all'
    all_methods = resolve_methods(["all"])
    assert "patchcore_resnet18" in all_methods
    assert "patchcore_vit" in all_methods
    assert "ae" in all_methods

    # Test subset
    subset = resolve_methods(["patchcore_resnet18", "invalid_method"])
    assert subset == ["patchcore_resnet18"]


def test_compute_optimal_thresholds():
    # Ground truth: 5 normal, 5 anomaly
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Scores: normal < anomaly
    y_scores = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.8, 0.85, 0.9, 0.95])

    thresholds = compute_optimal_thresholds(y_true, y_scores)

    assert "th_percentile_95" in thresholds
    assert "th_youden_j" in thresholds
    assert "th_f1_optimal" in thresholds
    assert "best_f1_score" in thresholds

    # Best F1 score on separable data should be 1.0
    assert thresholds["best_f1_score"] == pytest.approx(1.0, rel=1e-3)
    # Youden J threshold should be between normal and anomaly scores (0.3 < th <= 0.7)
    assert 0.25 < thresholds["th_youden_j"] <= 0.75


def test_generate_markdown_table():
    df = pd.DataFrame([
        {
            "category": "bottle",
            "category_type": "object",
            "method": "patchcore_resnet18",
            "backbone": "resnet18",
            "image_auroc": 1.0,
            "pixel_auroc": 0.9769,
            "pro_score": 0.9131,
            "latency_ms": 8.4,
            "memory_mb": 2.45,
        }
    ])
    md = generate_markdown_table(df)
    assert "| Category | Type | Method | Backbone |" in md
    assert "`bottle`" in md
    assert "100.00%" in md
    assert "97.69%" in md
    assert "91.31%" in md

