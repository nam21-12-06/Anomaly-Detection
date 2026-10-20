from pathlib import Path
import pytest

from src.data import (
    MVTEC_ALL_CATEGORIES,
    MVTEC_OBJECT_CATEGORIES,
    MVTEC_TEXTURE_CATEGORIES,
    compute_defect_area_statistics,
    get_mvtec_summary,
)


def test_category_constants():
    assert len(MVTEC_OBJECT_CATEGORIES) == 10
    assert len(MVTEC_TEXTURE_CATEGORIES) == 5
    assert len(MVTEC_ALL_CATEGORIES) == 15
    assert "bottle" in MVTEC_OBJECT_CATEGORIES
    assert "carpet" in MVTEC_TEXTURE_CATEGORIES


def test_mvtec_summary_aggregation():
    data_dir = Path("data")
    if not (data_dir / "bottle").exists():
        pytest.skip("Dataset not found in data/")

    df_summary = get_mvtec_summary(data_dir)
    assert len(df_summary) >= 1
    assert "Category" in df_summary.columns
    assert "Train Normal" in df_summary.columns
    assert "Test Defect" in df_summary.columns
    assert df_summary["Train Normal"].sum() > 0


def test_compute_defect_area_statistics():
    data_dir = Path("data")
    if not (data_dir / "bottle" / "ground_truth").exists():
        pytest.skip("Bottle ground truth not found in data/")

    df_masks = compute_defect_area_statistics(data_dir)
    assert len(df_masks) > 0
    assert "Area Ratio (%)" in df_masks.columns
    assert (df_masks["Area Ratio (%)"] >= 0.0).all()

