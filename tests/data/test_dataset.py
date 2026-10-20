from pathlib import Path
import pytest
import torch

from src.data import MVTecDataset, get_mvtec_dataloader, get_transforms


def test_transforms_binary_masks():
    img_tf, mask_tf = get_transforms(image_size=(256, 256), crop_size=(224, 224), normalize_imagenet=False)
    assert img_tf is not None
    assert mask_tf is not None


def test_dataset_train_split():
    data_dir = Path("data")
    if not (data_dir / "bottle").exists():
        pytest.skip("Bottle category data not found in data/")

    dataset = MVTecDataset(root_dir=data_dir, category="bottle", split="train")
    assert len(dataset) > 0

    item = dataset[0]
    assert "image" in item and "label" in item and "mask" in item
    assert item["label"] == 0, "Train split items must be normal (label=0)"
    assert item["defect_type"] == "good"
    assert (item["mask"] == 0.0).all(), "Train items must have zero ground truth mask"
    assert item["image"].shape == (3, 224, 224)


def test_dataset_test_split():
    data_dir = Path("data")
    if not (data_dir / "bottle").exists():
        pytest.skip("Bottle category data not found in data/")

    dataset = MVTecDataset(root_dir=data_dir, category="bottle", split="test")
    assert len(dataset) > 0

    labels = [s["label"] for s in dataset.samples]
    assert 0 in labels, "Test set must contain normal items"
    assert 1 in labels, "Test set must contain anomaly items"

    # Find an anomalous sample
    defect_idx = next(i for i, s in enumerate(dataset.samples) if s["label"] == 1)
    item = dataset[defect_idx]
    assert item["label"] == 1
    assert item["mask"].max() == 1.0, "Anomalous item mask must contain positive ground-truth pixels"
    # Ensure mask is strictly binary
    unique_vals = torch.unique(item["mask"])
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_dataloader_batch_shapes():
    data_dir = Path("data")
    if not (data_dir / "bottle").exists():
        pytest.skip("Bottle category data not found in data/")

    loader = get_mvtec_dataloader(
        root_dir=data_dir,
        category="bottle",
        split="train",
        batch_size=4,
        normalize_imagenet=False,
    )
    batch = next(iter(loader))

    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["mask"].shape == (4, 1, 224, 224)
    assert batch["label"].shape == (4,)

