import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.feature_extractor import ResNetExtractor
from src.models.patchcore import PatchCore


@pytest.fixture
def dummy_input():
    # Batch of 2 RGB images at 224x224
    return torch.randn(2, 3, 224, 224)


def test_resnet_extractor_wide_resnet50_shapes(dummy_input):
    """Verifies that wide_resnet50_2 is properly supported and outputs 1536-dim features."""
    extractor = ResNetExtractor(backbone="wide_resnet50_2", pretrained=False)
    # wide_resnet50_2: layer2 (512) + layer3 (1024) = 1536
    assert extractor.embedding_dim == 1536
    features = extractor(dummy_input)
    assert features.shape == (2, 1536, 28, 28)
    extractor.remove_hooks()


def test_resnet_extractor_patch_neighborhood_pooling(dummy_input):
    """Verifies that 3x3 patch pooling preserves spatial resolution while smoothing."""
    # Without pooling
    ext_raw = ResNetExtractor(backbone="resnet18", pretrained=False, patch_pooling=False)
    raw_feats = ext_raw(dummy_input)
    ext_raw.remove_hooks()

    # With 3x3 neighborhood pooling
    ext_pooled = ResNetExtractor(backbone="resnet18", pretrained=False, patch_pooling=True, pooling_kernel=3)
    pooled_feats = ext_pooled(dummy_input)
    ext_pooled.remove_hooks()

    assert raw_feats.shape == pooled_feats.shape == (2, 384, 28, 28)
    # The pooled features must differ from raw pointwise features due to 3x3 averaging
    assert not torch.allclose(raw_feats, pooled_feats)


def test_patchcore_initialization():
    model = PatchCore(
        backbone="resnet18",
        pretrained=False,
        patch_pooling=True,
        coreset_sampling_ratio=0.1,
        n_neighbors=5,
    )
    assert model.embedding_dim == 384
    assert model.patch_pooling is True
    assert model.coreset_sampling_ratio == 0.1
    assert model.n_neighbors == 5
    model.extractor.remove_hooks()


def test_patchcore_fit_and_predict_synthetic():
    # Synthetic normal images: 4 images of size (3, 224, 224)
    normal_images = torch.randn(4, 3, 224, 224)
    dataset = TensorDataset(normal_images)
    loader = DataLoader(dataset, batch_size=2)

    model = PatchCore(
        backbone="resnet18",
        pretrained=False,
        patch_pooling=True,
        coreset_sampling_ratio=0.05,  # 5% coreset
        n_neighbors=3,
        dimension_reduction=32,
    )

    # 1. Fit PatchCore
    metadata = model.fit(loader, verbose=False, random_state=42)
    # Total patches: 4 images * (28 * 28) = 3136 patches
    assert metadata["original_patches_count"] == 3136
    # 5% of 3136 = 157 coreset patches
    assert metadata["coreset_patches_count"] == 157
    assert len(model.memory_bank) == 157

    # 2. Predict on test images (2 images)
    test_images = torch.randn(2, 3, 224, 224)
    maps, scores = model.predict(test_images, reweight=True)

    # Spatial resolution must match full image resolution (224, 224)
    assert maps.shape == (2, 1, 224, 224)
    assert scores.shape == (2,)
    assert (maps >= 0.0).all()
    assert (scores >= 0.0).all()

    # Predict without reweighting
    maps_raw, scores_raw = model.predict(test_images, reweight=False)
    assert maps_raw.shape == (2, 1, 224, 224)
    assert scores_raw.shape == (2,)

    model.extractor.remove_hooks()


def test_patchcore_save_and_load():
    model = PatchCore(backbone="resnet18", pretrained=False, coreset_sampling_ratio=0.1)
    # Manually populate bank
    fake_coreset = torch.randn(20, 384)
    model.memory_bank.fit(fake_coreset)
    model.metadata = {"backbone": "resnet18", "coreset_size": 20}

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "patchcore_model.pt"
        model.save(ckpt_path)
        assert ckpt_path.exists()

        new_model = PatchCore(backbone="resnet18", pretrained=False)
        meta = new_model.load(ckpt_path)
        assert len(new_model.memory_bank) == 20
        assert meta["coreset_size"] == 20
        assert torch.allclose(model.memory_bank.bank.cpu(), new_model.memory_bank.bank.cpu())

        new_model.extractor.remove_hooks()
    model.extractor.remove_hooks()

