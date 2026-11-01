import tempfile
from pathlib import Path
import pytest
import torch

from src.models.feature_extractor import (
    FeatureMemoryBank,
    ResNetExtractor,
    interpolate_and_smooth_anomaly_maps,
)


@pytest.fixture
def dummy_input():
    # Batch of 2 RGB images at standard resolution 224x224
    return torch.randn(2, 3, 224, 224)


def test_resnet_extractor_initialization():
    extractor = ResNetExtractor(backbone="resnet18", pretrained=False)
    # Ensure all parameters are frozen (no grad)
    assert all(not p.requires_grad for p in extractor.model.parameters())
    # ResNet-18 layer2 (128) + layer3 (256) = 384
    assert extractor.embedding_dim == 384
    extractor.remove_hooks()


def test_resnet_extractor_resnet50_shapes(dummy_input):
    extractor = ResNetExtractor(backbone="resnet50", pretrained=False)
    # ResNet-50 layer2 (512) + layer3 (1024) = 1536
    assert extractor.embedding_dim == 1536
    features = extractor(dummy_input)
    assert features.shape == (2, 1536, 28, 28)
    extractor.remove_hooks()


def test_resnet_extractor_forward_shapes(dummy_input):
    extractor = ResNetExtractor(backbone="resnet18", pretrained=False)
    features = extractor(dummy_input)
    # Output spatial resolution should match layer2: (224 / 8) = 28
    assert features.shape == (2, 384, 28, 28)

    patches, spatial_shape = extractor.get_patch_embeddings(dummy_input)
    assert spatial_shape == (28, 28)
    assert patches.shape == (2, 28 * 28, 384)
    extractor.remove_hooks()


def test_resnet_extractor_hook_cleanup(dummy_input):
    extractor = ResNetExtractor(backbone="resnet18", pretrained=False)
    assert len(extractor._hook_handles) == 2
    extractor.remove_hooks()
    assert len(extractor._hook_handles) == 0
    assert len(extractor._hook_features) == 0


def test_feature_memory_bank_fit_and_predict():
    bank = FeatureMemoryBank()
    # 100 normal patches of dimension 64
    normal_features = torch.randn(100, 64)
    bank.fit(normal_features)
    assert len(bank) == 100

    # Test patch identical to the first bank entry -> distance must be 0.0
    identical_test = normal_features[:1].unsqueeze(0)  # (1, 1, 64)
    patch_scores, image_scores = bank.predict_distance(
        identical_test, spatial_shape=(1, 1), n_neighbors=1
    )

    assert patch_scores.shape == (1, 1, 1)
    assert image_scores.shape == (1,)
    assert pytest.approx(patch_scores.item(), abs=1e-2) == 0.0
    assert pytest.approx(image_scores.item(), abs=1e-2) == 0.0


def test_feature_memory_bank_k_neighbors_averaging():
    bank = FeatureMemoryBank()
    # 50 normal patches of dimension 16
    normal_features = torch.randn(50, 16)
    bank.fit(normal_features)

    test_patches = torch.randn(2, 4, 16)  # 2 samples, 4 patches each
    patch_scores_k1, img_scores_k1 = bank.predict_distance(
        test_patches, spatial_shape=(2, 2), n_neighbors=1
    )
    patch_scores_k3, img_scores_k3 = bank.predict_distance(
        test_patches, spatial_shape=(2, 2), n_neighbors=3
    )

    assert patch_scores_k1.shape == (2, 2, 2)
    assert patch_scores_k3.shape == (2, 2, 2)
    # Average of top 3 smallest distances >= smallest distance
    assert (patch_scores_k3 >= patch_scores_k1 - 1e-5).all()


def test_interpolate_and_smooth_anomaly_maps():
    # Batch of 2, 7x7 patch grid
    patch_scores = torch.rand(2, 7, 7)
    smoothed_maps = interpolate_and_smooth_anomaly_maps(
        patch_scores, output_size=(224, 224), sigma=4.0
    )

    assert smoothed_maps.shape == (2, 1, 224, 224)
    # Output must be non-negative
    assert (smoothed_maps >= 0.0).all()

    # Test with sigma=0 (pure interpolation)
    raw_interp = interpolate_and_smooth_anomaly_maps(
        patch_scores, output_size=(112, 112), sigma=0.0
    )
    assert raw_interp.shape == (2, 1, 112, 112)

    # Test invalid dimension raises ValueError
    with pytest.raises(ValueError):
        interpolate_and_smooth_anomaly_maps(torch.rand(2, 1, 7, 7))


def test_feature_memory_bank_predict_anomaly_maps():
    bank = FeatureMemoryBank()
    normal_features = torch.randn(30, 32)
    bank.fit(normal_features)

    # 2 images, 4 patches each (2x2 grid)
    test_patches = torch.randn(2, 4, 32)
    maps, scores = bank.predict_anomaly_maps(
        test_patches,
        spatial_shape=(2, 2),
        image_size=(64, 64),
        n_neighbors=1,
        sigma=2.0,
    )

    assert maps.shape == (2, 1, 64, 64)
    assert scores.shape == (2,)


def test_feature_memory_bank_save_and_load():
    bank = FeatureMemoryBank()
    normal_features = torch.randn(50, 32)
    bank.fit(normal_features)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_bank.pt"
        bank.save(save_path)
        assert save_path.exists()

        new_bank = FeatureMemoryBank()
        new_bank.load(save_path)
        assert len(new_bank) == 50
        assert torch.allclose(bank.bank.cpu(), new_bank.bank.cpu())
