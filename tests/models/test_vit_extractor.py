from pathlib import Path
import pytest
import torch

from src.models import ViTExtractor


@pytest.fixture
def dummy_input():
    # Batch of 2 RGB images at standard resolution 224x224
    return torch.randn(2, 3, 224, 224)


def test_vit_extractor_initialization():
    extractor = ViTExtractor(
        model_name="vit_base_patch16_224",
        pretrained=False,
        blocks=(11,),
    )
    # Ensure all parameters are frozen for transfer learning
    assert all(not p.requires_grad for p in extractor.model.parameters())
    # Single block embedding dimension equals model embedding dimension (768)
    assert extractor.embedding_dim == 768
    extractor.remove_hooks()


def test_vit_extractor_single_block_forward(dummy_input):
    extractor = ViTExtractor(
        model_name="vit_base_patch16_224",
        pretrained=False,
        blocks=(11,),
    )
    features = extractor(dummy_input)
    # 224 / 16 = 14 spatial grid size
    assert features.shape == (2, 768, 14, 14)

    patches, spatial_shape = extractor.get_patch_embeddings(dummy_input)
    assert spatial_shape == (14, 14)
    assert patches.shape == (2, 14 * 14, 768)
    extractor.remove_hooks()


def test_vit_extractor_multi_block_shapes(dummy_input):
    extractor = ViTExtractor(
        model_name="vit_base_patch16_224",
        pretrained=False,
        blocks=(6, 9, 11),
    )
    # 3 blocks concatenated: 768 * 3 = 2304 channels
    assert extractor.embedding_dim == 2304
    features = extractor(dummy_input)
    assert features.shape == (2, 2304, 14, 14)

    patches, spatial_shape = extractor.get_patch_embeddings(dummy_input)
    assert spatial_shape == (14, 14)
    assert patches.shape == (2, 196, 2304)
    extractor.remove_hooks()


def test_vit_extractor_hook_cleanup(dummy_input):
    extractor = ViTExtractor(
        model_name="vit_base_patch16_224",
        pretrained=False,
        blocks=(11,),
    )
    assert len(extractor._hook_handles) == 1
    _ = extractor(dummy_input)
    assert 11 in extractor._hook_features

    extractor.remove_hooks()
    assert len(extractor._hook_handles) == 0
    assert len(extractor._hook_features) == 0


def test_vit_extractor_invalid_block_index():
    # ViT-Base has 12 blocks (indices 0 to 11)
    with pytest.raises(ValueError, match="out of range"):
        _ = ViTExtractor(
            model_name="vit_base_patch16_224",
            pretrained=False,
            blocks=(15,),
        )

