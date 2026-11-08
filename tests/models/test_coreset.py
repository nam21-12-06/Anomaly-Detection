import pytest
import torch

from src.models.coreset import GreedyCoresetSampler


def test_coreset_sampler_sampling_ratio():
    # 100 random vectors of dim 32
    torch.manual_seed(42)
    features = torch.randn(100, 32)

    sampler = GreedyCoresetSampler(sampling_ratio=0.1, random_state=42)
    coreset, indices = sampler.sample(features)

    assert coreset.shape == (10, 32)
    assert indices.shape == (10,)
    # All indices must be unique
    assert len(torch.unique(indices)) == 10
    # Values in coreset must match original features at those indices
    assert torch.allclose(coreset, features[indices])


def test_coreset_sampler_explicit_n_samples():
    torch.manual_seed(42)
    features = torch.randn(200, 64)

    sampler = GreedyCoresetSampler(n_samples=25, random_state=42)
    coreset, indices = sampler.sample(features)

    assert coreset.shape == (25, 64)
    assert len(indices) == 25
    assert len(torch.unique(indices)) == 25


def test_coreset_sampler_dimension_reduction():
    torch.manual_seed(42)
    # 150 vectors of high dimension 384
    features = torch.randn(150, 384)

    sampler = GreedyCoresetSampler(
        sampling_ratio=0.1,
        dimension_reduction=64,
        random_state=42,
    )
    coreset, indices = sampler.sample(features)

    # Output features must remain in original feature dimension (384)
    assert coreset.shape == (15, 384)
    assert len(indices) == 15


def test_coreset_sampler_edge_cases():
    features = torch.randn(50, 16)

    # Requesting more samples than available returns all
    sampler_large = GreedyCoresetSampler(n_samples=100)
    coreset_large, indices_large = sampler_large.sample(features)
    assert coreset_large.shape == (50, 16)
    assert len(indices_large) == 50

    # Requesting ratio 1.0 returns all
    sampler_full = GreedyCoresetSampler(sampling_ratio=1.0)
    coreset_full, indices_full = sampler_full.sample(features)
    assert coreset_full.shape == (50, 16)

    # Invalid ratio raises ValueError
    with pytest.raises(ValueError):
        GreedyCoresetSampler(sampling_ratio=0.0)

    with pytest.raises(ValueError):
        GreedyCoresetSampler(sampling_ratio=1.5)

    # Empty tensor raises ValueError
    empty_sampler = GreedyCoresetSampler(sampling_ratio=0.1)
    with pytest.raises(ValueError):
        empty_sampler.sample(torch.empty(0, 16))

    # Non-2D tensor raises ValueError
    with pytest.raises(ValueError):
        empty_sampler.sample(torch.randn(10, 5, 16))


def test_coreset_minimax_cluster_coverage():
    """
    Mathematical verification:
    If data consists of 4 well-separated clusters, a greedy coreset
    of size 4 must pick exactly one point from each cluster.
    """
    torch.manual_seed(0)
    # 4 tight clusters centered at (0, 0), (100, 0), (0, 100), (100, 100)
    c1 = torch.randn(25, 2) * 0.1 + torch.tensor([0.0, 0.0])
    c2 = torch.randn(25, 2) * 0.1 + torch.tensor([100.0, 0.0])
    c3 = torch.randn(25, 2) * 0.1 + torch.tensor([0.0, 100.0])
    c4 = torch.randn(25, 2) * 0.1 + torch.tensor([100.0, 100.0])

    features = torch.cat([c1, c2, c3, c4], dim=0)  # Shape (100, 2)

    sampler = GreedyCoresetSampler(n_samples=4, dimension_reduction=None, random_state=42)
    _, indices = sampler.sample(features)

    # Map indices to cluster IDs (0 to 3)
    cluster_ids = [idx.item() // 25 for idx in indices]

    # Every cluster must be represented!
    assert sorted(cluster_ids) == [0, 1, 2, 3], f"Expected all 4 clusters to be covered, got {cluster_ids}"

