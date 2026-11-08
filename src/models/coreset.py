"""
Coreset Subsampling Algorithm for PatchCore Anomaly Detection.

Implements the Greedy Farthest-Point Sampling algorithm (Minimax Facility Location)
with optional Johnson-Lindenstrauss random projection for dimension reduction,
as described in:
    Roth et al. (2022). Towards Total Recall in Industrial Anomaly Detection (PatchCore). CVPR 2022.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch


class GreedyCoresetSampler:
    """
    Greedy Farthest-Point Subsampling (Minimax Facility Location).

    Selects a representative subset (coreset) C from a large memory bank M
    such that the maximum distance from any point in M to its nearest point in C is minimized:
        min_{C subset M} max_{m in M} min_{c in C} ||m - c||_2

    Args:
        sampling_ratio: Fraction of total samples to retain in coreset (e.g., 0.01 for 1%, 0.10 for 10%).
        n_samples: Explicit number of samples to select. If provided, overrides sampling_ratio.
        dimension_reduction: Optional dimension d to project features to via a random Gaussian
            matrix (Johnson-Lindenstrauss Lemma) to accelerate distance calculations.
            If None or if d >= feature_dim, no projection is applied.
        random_state: Seed for reproducible initial point selection and random projection matrix.
        device: Torch device to execute distance updates on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        sampling_ratio: float = 0.1,
        n_samples: Optional[int] = None,
        dimension_reduction: Optional[int] = 128,
        random_state: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if not (0.0 < sampling_ratio <= 1.0):
            raise ValueError(f"sampling_ratio must be in (0.0, 1.0], got {sampling_ratio}")

        self.sampling_ratio = sampling_ratio
        self.n_samples = n_samples
        self.dimension_reduction = dimension_reduction
        self.random_state = random_state
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_random_projection(
        self,
        in_features: int,
        out_features: int,
        comp_device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Creates a Johnson-Lindenstrauss random Gaussian projection matrix:
            R ~ N(0, 1 / out_features) of shape (in_features, out_features).
        """
        proj = torch.randn(
            (in_features, out_features),
            generator=generator,
            dtype=torch.float32,
            device=comp_device,
        ) / np.sqrt(out_features)
        return proj

    def sample(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Subsamples features down to the target coreset size.

        Args:
            features: Tensor of shape (N, D) where N is number of patch vectors and D is feature dimension.

        Returns:
            Tuple of:
                - coreset: Tensor of shape (K, D) containing the selected feature vectors (on same device as input).
                - selected_indices: 1D Tensor of shape (K,) with indices of the selected vectors.
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D tensor of shape (N, D), but got shape {tuple(features.shape)}")

        n_total, d_orig = features.shape
        if n_total == 0:
            raise ValueError("Cannot subsample an empty feature tensor (N=0).")

        orig_device = features.device
        comp_device = self.device

        # Determine target number of coreset samples K
        if self.n_samples is not None:
            target_k = min(self.n_samples, n_total)
        else:
            target_k = max(1, int(np.ceil(self.sampling_ratio * n_total)))

        # Edge case: If target_k >= n_total, retain entire set
        if target_k >= n_total:
            indices = torch.arange(n_total, dtype=torch.long, device=orig_device)
            return features.clone(), indices

        # Create generator on appropriate device
        if comp_device.type == "cuda":
            generator = torch.Generator(device=comp_device).manual_seed(self.random_state)
        else:
            generator = torch.Generator(device="cpu").manual_seed(self.random_state)

        features_dev = features.to(dtype=torch.float32, device=comp_device)

        # Apply Johnson-Lindenstrauss random projection if requested and beneficial
        if (
            self.dimension_reduction is not None
            and self.dimension_reduction > 0
            and self.dimension_reduction < d_orig
        ):
            proj_matrix = self._get_random_projection(d_orig, self.dimension_reduction, comp_device, generator)
            # Projected representation: (N, d)
            proj_features = torch.matmul(features_dev, proj_matrix)
        else:
            proj_features = features_dev

        # Step 1: Select the initial starting point
        # A random index is selected deterministically via the seeded generator
        first_idx = int(torch.randint(0, n_total, (1,), generator=generator, device=comp_device).item())

        selected_indices = [first_idx]

        # Step 2: Compute initial pairwise distances from all points to first selected point
        # min_distances[i] represents the distance from point i to the closest point in coreset C
        center = proj_features[first_idx].unsqueeze(0)  # (1, d)
        min_distances = torch.norm(proj_features - center, p=2, dim=1)  # (N,)

        # Step 3: Greedy Farthest Point Selection Loop
        for _ in range(1, target_k):
            # Find the index that is furthest from its nearest neighbor in the current coreset
            furthest_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(furthest_idx)

            # Compute distance from all points to the newly added coreset point
            new_point = proj_features[furthest_idx].unsqueeze(0)  # (1, d)
            dist_to_new = torch.norm(proj_features - new_point, p=2, dim=1)  # (N,)

            # Update minimum distance cache: min(old_min_dist, dist_to_new_point)
            min_distances = torch.minimum(min_distances, dist_to_new)

        indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=orig_device)
        coreset = features[indices_tensor]

        return coreset, indices_tensor
