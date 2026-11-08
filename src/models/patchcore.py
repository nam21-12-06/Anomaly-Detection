"""
PatchCore: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022).

Combines:
1. Locally aware multi-scale patch feature representation (ResNet / WideResNet + 3x3 AvgPool).
2. Greedy Farthest-Point Coreset Subsampling for high memory efficiency and fast inference.
3. Patch-level nearest-neighbor anomaly scoring.
4. Image-level anomaly score with neighborhood density re-weighting.
5. High-resolution anomaly localization heatmaps via bilinear interpolation + Gaussian smoothing.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.feature_extractor import (
    FeatureMemoryBank,
    ResNetExtractor,
    interpolate_and_smooth_anomaly_maps,
)


class PatchCore(nn.Module):
    """
    PatchCore industrial visual anomaly detection model.

    Args:
        backbone: CNN backbone architecture ('wide_resnet50_2', 'resnet50', or 'resnet18').
        pretrained: If True, uses official ImageNet-1K pretrained weights.
        layers: Intermediate residual layers to extract features from.
        patch_pooling: If True, applies 3x3 average pooling to incorporate local patch neighborhood context.
        pooling_kernel: Size of the neighborhood pooling kernel window (default: 3).
        coreset_sampling_ratio: Fraction of memory bank patches to retain (default: 0.01 for 1%).
        n_neighbors: Number of nearest neighbors for patch distance / density re-weighting (default: 9).
        dimension_reduction: Johnson-Lindenstrauss random projection dimension for fast coreset selection.
        image_size: Spatial target resolution (H, W) for input and anomaly maps (default: (224, 224)).
        sigma: Standard deviation for Gaussian spatial smoothing filter (default: 4.0).
        weights_dir: Directory to cache model weights.
        device: Torch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        pretrained: bool = True,
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        patch_pooling: bool = True,
        pooling_kernel: int = 3,
        coreset_sampling_ratio: float = 0.01,
        n_neighbors: int = 9,
        dimension_reduction: Optional[int] = 128,
        image_size: Tuple[int, int] = (224, 224),
        sigma: float = 4.0,
        weights_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self.patch_pooling = patch_pooling
        self.pooling_kernel = pooling_kernel
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.n_neighbors = n_neighbors
        self.dimension_reduction = dimension_reduction
        self.image_size = image_size
        self.sigma = sigma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature extractor with neighborhood patch pooling
        self.extractor = ResNetExtractor(
            backbone=backbone,
            pretrained=pretrained,
            layers=layers,
            patch_pooling=patch_pooling,
            pooling_kernel=pooling_kernel,
            weights_dir=weights_dir,
            device=self.device,
        )

        # Memory bank to store the coreset representation of normality
        self.memory_bank = FeatureMemoryBank(device=self.device)
        self.metadata: Dict[str, Any] = {}

    @property
    def embedding_dim(self) -> int:
        return self.extractor.embedding_dim

    def fit(
        self,
        train_loader: DataLoader,
        verbose: bool = True,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Builds the PatchCore memory bank from defect-free normal training images
        and applies greedy farthest-point coreset subsampling.

        Args:
            train_loader: DataLoader providing defect-free training images.
            verbose: If True, prints progress bars and extraction metrics.
            random_state: Seed for reproducible coreset subsampling.

        Returns:
            Dictionary containing memory bank and subsampling metadata.
        """
        self.extractor.eval()
        all_features: List[torch.Tensor] = []
        spatial_shape: Optional[Tuple[int, int]] = None

        t0 = time.time()
        iterator = tqdm(train_loader, desc=f"PatchCore Extraction ({self.backbone_name})") if verbose else train_loader

        with torch.no_grad():
            for batch in iterator:
                images = batch["image"] if isinstance(batch, dict) else batch[0]
                images = images.to(self.device)
                patches, s_shape = self.extractor.get_patch_embeddings(images)
                if spatial_shape is None:
                    spatial_shape = s_shape
                all_features.append(patches.cpu())

        t_extract = time.time() - t0

        # Flatten all patch vectors into a single 2D tensor: (N_total, D)
        full_bank_features = torch.cat(all_features, dim=0).reshape(-1, self.embedding_dim)
        n_total_patches = full_bank_features.shape[0]

        if verbose:
            print(f"Aggregated {n_total_patches:,} patch vectors ({n_total_patches * self.embedding_dim * 4 / (1024*1024):.2f} MB).")
            print(f"Running Coreset Subsampling (ratio = {self.coreset_sampling_ratio:.2%})...")

        # Step 2: Coreset Subsampling
        t_core_start = time.time()
        full_bank = FeatureMemoryBank(device=self.device)
        full_bank.fit(full_bank_features)

        self.memory_bank = full_bank.subsample_coreset(
            sampling_ratio=self.coreset_sampling_ratio,
            dimension_reduction=self.dimension_reduction,
            random_state=random_state,
        )
        t_subsample = time.time() - t_core_start

        coreset_size = len(self.memory_bank)
        coreset_mb = coreset_size * self.embedding_dim * 4 / (1024 * 1024)

        if verbose:
            print(
                f"Coreset created: {coreset_size:,} patches ({coreset_mb:.2f} MB) in {t_subsample:.2f}s "
                f"[Compression: {n_total_patches / max(1, coreset_size):.1f}x]."
            )

        self.metadata = {
            "backbone": self.backbone_name,
            "layers": list(self.layers),
            "patch_pooling": self.patch_pooling,
            "pooling_kernel": self.pooling_kernel,
            "embedding_dim": self.embedding_dim,
            "spatial_shape": spatial_shape,
            "image_size": self.image_size,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "n_neighbors": self.n_neighbors,
            "original_patches_count": n_total_patches,
            "coreset_patches_count": coreset_size,
            "extraction_time_s": t_extract,
            "subsampling_time_s": t_subsample,
        }
        self.memory_bank.metadata = self.metadata
        return self.metadata

    def compute_image_scores_reweighted(
        self,
        test_patches: torch.Tensor,
        max_patch_scores: torch.Tensor,
        argmax_indices: torch.Tensor,
        b_neighbors: int = 9,
    ) -> torch.Tensor:
        """
        Computes re-weighted image-level anomaly scores using neighborhood density
        scaling as described in PatchCore (Roth et al., CVPR 2022, Eq. 7).

        Scales the maximum patch anomaly score s* by a softmax weight computed
        over its b nearest neighbors in the memory bank to account for local density.
        """
        if self.memory_bank.bank is None or len(self.memory_bank) == 0:
            raise RuntimeError("Memory bank is empty.")

        b_batch = test_patches.shape[0]
        coreset = self.memory_bank.bank  # (N_coreset, D)

        # For each image in batch, extract its maximally anomalous patch embedding
        max_patches = torch.stack(
            [test_patches[i, argmax_indices[i]] for i in range(b_batch)],
            dim=0,
        )  # (B, D)

        # Pairwise distance from max patches to coreset: (B, N_coreset)
        dists = torch.cdist(max_patches, coreset, p=2.0)

        # Find top b nearest neighbors in coreset
        k_val = min(b_neighbors, coreset.shape[0])
        topk_dists, _ = dists.topk(k_val, dim=1, largest=False)  # (B, k_val)

        # Softmax re-weighting factor: Eq. (7) of PatchCore paper
        # w = 1 - (exp(s*) / sum_{m in N_b} exp(||p* - m||))
        s_star = topk_dists[:, 0:1]  # (B, 1) - distance to closest coreset point
        # Numerical stability: subtract s_star inside exponents
        exp_diffs = torch.exp(s_star - topk_dists)  # (B, k_val)
        sum_exp = exp_diffs.sum(dim=1)  # (B,)

        # Weight factor w = 1.0 - (1.0 / sum_exp)
        # When normal cluster is dense, sum_exp >> 1 => w -> 1.0
        # When normal point is an outlier, sum_exp -> 1.0 => w -> 0.0
        weights = 1.0 - (1.0 / sum_exp.clamp(min=1e-6))
        reweighted_scores = weights * max_patch_scores

        return reweighted_scores

    def predict(
        self,
        images: torch.Tensor,
        reweight: bool = True,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs full PatchCore inference on input test images.

        Args:
            images: Tensor of shape (B, 3, H, W).
            reweight: If True, uses softmax neighborhood re-weighting for image anomaly scores.
            batch_size: Patch chunk size for memory-safe distance calculations.

        Returns:
            Tuple of:
                - anomaly_maps: Smoothed pixel heatmaps of shape (B, 1, H, W).
                - image_scores: Image-level anomaly scores of shape (B,).
        """
        self.extractor.eval()
        if self.memory_bank.bank is None or len(self.memory_bank) == 0:
            raise RuntimeError("PatchCore memory bank is empty. Call fit() or load() before predicting.")

        images = images.to(self.device)
        with torch.no_grad():
            test_patches, spatial_shape = self.extractor.get_patch_embeddings(images)

        b, n_patches, d = test_patches.shape
        h, w = spatial_shape

        # Patch-level anomaly scores via 1-NN Euclidean distance to coreset
        patch_scores_2d, max_patch_scores = self.memory_bank.predict_distance(
            test_patches,
            spatial_shape=spatial_shape,
            n_neighbors=1,
            batch_size=batch_size,
        )

        # Image-level anomaly score computation
        if reweight and len(self.memory_bank) >= 2:
            # Flatten spatial grid to find argmax patch index per image
            flat_patch_scores = patch_scores_2d.reshape(b, -1)
            argmax_indices = flat_patch_scores.argmax(dim=1)
            image_scores = self.compute_image_scores_reweighted(
                test_patches=test_patches,
                max_patch_scores=max_patch_scores,
                argmax_indices=argmax_indices,
                b_neighbors=self.n_neighbors,
            )
        else:
            image_scores = max_patch_scores

        # Pixel-level anomaly heatmaps: Bilinear upsample + Gaussian spatial smoothing
        anomaly_maps = interpolate_and_smooth_anomaly_maps(
            patch_scores_2d,
            output_size=self.image_size,
            sigma=self.sigma,
        )

        return anomaly_maps, image_scores

    def save(self, path: Union[str, Path]) -> None:
        """Serializes the PatchCore model artifact (coreset memory bank + metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coreset_bank": self.memory_bank.bank.cpu() if self.memory_bank.bank is not None else None,
            "metadata": self.metadata,
        }
        torch.save(payload, path)

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Loads a serialized PatchCore model artifact from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PatchCore artifact not found at: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.memory_bank.bank = payload["coreset_bank"].to(self.device)
        self.metadata = payload.get("metadata", {})
        self.memory_bank.metadata = self.metadata
        return self.metadata

