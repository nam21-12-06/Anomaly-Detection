"""
Deep Feature Extractor & Memory Bank for Visual Anomaly Detection.

Implements:
1. ResNetExtractor: PyTorch forward-hook based multi-scale feature extractor
   from pretrained ResNet backbones (layer2 + layer3 spatial features).
2. FeatureMemoryBank: Storage, serialization, and nearest-neighbor distance
   scoring engine for normal representation modeling.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF


class ResNetExtractor(nn.Module):
    """
    Extracts multi-scale deep spatial representations from pretrained ResNet backbones.

    Uses non-invasive PyTorch forward hooks on intermediate residual blocks
    (default: 'layer2' and 'layer3') to preserve spatial resolution (28x28 and 14x14
    for a 224x224 input) rather than collapsing into a global average-pooled vector.

    Args:
        backbone: Name of torchvision backbone ('resnet18' or 'resnet50').
        pretrained: If True, loads official ImageNet-1K pretrained weights.
        layers: Tuple of layer names to attach hooks to.
        device: Torch device to host the model on.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        weights_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.layers = layers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure local hub directory to avoid storing weights on C:\ drive
        if weights_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            weights_dir = project_root / "weights"
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.weights_dir.resolve()))

        # Load backbone
        weights = "DEFAULT" if pretrained else None
        if self.backbone_name == "resnet18":
            self.model = models.resnet18(weights=weights)
        elif self.backbone_name == "resnet50":
            self.model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose 'resnet18' or 'resnet50'.")

        # Freeze all backbone parameters - purely inference / feature extraction
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval().to(self.device)

        # Hook storage and registration
        self._hook_features: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attaches forward hooks to designated intermediate layers."""
        self.remove_hooks()
        for layer_name in self.layers:
            if not hasattr(self.model, layer_name):
                raise AttributeError(f"Model {self.backbone_name} does not have attribute '{layer_name}'")
            layer_module = getattr(self.model, layer_name)
            handle = layer_module.register_forward_hook(self._make_hook(layer_name))
            self._hook_handles.append(handle)

    def _make_hook(self, layer_name: str):
        """Creates a closure hook function to cache layer outputs."""
        def hook(module, input_tensor, output_tensor):
            self._hook_features[layer_name] = output_tensor
        return hook

    def remove_hooks(self) -> None:
        """Removes all registered hooks to prevent memory leaks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._hook_features.clear()

    @property
    def embedding_dim(self) -> int:
        """Returns the total channel dimension after concatenating layers."""
        dim_map = {
            "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
            "resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
        }
        return sum(dim_map[self.backbone_name][l] for l in self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: passes image through backbone, extracts intermediate maps,
        bilinearly aligns spatial dimensions to the shallowest hook layer, and concatenates channels.

        Args:
            x: Input images tensor of shape (B, 3, H, W).

        Returns:
            Multi-scale spatial feature map tensor of shape (B, embedding_dim, H_out, W_out).
        """
        x = x.to(self.device)
        self._hook_features.clear()

        # Run forward pass (features captured via hooks)
        with torch.no_grad():
            _ = self.model(x)

        extracted = [self._hook_features[name] for name in self.layers]
        target_size = extracted[0].shape[-2:]  # (H, W) of the shallowest layer (e.g. layer2: 28x28)

        # Bilinear interpolation to align deeper features to target spatial resolution
        aligned_features = []
        for feat in extracted:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            aligned_features.append(feat)

        # Concatenate along channel dimension
        fused = torch.cat(aligned_features, dim=1)  # (B, embedding_dim, H_out, W_out)
        return fused

    def get_patch_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extracts patch embeddings from input images, flattens spatial grid into vectors.

        Args:
            x: Input images tensor of shape (B, 3, H, W).

        Returns:
            Tuple of:
                - patch_embeddings: Shape (B, N_patches, embedding_dim) where N_patches = H_out * W_out.
                - spatial_shape: (H_out, W_out).
        """
        feat_map = self.forward(x)  # (B, C, H_out, W_out)
        b, c, h, w = feat_map.shape
        # Permute to (B, H_out, W_out, C) and flatten spatial dimensions
        patches = feat_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return patches, (h, w)

    def __del__(self):
        self.remove_hooks()


class FeatureMemoryBank:
    """
    Manages a memory bank of normal patch embeddings and computes nearest-neighbor distances.

    Stores:
        bank: Tensor of shape (N_total_patches, embedding_dim) containing normal feature vectors.

    Features:
        - Batch-friendly Euclidean / Cosine distance computation.
        - Patch-level distance maps and image-level max anomaly scores.
        - Serialization to disk (.pt).
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bank: Optional[torch.Tensor] = None
        self.metadata: Dict = {}

    def fit(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """
        Aggregates normal feature representations into the memory bank.

        Args:
            features: Tensor of shape (N, D) or (B, N_patches, D), or list of tensors.
        """
        if isinstance(features, list):
            flat_list = [f.reshape(-1, f.shape[-1]).cpu() for f in features]
            self.bank = torch.cat(flat_list, dim=0).to(self.device)
        elif isinstance(features, torch.Tensor):
            self.bank = features.reshape(-1, features.shape[-1]).to(self.device)
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

    def __len__(self) -> int:
        return self.bank.shape[0] if self.bank is not None else 0

    def predict_distance(
        self,
        test_patches: torch.Tensor,
        spatial_shape: Tuple[int, int],
        n_neighbors: int = 1,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the nearest-neighbor anomaly score for test patch embeddings.

        For each test patch, computes:
            s(p) = min_{m in Bank} ||p - m||_2

        Args:
            test_patches: Tensor of shape (B, N_patches, D).
            spatial_shape: (H_out, W_out) to reconstruct 2D anomaly maps.
            n_neighbors: Number of nearest neighbors to average (default: 1).
            batch_size: Chunk size of test patches to compare simultaneously to preserve RAM.

        Returns:
            Tuple of:
                - patch_scores: Tensor of shape (B, H_out, W_out) with anomaly scores per patch.
                - image_scores: Tensor of shape (B,) with image-level max anomaly scores.
        """
        if self.bank is None or len(self.bank) == 0:
            raise RuntimeError("Memory bank is empty. Call fit() before predicting.")

        test_patches = test_patches.to(self.device)
        b, n_patches, d = test_patches.shape
        h, w = spatial_shape
        assert n_patches == h * w, f"Mismatch: n_patches ({n_patches}) != h * w ({h * w})"

        flat_patches = test_patches.reshape(-1, d)  # (B * N_patches, D)
        total_test_patches = flat_patches.shape[0]

        min_distances = []

        # Chunked distance calculation to avoid VRAM OOM when bank is large
        with torch.no_grad():
            for i in range(0, total_test_patches, batch_size):
                chunk = flat_patches[i : i + batch_size]  # (chunk_size, D)
                # Compute pairwise Euclidean distance: (chunk_size, N_bank)
                dists = torch.cdist(chunk, self.bank, p=2.0)
                if n_neighbors == 1:
                    topk_dists, _ = dists.min(dim=1)
                else:
                    topk_dists, _ = dists.topk(n_neighbors, dim=1, largest=False)
                    topk_dists = topk_dists.mean(dim=1)
                min_distances.append(topk_dists)

            all_dists = torch.cat(min_distances, dim=0)  # (B * N_patches,)

        # Reshape to (B, H_out, W_out)
        patch_scores = all_dists.reshape(b, h, w)
        # Image-level score is the maximum patch score across the spatial grid
        image_scores = patch_scores.reshape(b, -1).max(dim=1).values

        return patch_scores, image_scores

    def predict_anomaly_maps(
        self,
        test_patches: torch.Tensor,
        spatial_shape: Tuple[int, int],
        image_size: Tuple[int, int] = (224, 224),
        n_neighbors: int = 1,
        sigma: float = 4.0,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end inference from test patch embeddings to smoothed pixel-level
        anomaly maps of shape (B, 1, H, W) and image-level max anomaly scores of shape (B,).

        Args:
            test_patches: Tensor of shape (B, N_patches, D).
            spatial_shape: (H_out, W_out) patch grid dimensions.
            image_size: Target (H, W) resolution for output anomaly heatmaps.
            n_neighbors: Number of nearest neighbors to average (default: 1).
            sigma: Standard deviation for Gaussian spatial smoothing filter.
            batch_size: Patch chunk size for memory-bounded distance queries.

        Returns:
            Tuple of:
                - anomaly_maps: Smoothed heatmaps of shape (B, 1, image_size[0], image_size[1]).
                - image_scores: Image-level anomaly scores of shape (B,).
        """
        patch_scores, image_scores = self.predict_distance(
            test_patches,
            spatial_shape=spatial_shape,
            n_neighbors=n_neighbors,
            batch_size=batch_size,
        )
        anomaly_maps = interpolate_and_smooth_anomaly_maps(
            patch_scores,
            output_size=image_size,
            sigma=sigma,
        )
        return anomaly_maps, image_scores

    def save(self, path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """Serializes the memory bank and optional metadata to disk."""
        if self.bank is None:
            raise RuntimeError("Cannot save an empty memory bank.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"bank": self.bank.cpu(), "metadata": metadata or self.metadata}
        torch.save(payload, path)

    def load(self, path: Union[str, Path]) -> Dict:
        """Loads a saved memory bank and metadata from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Memory bank file not found at: {path}")
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.bank = data["bank"].to(self.device)
        self.metadata = data.get("metadata", {})
        return self.metadata


def interpolate_and_smooth_anomaly_maps(
    patch_scores: torch.Tensor,
    output_size: Tuple[int, int] = (224, 224),
    sigma: float = 4.0,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Interpolates 2D patch anomaly scores to high-resolution image maps and applies
    Gaussian spatial smoothing to eliminate blocky patch grid boundaries.

    Args:
        patch_scores: Anomaly scores per patch of shape (B, H_patch, W_patch).
        output_size: Target (H, W) spatial resolution (default: (224, 224)).
        sigma: Standard deviation for Gaussian spatial blur (default: 4.0).
        kernel_size: Size of Gaussian kernel filter window. If None, computed as 2 * int(4 * sigma) + 1.

    Returns:
        Tensor of shape (B, 1, output_size[0], output_size[1]) containing smoothed anomaly heatmaps.
    """
    if patch_scores.ndim != 3:
        raise ValueError(f"Expected patch_scores of shape (B, H, W), got ndim={patch_scores.ndim}")

    b, hp, wp = patch_scores.shape

    # Upsample from (B, 1, hp, wp) to (B, 1, H_out, W_out)
    upsampled = F.interpolate(
        patch_scores.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    if sigma > 0.0:
        if kernel_size is None:
            # 2 * ceil(4 * sigma) + 1 guarantees full 4-sigma coverage
            k = int(2 * np.ceil(4.0 * sigma) + 1)
            kernel_size = k if k % 2 == 1 else k + 1

        smoothed = TF.gaussian_blur(
            upsampled,
            kernel_size=[kernel_size, kernel_size],
            sigma=[float(sigma), float(sigma)],
        )
    else:
        smoothed = upsampled

    return smoothed

