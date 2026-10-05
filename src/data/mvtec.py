from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.data.transforms import get_transforms


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec Anomaly Detection dataset.

    Args:
        root_dir: Path to dataset directory (e.g., './data' or './data/bottle').
        category: Name of product category (e.g., 'bottle', 'carpet', etc.).
        split: 'train' for normal training images, 'test' for evaluation images.
        transform: Optional transform applied to input image.
        mask_transform: Optional transform applied to ground-truth mask.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        category: str = "bottle",
        split: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split.lower()

        if transform is None or mask_transform is None:
            default_img_tf, default_mask_tf = get_transforms()
            self.transform = transform if transform is not None else default_img_tf
            self.mask_transform = mask_transform if mask_transform is not None else default_mask_tf
        else:
            self.transform = transform
            self.mask_transform = mask_transform

        assert self.split in ["train", "test"], f"Invalid split: {self.split}. Expected 'train' or 'test'."

        # Resolve dataset category directory
        if (self.root_dir / self.category).exists():
            self.category_dir = self.root_dir / self.category
        elif (self.root_dir / "test").exists():
            self.category_dir = self.root_dir
        else:
            raise FileNotFoundError(
                f"Cannot find category directory at '{self.root_dir / self.category}' or '{self.root_dir}'."
            )

        self.samples: List[Dict[str, Union[Path, int, str]]] = []
        self._load_samples()

    def _load_samples(self):
        split_dir = self.category_dir / self.split
        gt_dir = self.category_dir / "ground_truth"

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        if self.split == "train":
            # Train split contains only 'good' (normal) samples
            good_dir = split_dir / "good"
            if not good_dir.exists():
                raise FileNotFoundError(f"Train good directory not found: {good_dir}")

            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": 0,  # 0: Normal
                        "mask_path": None,
                        "defect_type": "good",
                    }
                )
        else:
            # Test split contains both 'good' and multiple defect types
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue

                defect_type = defect_dir.name
                is_normal = defect_type == "good"

                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if not is_normal:
                        # Defect ground truth mask convention: <stem>_mask.png
                        expected_mask = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                        if expected_mask.exists():
                            mask_path = expected_mask
                        else:
                            # Fallback if named without _mask
                            fallback_mask = gt_dir / defect_type / img_path.name
                            if fallback_mask.exists():
                                mask_path = fallback_mask

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "label": 0 if is_normal else 1,  # 0: Normal, 1: Anomaly
                            "mask_path": mask_path,
                            "defect_type": defect_type,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        item = self.samples[idx]
        img_path = item["image_path"]
        label = item["label"]
        defect_type = item["defect_type"]
        mask_path = item["mask_path"]

        # Load RGB image
        image = Image.open(img_path).convert("RGB")

        # Load ground truth mask
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            # For normal images or missing masks, create empty black mask
            mask = Image.new("L", image.size, color=0)

        # Apply transformations
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        if self.mask_transform is not None:
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = transforms.ToTensor()(mask)

        # Ensure mask is strictly binary [0.0, 1.0]
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "label": label,
            "mask": mask_tensor,
            "defect_type": defect_type,
            "image_path": str(img_path),
        }


def get_mvtec_dataloader(
    root_dir: Union[str, Path] = "./data",
    category: str = "bottle",
    split: str = "train",
    image_size: Tuple[int, int] = (256, 256),
    crop_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    normalize_imagenet: bool = True,
) -> DataLoader:
    """Helper factory to create PyTorch DataLoader for MVTec AD."""
    img_transform, mask_transform = get_transforms(
        image_size=image_size,
        crop_size=crop_size,
        normalize_imagenet=normalize_imagenet,
    )

    dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split=split,
        transform=img_transform,
        mask_transform=mask_transform,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
