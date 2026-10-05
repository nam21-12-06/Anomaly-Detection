from typing import Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transforms(
    image_size: Tuple[int, int] = (256, 256),
    crop_size: Tuple[int, int] = (224, 224),
    normalize_imagenet: bool = True,
):
    """
    Creates standard image and mask transformations for visual anomaly detection.

    Args:
        image_size: (H, W) to resize the image to.
        crop_size: (H, W) for center cropping (set equal to image_size to skip cropping).
        normalize_imagenet: If True, applies ImageNet mean and std normalization.
                           Set to False for Autoencoders reconstructing raw pixel intensities in [0, 1].

    Returns:
        img_transform: torchvision.transforms.Compose for input images.
        mask_transform: torchvision.transforms.Compose for binary ground-truth masks.
    """
    # Image pipeline
    img_transform_list = [
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
    ]
    if crop_size != image_size:
        img_transform_list.append(transforms.CenterCrop(crop_size))
    img_transform_list.append(transforms.ToTensor())

    if normalize_imagenet:
        img_transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    img_transform = transforms.Compose(img_transform_list)

    # Mask pipeline: must use NEAREST interpolation to preserve crisp binary edges
    mask_transform_list = [
        transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
    ]
    if crop_size != image_size:
        mask_transform_list.append(transforms.CenterCrop(crop_size))
    mask_transform_list.append(transforms.ToTensor())

    mask_transform = transforms.Compose(mask_transform_list)

    return img_transform, mask_transform
