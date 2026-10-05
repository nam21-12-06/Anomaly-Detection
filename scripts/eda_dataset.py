"""
EDA Script: Inspect dataset splits, class balances, and save sample visualizations.
Usage:
    python scripts/eda_dataset.py
"""

import sys
from pathlib import Path
import yaml

# Add root directory to sys.path to enable imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import MVTecDataset, get_mvtec_dataloader
from src.utils import plot_defect_gallery


def main():
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        dataset_cfg = cfg.get("dataset", {})
        category = dataset_cfg.get("category", "bottle")
        root_dir = dataset_cfg.get("root_dir", "./data")
    else:
        category = "bottle"
        root_dir = "./data"

    print("=" * 60)
    print(f"Loading MVTec Category: '{category.upper()}' from '{root_dir}'")
    print("=" * 60)

    # 1. Load train dataset (Normal only)
    train_ds = MVTecDataset(root_dir=root_dir, category=category, split="train")
    print(f"Train samples (Normal only): {len(train_ds)}")

    # 2. Load test dataset (Normal + Anomalies)
    test_ds = MVTecDataset(root_dir=root_dir, category=category, split="test")
    print(f"Test samples total:          {len(test_ds)}")

    # Count breakdown of defect types
    breakdown = {}
    for sample in test_ds.samples:
        dtype = sample["defect_type"]
        breakdown[dtype] = breakdown.get(dtype, 0) + 1

    print("\nTest breakdown:")
    for dtype, count in breakdown.items():
        label_str = "Normal" if dtype == "good" else "Anomaly"
        print(f"  - {dtype:15s}: {count:3d} images ({label_str})")

    # 3. Test DataLoader
    loader = get_mvtec_dataloader(
        root_dir=root_dir,
        category=category,
        split="test",
        batch_size=8,
        shuffle=False,
    )
    batch = next(iter(loader))
    print("\nBatch tensor shapes from DataLoader:")
    print(f"  - images: {batch['image'].shape} (dtype: {batch['image'].dtype})")
    print(f"  - masks:  {batch['mask'].shape}  (dtype: {batch['mask'].dtype})")
    print(f"  - labels: {batch['label'].shape} (dtype: {batch['label'].dtype})")

    # 4. Generate & save Defect Gallery
    output_path = Path("results/eda_gallery.png")
    plot_defect_gallery(test_ds, save_path=output_path)
    print(f"\nVisualization successfully saved to: {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
