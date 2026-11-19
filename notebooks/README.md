# Notebooks Directory

This folder is intended for interactive exploration and step-by-step experimentation:

- `01_dataset_exploration.ipynb`: Visualize MVTec AD samples, inspect labels, and compute ground truth defect morphometry.
- `02_autoencoder_baseline.ipynb`: Train and evaluate ConvAE and ConvVAE reconstruction error baselines with L1 + SSIM loss.
- `03_feature_extraction.ipynb`: Extract deep multi-scale features from CNN backbones and test Nearest Neighbor anomaly scoring.
- `04_vit_features.ipynb`: Deep dive into Vision Transformers (ViT), Multi-Head Self-Attention, and feature representation.
- `05_patchcore.ipynb`: SOTA PatchCore memory bank, Greedy Farthest-Point Coreset Subsampling, and Generalization Paradox resolution.
- `06_full_results.ipynb`: Comprehensive multi-category benchmark across 5 paradigms, thresholding, localization gallery, failure case analysis, and ablations.
