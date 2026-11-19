# Scripts Directory

Standalone utility and execution scripts:

- `train_autoencoder.py`: Script to train ConvAE and ConvVAE baseline models with L1 + SSIM loss.
- `evaluate_autoencoder.py`: Evaluate ConvAE and ConvVAE baselines on MVTec AD test sets.
- `build_feature_bank.py`: Build and serialize offline deep feature memory banks.
- `evaluate_feature_knn.py`: Evaluate feature memory banks with k-NN distance scoring.
- `run_patchcore.py`: End-to-end training and evaluation of PatchCore with ResNet and ViT backbones.
- `full_evaluation.py`: Automated multi-category, multi-paradigm benchmark pipeline with optimal thresholding and metric export.
