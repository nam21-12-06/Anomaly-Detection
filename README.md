<div align="center">

# ANOMALY-VISION

### Deep Learning-Based Industrial Visual Anomaly Detection

<p>
  <img src="https://img.shields.io/badge/Status-In%20Progress-FFA500?style=for-the-badge" alt="Status: In Progress" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA Enabled" />
  <img src="https://img.shields.io/badge/Tests-36%20Passing-brightgreen?style=for-the-badge" alt="Tests Passing" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License" />
</p>

<p align="center">
  A modular framework for unsupervised defect detection and pixel-level anomaly localization on the MVTec AD benchmark.
</p>

</div>

<hr />

## Project Overview

In industrial manufacturing, anomalies such as cracks, scratches, structural deformations, and contaminations occur rarely. Standard supervised classifiers fail due to severe class imbalance and the unbounded variety of potential defects.

**ANOMALY-VISION** addresses this cold-start quality inspection problem via **One-Class Visual Representation Learning**:
- Training is conducted strictly on defect-free samples (normal data).
- The model establishes a distribution of normality.
- At test time, deviations from this distribution are detected, scored, and localized down to individual pixels.

<div style="background-color: #f6f8fa; border-left: 4px solid #0969da; padding: 12px 16px; margin: 16px 0; border-radius: 4px;">
  <strong>Current Progress:</strong> Phases 1, 2, and 3 are Complete. Evaluated on MVTec <em>bottle</em>:
  <br />- <strong>ConvAE:</strong> Image-AUROC 93.3%, Pixel-AUROC 82.7%, PRO Score 59.2%
  <br />- <strong>ConvVAE:</strong> Image-AUROC 89.1%, Pixel-AUROC 80.1%, PRO Score 51.5%
  <br />- <strong>kNN-ResNet18 (k=3):</strong> Image-AUROC <strong>99.6%</strong>, Pixel-AUROC <strong>97.6%</strong>, PRO Score <strong>93.8%</strong>
  <br /><strong>Phase 4 Next:</strong> PatchCore Coreset Subsampling.
</div>

---

## Technical Pipeline

```text
Normal Training Images
         |
         v
Deep Feature Representation (CNN / ViT / Autoencoder)
         |
         v
Normality Modeling (Latent Reconstruction / Patch Memory Bank)
         |
         v
Inference Image
         |
         +---> 1. Image-Level Anomaly Score (Normal vs. Defect)
         |
         +---> 2. Pixel-Level Heatmap (Defect Localization)
```

---

## Implementation Roadmap

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Method</th>
      <th>Core Mechanism</th>
      <th>Key Metrics</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Phase 1</b></td>
      <td>Dataset & Preprocessing</td>
      <td>Modular PyTorch DataLoader, Nearest-neighbor mask interpolation, Multi-category EDA</td>
      <td>Data integrity</td>
      <td><b>Completed</b></td>
    </tr>
    <tr>
      <td><b>Phase 2</b></td>
      <td>Reconstruction Baseline</td>
      <td>ConvAutoencoder & ConvVAE with L1 + SSIM loss, KL divergence, and training resume</td>
      <td>Reconstruction Error, Image-AUROC</td>
      <td><b>Complete</b> (AE: 93.3% / VAE: 89.1% Image-AUROC on <em>bottle</em>)</td>
    </tr>
    <tr>
      <td><b>Phase 3</b></td>
      <td>Feature Embedding</td>
      <td>Pretrained ResNet forward-hook representations + Memory Bank k-NN scoring</td>
      <td>Image-AUROC, Pixel-AUROC, PRO</td>
      <td><b>Complete</b> (kNN-ResNet18: 99.6% Image-AUROC, 97.6% Pixel-AUROC, 93.8% PRO on <em>bottle</em>)</td>
    </tr>
    <tr>
      <td><b>Phase 4</b></td>
      <td>PatchCore (SOTA)</td>
      <td>Patch-level features, Greedy farthest-point coreset subsampling, Memory Bank</td>
      <td>Image-AUROC, Pixel-AUROC</td>
      <td>Pending</td>
    </tr>
    <tr>
      <td><b>Phase 5</b></td>
      <td>Localization & Metrics</td>
      <td>Gaussian-smoothed anomaly heatmaps, Per-Region Overlap (PRO) across categories</td>
      <td>PRO Metric</td>
      <td>Partially Done (PRO metric active)</td>
    </tr>
    <tr>
      <td><b>Phase 6</b></td>
      <td>Deployment Demo</td>
      <td>Interactive web interface with Gradio for real-time inference</td>
      <td>Inference Latency</td>
      <td>Pending</td>
    </tr>
  </tbody>
</table>

---

## Core Formulations & Baseline Architecture

### 1. Reconstruction Baselines (Autoencoder & VAE)

#### Anomaly Residual Map
$$\text{Map}(u, v) = \frac{1}{C} \sum_{c=1}^C |x_c(u, v) - \hat{x}_c(u, v)|$$
- $x$: Original input image normalized to $[0.0, 1.0]$.
- $\hat{x}$: Reconstructed image produced by the decoder.
- $C$: Number of channels ($C=3$).

#### Composite Reconstruction Loss (L1 + SSIM)
$$\mathcal{L}_{\text{recon}} = \alpha \mathcal{L}_1(x, \hat{x}) + (1 - \alpha)(1 - \text{SSIM}(x, \hat{x}))$$
- $\alpha = 0.85$: Balances pixel-wise absolute error ($\mathcal{L}_1$) with structural similarity.
- $\text{SSIM}$: Evaluates localized luminance, contrast, and structural correlation over 11x11 2D Gaussian kernels to prevent blurred edges.

#### Variational Objective (VAE)
$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \parallel p(z))$$
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
- $\mu, \sigma^2$: Mean and variance predicted by the encoder for latent dimension $d=128$.
- Reparameterization during training: $z = \mu + \epsilon \odot \sigma$ with $\epsilon \sim \mathcal{N}(0, I)$. Evaluated deterministically at test time via $z = \mu$.
- $\beta = 0.0001$: Weight of the Kullback-Leibler divergence regularizer.

#### Image-Level Anomaly Score (Reconstruction)
$$S_{\text{image}} = \frac{1}{|K|} \sum_{(u, v) \in K} \text{Map}(u, v)$$
- $K$: Set of spatial coordinates corresponding to the top 5% highest residual error pixels, preventing normal background pixels from diluting localized defect signals.

---

### 2. Deep Feature Embedding & k-NN Memory Bank

#### Multi-Scale Intermediate Feature Alignment
$$f(p) = \left[ f_{\text{layer2}}(p) \;\parallel\; \text{Upsample}(f_{\text{layer3}}(p)) \right] \in \mathbb{R}^D$$
- $f_{\text{layer2}}(p) \in \mathbb{R}^{128}$: Intermediate activations captured via PyTorch forward hooks from ResNet-18 `layer2` (spatial grid 28x28 for 224x224 input).
- $f_{\text{layer3}}(p) \in \mathbb{R}^{256}$: Activations from `layer3` (14x14 grid), bilinearly upsampled to 28x28 to align spatial resolution.
- $D = 128 + 256 = 384$: Fused feature embedding dimension for each of the $P = 28 \times 28 = 784$ spatial patches.

#### Patch Anomaly Scoring via k-NN Euclidean Distance
$$s(p) = \frac{1}{k} \sum_{j=1}^k \|f(p) - m_{(j)}\|_2$$
- $f(p) \in \mathbb{R}^{384}$: Query patch embedding of the test image at coordinate $p$.
- $m_{(j)} \in \mathcal{M}$: The $j$-th nearest neighbor vector within the normal training Memory Bank $\mathcal{M}$ ($N_{\text{bank}} = 163,856$ vectors for *bottle*).
- $\|\cdot\|_2$: Euclidean distance ($L_2$ norm) in the unconstrained 384-dimensional feature space.
- $k$: Number of nearest neighbors averaged (default: $k=1$ or $k=3$).

#### Image-Level Anomaly Score (Feature-Based)
$$S_{\text{image}} = \max_{p \in \{1, \dots, P\}} s(p)$$
- The image anomaly score is defined as the maximum patch distance across the entire image.

#### Pixel-Level Anomaly Heatmap (Bilinear Upsample + Gaussian Spatial Smoothing)
$$\mathcal{A}(u, v) = \left( s_{\text{upsampled}} \ast G_\sigma \right)(u, v) = \sum_{u', v'} s_{\text{upsampled}}(u', v') \cdot \frac{1}{2\pi\sigma^2} \exp\left(-\frac{(u - u')^2 + (v - v')^2}{2\sigma^2}\right)$$
- $s_{\text{upsampled}}$: Patch score map $s(p)$ bilinearly interpolated from $28 \times 28$ to full resolution $224 \times 224$.
- $G_\sigma$: 2D Gaussian filter kernel.
- $\sigma = 4.0$: Standard deviation of the Gaussian filter. Eliminates discrete patch boundary grid artifacts while preserving true defect boundaries.

---

## Quantitative Benchmarks (`bottle` Category)

Evaluated on the MVTec AD *bottle* test set (83 images: 20 normal, 63 defective across `broken_large`, `broken_small`, `contamination`):

| Model Architecture | Image AUROC | Pixel AUROC | PRO Score (FPR <= 0.3) | Inference Latency |
|---|---|---|---|---|
| Convolutional Autoencoder (AE) | 93.33% | 82.74% | 59.18% | ~5 ms / image |
| Convolutional VAE (VAE) | 89.13% | 80.12% | 51.48% | ~5 ms / image |
| **kNN-ResNet18 (k=3)** | **99.60%** | **97.62%** | **93.80%** | **13.46s / 83 images (6.2 img/s)** |

---

## Directory Structure

```text
Anomaly Detection/
|-- configs/
|   `-- config.yaml                 # Central experiment configurations
|-- data/                           # MVTec AD dataset directory (15 categories)
|   |-- bottle/
|   `-- ...
|-- notebooks/
|   |-- 01_dataset_exploration.ipynb# Multi-category EDA & morphometry analysis
|   |-- 02_autoencoder_baseline.ipynb# AE & VAE quantitative evaluation
|   `-- README.md
|-- results/                        # Exported figures, CSVs, and localization heatmaps
|   |-- feature_knn/                # k-NN benchmark CSVs and localization plots
|   |-- bottle_baseline_benchmark.csv
|   `-- eda_gallery.png
|-- scripts/
|   |-- train_autoencoder.py        # AE & VAE training with resume, tqdm, & TensorBoard
|   |-- evaluate_autoencoder.py     # AE & VAE evaluation suite (AUROC & PRO)
|   |-- build_feature_bank.py       # Offline ResNet feature memory bank builder
|   `-- evaluate_feature_knn.py     # k-NN anomaly evaluation & heatmap export
|-- src/
|   |-- data/
|   |   |-- mvtec.py                # Dataset loaders and statistical aggregators
|   |   `-- transforms.py           # Standardized image & mask transformations
|   |-- models/
|   |   |-- autoencoder.py          # ConvAutoencoder and ConvVAE classes
|   |   |-- feature_extractor.py    # ResNetExtractor & FeatureMemoryBank
|   |   `-- losses.py               # SSIMLoss, AutoencoderLoss, VAELoss
|   `-- utils/
|       |-- metrics.py              # AUROC, Pixel AUROC, and PRO score computation
|       `-- visualization.py        # EDA plots, mask overlays, defect galleries
|-- tests/                          # 36 automated Pytest test cases across all modules
|-- anomaly_vision_6week_plan.md    # 6-week progressive learning roadmap
|-- proposal.md                     # Research scope and development roadmap
|-- requirements.txt                # Python package dependencies
`-- README.md                       # Main project documentation
```

---

## Getting Started

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .env
.\.env\Scripts\Activate.ps1

# Install package dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (36 Tests)

```bash
# Run full repository test suite
python -m pytest tests/ -v

# Run model and loss unit tests
python -m pytest tests/models/ -v
```

### 3. Reconstruction Baseline Pipeline (AE & VAE)

```bash
# Train Autoencoder
python scripts/train_autoencoder.py --model ae --category bottle --epochs 50 --batch_size 16 --lr 1e-3

# Train VAE
python scripts/train_autoencoder.py --model vae --category bottle --epochs 50 --batch_size 16 --lr 1e-3

# Resume training from latest checkpoint
python scripts/train_autoencoder.py --model ae --category bottle --resume auto

# Evaluate trained Autoencoder
python scripts/evaluate_autoencoder.py --category bottle --model ae
```

### 4. Deep Feature Extraction & k-NN Memory Bank Pipeline

```bash
# Step 1: Extract normal features and serialize memory bank
python scripts/build_feature_bank.py --category bottle --backbone resnet18 --batch_size 16

# Step 2: Run k-NN evaluation and generate localization plots
python scripts/evaluate_feature_knn.py --category bottle --backbone resnet18 --n_neighbors 3
```

---

## References

- Bergmann, P., et al. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019.
- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114.
- Wang, Z., et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)*. IEEE TIP.
- Roth, K., et al. (2022). *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*. CVPR 2022.
