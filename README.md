<div align="center">

# ANOMALY-VISION

### Deep Learning-Based Industrial Visual Anomaly Detection

<p>
  <img src="https://img.shields.io/badge/Status-In%20Progress-FFA500?style=for-the-badge" alt="Status: In Progress" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA Enabled" />
  <img src="https://img.shields.io/badge/Tests-26%20Passing-brightgreen?style=for-the-badge" alt="Tests Passing" />
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
  <strong>Current Phase:</strong> Phase 1 &amp; 2 Complete. ConvAE &amp; ConvVAE evaluated on MVTec <em>bottle</em>: Image-AUROC <strong>AE 93.3%</strong> / <strong>VAE 89.1%</strong>, Pixel-AUROC <strong>AE 82.7%</strong> / <strong>VAE 80.1%</strong>, PRO <strong>AE 59.2%</strong> / <strong>VAE 51.5%</strong>. TensorBoard logging added. <strong>Phase 3 Next:</strong> ResNet Feature Extraction + kNN.
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
      <td>ConvAutoencoder & ConvVAE with L1 + SSIM loss, KL divergence, and training script with resume</td>
      <td>Reconstruction Error, Image-AUROC</td>
      <td><b>&#x2705; Complete</b> &mdash; AE: 93.3% / VAE: 89.1% Image-AUROC on <em>bottle</em></td>
    </tr>
    <tr>
      <td><b>Phase 3</b></td>
      <td>Feature Embedding</td>
      <td>Pretrained ResNet intermediate activations + kNN anomaly scoring</td>
      <td>Image-AUROC</td>
      <td>Pending</td>
    </tr>
    <tr>
      <td><b>Phase 4</b></td>
      <td>PatchCore (SOTA)</td>
      <td>Patch-level features, Coreset subsampling, Memory Bank</td>
      <td>Image-AUROC, Pixel-AUROC</td>
      <td>Pending</td>
    </tr>
    <tr>
      <td><b>Phase 5</b></td>
      <td>Localization & Metrics</td>
      <td>Gaussian-smoothed anomaly heatmaps, Per-Region Overlap (PRO)</td>
      <td>PRO Metric</td>
      <td>Pending</td>
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

## Baseline Architecture & Theory (Autoencoders & VAEs)

Phase 2 establishes baseline reconstruction models to demonstrate how unsupervised identity mapping exposes industrial anomalies.

### 1. Conceptual Premise
During training on normal data $\mathcal{X}_{\text{normal}}$, the network optimizes parameters $\theta$ to compress images into a bottleneck latent space and reconstruct them accurately. 

When a test image containing an anomaly $x_{\text{anomaly}}$ is fed into the network:
1. The encoder projects the image into the latent space.
2. Because the decoder has only observed normal structures, it projects the latent representation back onto the learned manifold of normality.
3. The decoder attempts to "repair" or replace the defect with normal texture or geometry.
4. Subtracting the reconstruction from the original image yields large residual errors precisely at the anomalous coordinates:
$$\text{Anomaly Map}(u, v) = \frac{1}{C} \sum_{c=1}^C |x_c(u, v) - \hat{x}_c(u, v)|$$

### 2. Standard Convolutional Autoencoder (AE)
- **Encoder (`ConvEncoder`):** 5 progressive convolutional blocks with stride 2, BatchNorm, and LeakyReLU(0.2) downsampling from $(3, 224, 224)$ to $(512, 7, 7)$, followed by linear projection to latent dimension $z \in \mathbb{R}^{d}$ (e.g., $d=128$).
- **Decoder (`ConvDecoder`):** Symmetric transposed convolutions upsampling from latent vector $z$ back to $(3, 224, 224)$ with a final `Sigmoid` activation bounding output pixels to $[0.0, 1.0]$.

### 3. Variational Autoencoder (VAE)
Standard autoencoders can produce irregular latent spaces with empty gaps, leading to inconsistent reconstructions. The Variational Autoencoder imposes a probabilistic prior $p(z) = \mathcal{N}(0, I)$ over the latent space:
- The encoder predicts distribution parameters: mean $\mu(x)$ and log-variance $\log \sigma^2(x)$.
- **Reparameterization Trick:** Enables backpropagation through stochastic sampling:
$$z = \mu(x) + \epsilon \odot \sigma(x), \quad \epsilon \sim \mathcal{N}(0, I)$$
- At inference time, evaluation is deterministic using $z = \mu(x)$.

### 4. Loss Formulation: L1 + SSIM vs. MSE
Standard Mean Squared Error (MSE) penalizes errors quadratically:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum (x - \hat{x})^2$$
MSE tends to average high-frequency details, resulting in blurry reconstructions that obscure fine boundaries.

To preserve sharp edges and surface textures, we implement a composite loss combining **L1 Loss** and **Structural Similarity (SSIM)**:
$$\mathcal{L}_{\text{recon}} = \alpha \mathcal{L}_1(x, \hat{x}) + (1 - \alpha) (1 - \text{SSIM}(x, \hat{x}))$$
where $\alpha = 0.85$, and SSIM is evaluated over localized 2D Gaussian kernels comparing luminance, contrast, and structural correlation.

For the VAE, the total objective includes the Kullback-Leibler (KL) divergence penalty:
$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \parallel p(z))$$
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 5. Anomaly Scoring
- **Pixel-Level Localization:** Channel-averaged absolute residual smoothed with Gaussian filtering.
- **Image-Level Anomaly Score:** Average of the top 5% highest residual error pixels across the image, preventing global background pixels from diluting localized defect signals.

---

## Directory Structure

```text
Anomaly Detection/
|-- configs/
|   `-- config.yaml                 # Central experiment configurations
|-- data/                           # MVTec AD dataset directory (15 categories)
|   |-- bottle/
|   |-- cable/
|   `-- ...
|-- notebooks/
|   |-- 01_dataset_exploration.ipynb# Multi-category EDA & morphometry analysis
|   |-- 02_autoencoder_baseline.ipynb# AE & VAE evaluation: AUROC, ROC curves, galleries
|   `-- README.md
|-- results/                        # Exported figures and evaluation artifacts
|   |-- baseline_evaluation/        # AUROC plots, score distributions, galleries
|   `-- eda_gallery.png
|-- scripts/
|   |-- train_autoencoder.py        # Training script for AE & VAE with resume & tqdm
|   |-- evaluate_autoencoder.py     # Evaluation script for AE & VAE (AUROC & PRO)
|   `-- README.md
|-- src/
|   |-- data/
|   |   |-- mvtec.py                # Dataset loaders and statistical aggregators
|   |   `-- transforms.py           # Standardized image & mask transformations
|   |-- models/
|   |   |-- autoencoder.py          # ConvAutoencoder and ConvVAE classes
|   |   `-- losses.py               # SSIMLoss, AutoencoderLoss, VAELoss
|   `-- utils/
|       |-- metrics.py              # AUROC, Pixel AUROC, and PRO score computation
|       `-- visualization.py        # EDA plots, mask overlays, defect galleries
|-- tests/
|   |-- conftest.py                 # Pytest configuration
|   |-- data/                       # Dataset & DataLoader test suite
|   |   |-- test_dataset.py
|   |   `-- test_summary.py
|   |-- models/                     # Architecture & Loss test suite
|   |   |-- test_autoencoder.py
|   |   |-- test_vae.py
|   |   `-- test_losses.py
|   `-- utils/                      # Evaluation metrics test suite
|       `-- test_metrics.py
|-- anomaly_vision_6week_plan.md    # 6-week progressive learning roadmap
|-- proposal.md                     # Research scope and development roadmap
|-- requirements.txt                # Python package dependencies
`-- README.md                       # Main project documentation
```

---

## Getting Started

### 1. Environment Setup

Clone the repository and set up the virtual environment:

```bash
# Create virtual environment
python -m venv .env

# Activate virtual environment
# Windows (PowerShell):
.\.env\Scripts\Activate.ps1
# Windows (Git Bash):
source .env/Scripts/activate
# Linux / macOS:
source .env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If utilizing an NVIDIA GPU with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 2. Verify Data Pipeline & Multi-Category EDA

Launch Jupyter and open the exploratory notebook:

```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### 3. Training Baseline Models (AE & VAE)

The training script `scripts/train_autoencoder.py` supports standard Autoencoder and Variational Autoencoder, GPU acceleration, real-time `tqdm` progress metrics, and a training resume mechanism.

#### Train Standard Autoencoder (AE):
```bash
python scripts/train_autoencoder.py --model ae --category bottle --epochs 50 --batch_size 16 --lr 1e-3
```

#### Train Variational Autoencoder (VAE):
```bash
python scripts/train_autoencoder.py --model vae --category bottle --epochs 50 --batch_size 16 --lr 1e-3
```

#### Resume Interrupted Training:
You can resume training from the latest checkpoint automatically:
```bash
python scripts/train_autoencoder.py --model ae --category bottle --epochs 50 --resume auto
```
Or specify an exact checkpoint file:
```bash
python scripts/train_autoencoder.py --model ae --category bottle --resume checkpoints/bottle_ae/last.pth
```

Checkpoints are organized by category and model type under `checkpoints/{category}_{model}/`:
- `best.pth`: Weights achieving the lowest validation reconstruction loss.
- `last.pth`: Checkpoint containing current model, optimizer, and scheduler states for seamless resumption.
- `epoch_{N}.pth`: Periodic checkpoints saved every `--save_freq` epochs.

### 4. Run Automated Test Suite

The test suite is organized into modular folders (`tests/models/`, `tests/data/`, and `tests/utils/`):

```bash
# Run all tests across the entire repository (26 tests)
python -m pytest tests/ -v

# Run only model and loss tests
python -m pytest tests/models/ -v

# Run only dataset and dataloader tests
python -m pytest tests/data/ -v

# Run only evaluation metric tests
python -m pytest tests/utils/ -v
```

---

## Evaluation Protocol

Model performance is evaluated across two primary granularities:

1. **Image-Level Detection:**
   - **AUROC (Area Under the Receiver Operating Characteristic Curve):** Evaluates binary discriminative capacity between normal and anomalous samples independent of classification threshold.

2. **Pixel-Level Localization:**
   - **Pixel-AUROC:** Measures per-pixel anomaly segmentation accuracy against ground-truth defect masks.
   - **PRO (Per-Region Overlap):** Computes average overlap across individual connected defect components, weighting small and large defects equally.

---

## References

- Bergmann, P., et al. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019.
- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114.
- Wang, Z., et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)*. IEEE TIP.
- Roth, K., et al. (2022). *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*. CVPR 2022.
