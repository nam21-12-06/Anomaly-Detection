<div align="center">

# ANOMALY-VISION

### Deep Learning-Based Industrial Visual Anomaly Detection

<p>
  <img src="https://img.shields.io/badge/Status-In%20Progress-FFA500?style=for-the-badge" alt="Status: In Progress" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA Enabled" />
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
  <strong>Current Phase:</strong> Step 1 Complete (Dataset Pipeline, Transforms, EDA Visualization). Step 2 Underway (Convolutional Autoencoder baseline).
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

## Planned Approaches

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
      <td>Custom PyTorch DataLoader, Nearest-neighbor mask interpolation</td>
      <td>Data integrity</td>
      <td><b>Completed</b></td>
    </tr>
    <tr>
      <td><b>Phase 2</b></td>
      <td>Reconstruction-Based</td>
      <td>Convolutional Autoencoder with L1 / SSIM Reconstruction Loss</td>
      <td>MSE, Reconstruction Error</td>
      <td><b>In Progress</b></td>
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

## Directory Structure

```text
Anomaly Detection/
|-- configs/
|   `-- config.yaml                 # Central experiment configurations
|-- data/
|   |-- bottle/                     # Extracted MVTec AD category data
|   |   |-- ground_truth/           # Pixel-level defect masks
|   |   |-- test/                   # Evaluation sets (good and defect classes)
|   |   `-- train/                  # Normal-only training samples
|   `-- README.md                   # Dataset layout and acquisition instructions
|-- notebooks/
|   |-- 01_dataset_exploration.ipynb# Interactive EDA and mask verification
|   `-- README.md
|-- results/                        # Exported plots and visualization artifacts
|-- scripts/
|   |-- eda_dataset.py              # CLI dataset verification and gallery export
|   `-- README.md
|-- src/
|   |-- data/
|   |   |-- mvtec.py                # PyTorch MVTecDataset and DataLoader factory
|   |   `-- transforms.py           # Standardized transforms for images and binary masks
|   |-- models/                     # Autoencoders, Feature Extractors, PatchCore
|   `-- utils/
|       `-- visualization.py        # Mask overlay and defect gallery plotting
|-- proposal.md                     # Research scope and development roadmap
|-- requirements.txt                # Python package dependencies
`-- README.md                       # Main project documentation
```

---

## Getting Started

### 1. Environment Setup

Clone the repository and set up a dedicated virtual environment:

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
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

If utilizing an NVIDIA GPU with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 2. Dataset Preparation

Download the desired categories from the official **MVTec Anomaly Detection** dataset and extract into `data/`:

```text
data/
`-- bottle/
    |-- ground_truth/
    |-- test/
    `-- train/
```

### 3. Verify Data Pipeline

Execute the exploratory data analysis script:

```bash
python scripts/eda_dataset.py
```

The script verifies dataset splits, checks binary mask alignment, and saves a defect gallery to `results/eda_gallery.png`.

Alternatively, launch the interactive notebook:

```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
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
- Roth, K., et al. (2022). *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*. CVPR 2022.
