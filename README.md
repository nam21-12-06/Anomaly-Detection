<div align="center">

# ANOMALY-VISION

### Deep Learning-Based Industrial Visual Anomaly Detection and Defect Localization on MVTec AD

<p>
  <img src="https://img.shields.io/badge/Status-Phases%201--4%20Complete-brightgreen?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA Enabled" />
  <img src="https://img.shields.io/badge/Tests-33%20Passing-brightgreen?style=for-the-badge" alt="Tests Passing" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License" />
</p>

<p align="center">
  A modular framework investigating, implementing, and benchmarking deep unsupervised visual representation learning methods for industrial quality inspection and pixel-level anomaly localization on the MVTec AD benchmark.
</p>

</div>

<hr />

## Table of Contents
1. [Core Motivation and Paradigm Evolution](#1-core-motivation-and-paradigm-evolution)
2. [Implementation Roadmap](#2-implementation-roadmap)
3. [Mathematical Formulations and Algorithmic Details](#3-mathematical-formulations-and-algorithmic-details)
   - [3.1. Reconstruction Baselines: Convolutional Autoencoder and VAE](#31-reconstruction-baselines-convolutional-autoencoder-and-vae)
   - [3.2. SOTA Framework: PatchCore (Memory Bank and Coreset Subsampling)](#32-sota-framework-patchcore-memory-bank-and-coreset-subsampling)
4. [Quantitative Benchmark Results](#4-quantitative-benchmark-results)
5. [Qualitative Experimental Visualizations](#5-qualitative-experimental-visualizations)
6. [Repository Structure](#6-repository-structure)
7. [Installation and Getting Started](#7-installation-and-getting-started)
8. [References](#8-references)

---

## 1. Core Motivation and Paradigm Evolution

### The Industrial Visual Inspection Challenge
In modern automated manufacturing (semiconductor wafer fabrication, glass container bottling, pharmaceutical packaging, textile weaving, precision metal machining):
- **Extreme Class Imbalance:** Defective items constitute a negligible fraction of manufacturing yields (< 1%).
- **Open-Set Defect Space:** Real-world defects encompass an unbounded diversity of structural manifestations (hairline cracks, micro-scratches, edge deformations, foreign contaminations, void punctures, weave misalignments). Consequently, closed-set supervised classifiers fail when encountering unseen anomaly modes.

### Paradigm: One-Class Visual Representation Learning
ANOMALY-VISION addresses this cold-start inspection challenge via unsupervised one-class learning: the system trains exclusively on defect-free nominal images to construct a manifold of normality. At test time, any statistical or structural divergence from this nominal distribution is quantified as an anomaly score and localized down to individual pixel coordinates.

This repository tracks the theoretical and empirical evolution across three paradigm generations:

```text
                           EVOLUTION OF ANOMALY DETECTION PARADIGMS
+----------------------------------------------------------------------------------------+
| GENERATION 1: RECONSTRUCTION-BASED (ConvAE, ConvVAE)                                   |
| - Principle: Compresses nominal patterns into a bottleneck latent space and decodes.   |
| - Anomaly Metric: Pixel-wise residual error |x - x^|.                                  |
| - Limitations: Reconstruction blurriness; tendency to "reconstruct through defects"   |
|   on high-frequency textures; sensitive to illumination fluctuations.                  |
+-----------------------------------------+----------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
| GENERATION 2: DEEP FEATURE MEMORY BANK (k-NN)                                          |
| - Principle: Transfers rich semantic representations from ImageNet-pretrained CNNs.    |
| - Anomaly Metric: Distance to the k-nearest nominal patch vectors in a memory bank.    |
| - Limitations: Massive memory footprint (240 MB for 163K vectors/category);            |
|   high k-NN search latency during real-time deployment.                                |
+-----------------------------------------+----------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
| GENERATION 3 (SOTA): PATCHCORE (Roth et al., CVPR 2022)                                |
| - Innovation 1: Local Neighborhood Aggregation (3x3 AvgPool) captures spatial context. |
| - Innovation 2: Minimax Facility Location Coreset Subsampling reduces bank by 99%      |
|   (240 MB -> 2.4 MB) while preserving exact convex hull boundaries.                    |
| - Innovation 3: Softmax Neighborhood Density Re-weighting eliminates boundary false    |
|   positives caused by sparse nominal outliers.                                         |
| - Performance: Achieves 100.0% Image AUROC, 97.7% Pixel AUROC on 'bottle' in 92 ms.    |
+----------------------------------------------------------------------------------------+
```

---

## 2. Implementation Roadmap

| Phase | Methodology | Core Mechanism | Evaluation Targets | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Phase 1** | Dataset and Pipeline | Multi-category MVTec AD DataLoader, Nearest-Neighbor mask interpolation, EDA visualization | Data integrity and defect statistics | **Complete** |
| **Phase 2** | Reconstruction Baseline | ConvAutoencoder and ConvVAE with Composite L1 + SSIM Loss and KL divergence | Image AUROC, Pixel AUROC | **Complete** <br />(AE: 93.3% / VAE: 89.1% Image AUROC) |
| **Phase 3** | Feature Embedding Bank | Intermediate PyTorch forward hooks on ResNet-18, multi-scale concatenation, k-NN memory bank | Image AUROC, Pixel AUROC | **Complete** <br />(kNN-ResNet18: 99.6% Image AUROC, 97.6% Pixel AUROC) |
| **Phase 4** | **PatchCore (SOTA)** | 3x3 Neighborhood Pooling, Greedy Farthest-Point Coreset Subsampling (1%), Softmax Density Re-weighting | Image AUROC, Pixel AUROC, PRO Score | **Complete** <br />(PatchCore: **100.0%** Image AUROC, **97.7%** Pixel AUROC, **91.3%** PRO) |
| **Phase 5** | Multi-Category Benchmark | Extension across Object ('bottle') and Texture ('carpet') categories with Per-Region Overlap (PRO) | Multi-category PRO and Latency | **Complete** |
| **Phase 6** | Deployment Demo | Interactive Gradio inspection interface with calibrated PASS/DEFECT thresholding | Real-time edge latency | **Complete** |

---

## 3. Mathematical Formulations and Algorithmic Details

### 3.1. Reconstruction Baselines: Convolutional Autoencoder and VAE

#### 1. Pixel-Level Residual Anomaly Map
Given an input test image $x \in \mathbb{R}^{C \times H \times W}$ and its reconstructed counterpart $\hat{x} \in \mathbb{R}^{C \times H \times W}$ produced by the convolutional decoder:
$$\mathcal{A}_{\text{residual}}(u, v) = \frac{1}{C} \sum_{c=1}^C |x_c(u, v) - \hat{x}_c(u, v)|$$

#### 2. Composite Reconstruction Objective ($\mathcal{L}_1 + \text{SSIM}$)
To prevent the blurring artifacts inherent to Mean Squared Error ($\mathcal{L}_2$), the network optimizes a convex combination of point-wise absolute deviation and localized structural consistency:
$$\mathcal{L}_{\text{recon}}(x, \hat{x}) = \alpha \mathcal{L}_1(x, \hat{x}) + (1 - \alpha) \left( 1 - \text{SSIM}(x, \hat{x}) \right)$$
where $\alpha = 0.85$. The Structural Similarity Index ($\text{SSIM}$) is computed over an $11 \times 11$ normalized Gaussian window:
$$\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}$$

#### 3. Variational Inference Objective (ConvVAE)
For ConvVAE, latent representation $z \sim q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ is constrained via the Kullback-Leibler divergence with respect to an uninformative standard normal prior $p(z) = \mathcal{N}(0, I)$:
$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}\left( q_\phi(z|x) \parallel p(z) \right)$$
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
where latent dimension $d = 128$ and regularizer weight $\beta = 0.0001$. Backpropagation utilizes the reparameterization trick $z = \mu + \epsilon \odot \sigma$ with $\epsilon \sim \mathcal{N}(0, I)$.

#### 4. Image-Level Anomaly Scoring (Top-$k\%$ Pixel Pooling)
To prevent extensive nominal background regions from attenuating localized anomaly signals:
$$S_{\text{image}} = \frac{1}{|K|} \sum_{(u, v) \in K} \mathcal{A}_{\text{residual}}(u, v)$$
where $K$ represents the spatial coordinates corresponding to the top $5\%$ highest residual error magnitudes.

---

### 3.2. SOTA Framework: PatchCore (Memory Bank and Coreset Subsampling)

PatchCore (Roth et al., CVPR 2022) eliminates reconstruction artifacts by mapping images into the semantic representation manifold of an ImageNet-pretrained convolutional hierarchy.

```text
                                 PATCHCORE SYSTEM ARCHITECTURE
+--------------+     +-------------+     +-----------------------------------------------------+
| Input Image  | --> | ResNet-18   | --> | layer2: (128 x 28 x 28)                             |
| (3x224x224)  |     | (Pretrained)|     | layer3: (256 x 14 x 14) --[Bilinear Upsample]------> | (256 x 28 x 28)
+--------------+     +-------------+     +--------------------------+--------------------------+
                                                                    |
                                                                    v
                                         +-----------------------------------------------------+
                                         | Channel Concatenation: F(x) in R^(384 x 28 x 28)    |
                                         | Neighborhood 3x3 AvgPool: Context-Aware Patches     |
                                         +--------------------------+--------------------------+
                                                                    |
                                                                    v
+----------------------------------------------------------------------------------------------+
| FULL NOMINAL MEMORY BANK (M): 163,856 feature vectors (240 MB for 'bottle')                  |
|                                                                                              |
|       [Minimax Greedy Farthest-Point Coreset Subsampling (Ratio gamma = 0.01)]               |
|                                                                                              |
| COMPRESSED CORESET BANK (M_C): 1,639 representative vectors (2.4 MB) - 100x Reduction        |
+-----------------------------------------------------------+----------------------------------+
                                                            |
                                                            v
+-------------------------------+               +----------------------------------------------+
|       TEST QUERY IMAGE        | ------------> | Patch-Level 1-NN Euclidean Distance:         |
|                               |               | s(p) = min_{c in M_C} ||f(p) - c||_2         |
+-------------------------------+               +-----------------------+----------------------+
                                                                        |
                                                                        v
                                                +----------------------------------------------+
                                                | 1. Anomaly Heatmap: Bilinear + Gauss Filter  |
                                                | 2. Image Score: Softmax Density Re-weighted  |
                                                +----------------------------------------------+
```

#### 1. Locally Aware Multi-Scale Patch Representation
For an input $x \in \mathbb{R}^{3 \times 224 \times 224}$, feature maps are extracted from hierarchical stages `layer2` ($\phi_2(x) \in \mathbb{R}^{128 \times 28 \times 28}$) and `layer3` ($\phi_3(x) \in \mathbb{R}^{256 \times 14 \times 14}$).
- Spatial alignment via bilinear interpolation: $\tilde{\phi}_3(x) = \text{BilinearUpsample}(\phi_3(x)) \in \mathbb{R}^{256 \times 28 \times 28}$.
- Channel concatenation:
  $$F(x) = \left[ \phi_2(x) \;\parallel\; \tilde{\phi}_3(x) \right] \in \mathbb{R}^{D \times 28 \times 28} \quad (D = 128 + 256 = 384)$$
- Local Neighborhood Aggregation:
  To incorporate receptive field context and establish robustness against minor misalignments, each spatial feature is smoothed over a $K \times K$ neighborhood ($K=3$, padding 1, stride 1):
  $$p(u, v) = \frac{1}{K^2} \sum_{i=-\lfloor K/2 \rfloor}^{\lfloor K/2 \rfloor} \sum_{j=-\lfloor K/2 \rfloor}^{\lfloor K/2 \rfloor} F(x)(u+i, v+j) \in \mathbb{R}^D$$
- The aggregate nominal memory bank $\mathcal{M}$ aggregates all patches from normal training set $\mathcal{X}_{\text{train}}$:
  $$\mathcal{M} = \bigcup_{x \in \mathcal{X}_{\text{train}}} \{p(u, v) \mid 1 \le u, v \le 28\} \subset \mathbb{R}^D \quad (|\mathcal{M}| = 163,856 \text{ for 'bottle'})$$

#### 2. Minimax Facility Location Coreset Subsampling
The objective is to find a compact coreset $\mathcal{M}_C \subset \mathcal{M}$ of cardinality $|\mathcal{M}_C| = N_C = \lceil \gamma |\mathcal{M}| \rceil$ ($\gamma = 0.01$ for 1%) such that the maximum distance from any candidate in $\mathcal{M}$ to its nearest representative in $\mathcal{M}_C$ is minimized:
$$\mathcal{M}_C^* = \arg\min_{\mathcal{M}_C \subset \mathcal{M}} \max_{m \in \mathcal{M}} \min_{c \in \mathcal{M}_C} \|m - c\|_2$$

**Iterative Greedy Farthest-Point Algorithm:**
1. Initialize by selecting an arbitrary starting vector $c_1 \sim \mathcal{M}$, setting $\mathcal{M}_C = \{c_1\}$.
2. Initialize distance cache: $d(m) = \|m - c_1\|_2, \quad \forall m \in \mathcal{M}$.
3. For iterations $t = 2, \dots, N_C$:
   - Select the vector furthest from the current coreset set:
     $$c_t = \arg\max_{m \in \mathcal{M}} d(m)$$
   - Append to coreset: $\mathcal{M}_C \leftarrow \mathcal{M}_C \cup \{c_t\}$.
   - Update cached minimum distances in $O(|\mathcal{M}|)$ operations:
     $$d(m) \leftarrow \min\left(d(m), \|m - c_t\|_2\right), \quad \forall m \in \mathcal{M}$$
4. **Acceleration via Johnson-Lindenstrauss Random Projection:**
   Prior to coreset extraction, vectors are projected to a lower-dimensional subspace $d = 128 < D = 384$ via a random Gaussian matrix $R \in \mathbb{R}^{D \times d}$ ($R_{ij} \sim \mathcal{N}(0, 1/d)$):
   $$\tilde{m} = m R \in \mathbb{R}^{128}$$
   By the Johnson-Lindenstrauss Lemma, pairwise Euclidean distances are preserved within a factor of $(1 \pm \epsilon)$, executing the subsampling process in under 3.1 seconds on GPU.

#### 3. Patch-Level Anomaly Scoring
For each test patch $f(p)$ from query image $x_{\text{test}}$, its anomaly score corresponds to the Euclidean distance to its nearest coreset neighbor:
$$s(p) = \min_{c \in \mathcal{M}_C} \|f(p) - c\|_2$$

#### 4. Softmax Neighborhood Density Re-weighting
Let $p^* = \arg\max_p s(p)$ denote the maximally anomalous patch in the query image, with score $s^* = s(p^*)$.
- Identify the set of $b$ nearest coreset neighbors ($b=9$) surrounding $p^*$: $\mathcal{N}_b(p^*) \subset \mathcal{M}_C$.
- The neighborhood density weight $w$ is defined as:
  $$w = 1 - \frac{\exp(s^*)}{\sum_{c \in \mathcal{N}_b(p^*)} \exp(\|p^* - c\|_2)} = 1 - \frac{1}{\sum_{c \in \mathcal{N}_b(p^*)} \exp\left(s^* - \|p^* - c\|_2\right)}$$
  - When $p^*$ is near a dense nominal cluster, the denominator dominates $\implies w \to 1.0$ (confident anomaly score).
  - When $p^*$ is an isolated boundary point in a sparse region of normal variation, the denominator approaches $1.0 \implies w \to 0.0$ (down-weighted, preventing False Positives).
- The final re-weighted image-level anomaly score is:
  $$S_{\text{image}} = w \cdot s^*$$

#### 5. High-Resolution Bilinear Interpolation and Gaussian Spatial Smoothing
The $28 \times 28$ patch score grid is bilinearly upsampled to the native resolution $(224, 224)$, followed by 2D Gaussian filtering ($G_\sigma$ with $\sigma = 4.0$ and kernel window $K_{\text{Gauss}} = 2 \lceil 4\sigma \rceil + 1$) to eliminate discrete boundary artifacts:
$$\mathcal{A}(u, v) = (s_{\text{upsampled}} \ast G_\sigma)(u, v) = \sum_{u', v'} s_{\text{upsampled}}(u', v') \cdot \frac{1}{2\pi\sigma^2} \exp\left(-\frac{(u - u')^2 + (v - v')^2}{2\sigma^2}\right)$$

---

## 4. Quantitative Benchmark Results

Evaluated across Object ('bottle') and Texture ('carpet') categories on the official MVTec AD test partitions:

| Category | Category Type | Model / Architecture | Image AUROC | Pixel AUROC | PRO Score (FPR <= 0.3) | Memory Bank Size | Inference Latency (GPU) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **bottle** | Object | Convolutional Autoencoder (AE) | 93.33% | 82.74% | 59.18% | N/A (Weights ~2.5 MB) | ~5 ms / image |
| **bottle** | Object | Convolutional VAE (VAE) | 89.13% | 80.12% | 51.48% | N/A (Weights ~2.5 MB) | ~5 ms / image |
| **bottle** | Object | kNN-ResNet18 (Full Memory Bank 100%) | 99.60% | 97.62% | 93.80% | 163,856 patches (240.0 MB) | 162 ms / image |
| **bottle** | Object | **PatchCore-ResNet18 (Coreset 1%)** | **100.00%** | **97.69%** | **91.31%** | **1,639 patches (2.4 MB)** | **92 ms / image** |
| **carpet** | Texture | **PatchCore-ResNet18 (Coreset 1%)** | **97.27%** | **98.65%** | **93.47%** | **2,196 patches (3.2 MB)** | **148 ms / image** |

> Note: Compared to the uncompressed 100% memory bank (163,856 vectors), 1% Coreset Subsampling delivers a 100x storage compression (from 240 MB down to 2.4 MB) and nearly doubles inference throughput, while attaining 100.00% Image AUROC on 'bottle'.

---

## 5. Qualitative Experimental Visualizations

The following figures illustrate defect detection and localization outputs generated across project phases:

### 5.1. Multi-Category Defect Exploration (Dataset Analysis)
Industrial defects manifest across distinct morphological profiles (fractures, contamination, thread pull, structural puncture):

<div align="center">
  <img src="results/eda_gallery.png" alt="MVTec Defect Exploration Gallery" width="90%" />
  <p><em>Figure 1: Defect distribution and ground-truth localization masks across representative MVTec AD categories.</em></p>
</div>

---

### 5.2. Reconstruction Baseline Defect Localization (ConvAE and ConvVAE)
Reconstruction baselines localize large, high-contrast defects via residual error maps, but exhibit background noise and edge blurriness around reflective glass boundaries:

<div align="center">
  <img src="results/bottle_ae_localization.png" alt="Autoencoder Localization" width="85%" />
  <p><em>Figure 2: Defect localization on 'bottle' using Convolutional Autoencoder optimized with Composite L1 + SSIM Loss.</em></p>
</div>

<div align="center">
  <img src="results/bottle_vae_localization.png" alt="VAE Localization" width="85%" />
  <p><em>Figure 3: Defect localization on 'bottle' using Convolutional VAE with regularized latent distribution.</em></p>
</div>

---

### 5.3. Feature Embedding k-NN Localization (Full Memory Bank)
Extracting deep hierarchical activations from intermediate CNN layers produces significantly sharper anomaly contours:

<div align="center">
  <img src="results/feature_knn/bottle_resnet18_k3_localization.png" alt="Feature k-NN Localization" width="90%" />
  <p><em>Figure 4: Defect localization via k-NN distance scoring (k=3) on the full ResNet-18 memory bank (240 MB).</em></p>
</div>

---

### 5.4. SOTA PatchCore Localization (1% Coreset Subsampling)
With 3x3 neighborhood pooling and 1% minimax coreset subsampling, PatchCore achieves high-precision defect boundary delineation across both rigid objects and structural textiles:

#### Rigid Object Category: 'bottle' (Broken rim, hairline glass crack, bottom contamination)
<div align="center">
  <img src="results/patchcore/bottle_resnet18_c0.01_localization.png" alt="PatchCore Bottle Localization" width="95%" />
  <p><em>Figure 5: High-resolution defect heatmaps and overlays produced by PatchCore-ResNet18 on 'bottle' (1% Coreset).</em></p>
</div>

#### Texture Category: 'carpet' (Hole puncture, thread flaw, cut anomaly)
<div align="center">
  <img src="results/patchcore/carpet_resnet18_c0.01_localization.png" alt="PatchCore Carpet Localization" width="95%" />
  <p><em>Figure 6: High-resolution defect heatmaps and overlays produced by PatchCore-ResNet18 on 'carpet' (1% Coreset).</em></p>
</div>

---

## 6. Repository Structure

```text
Anomaly Detection/
|-- app/
|   `-- demo.py                     # Real-time Gradio inspection app (PASS/DEFECT and Heatmap)
|-- configs/
|   `-- config.yaml                 # Central experiment configurations and hyperparameters
|-- data/                           # MVTec AD dataset directory (bottle, carpet, ...)
|   |-- bottle/
|   `-- carpet/
|-- checkpoints/                    # Serialized model weights and memory bank artifacts
|   |-- bottle_ae/best.pth          # Trained ConvAutoencoder checkpoint
|   |-- bottle_vae/best.pth         # Trained ConvVAE checkpoint
|   `-- patchcore/                  # PatchCore memory bank artifacts (.pt)
|       |-- bottle_resnet18_c0.01.pt
|       `-- carpet_resnet18_c0.01.pt
|-- results/                        # Quantitative CSV logs and qualitative figures
|   |-- eda_gallery.png
|   |-- bottle_ae_localization.png
|   |-- bottle_vae_localization.png
|   |-- feature_knn/
|   `-- patchcore/                  # PatchCore benchmark metrics and heatmap plots
|-- scripts/
|   |-- train_autoencoder.py        # AE/VAE training script with auto-resume and TensorBoard
|   |-- evaluate_autoencoder.py     # AE/VAE evaluation suite (AUROC and PRO)
|   |-- build_feature_bank.py       # Offline ResNet feature memory bank builder
|   |-- evaluate_feature_knn.py     # k-NN anomaly evaluation and visualization export
|   `-- run_patchcore.py            # End-to-end PatchCore training, subsampling and benchmark CLI
|-- src/
|   |-- data/
|   |   |-- mvtec.py                # PyTorch MVTecDataset loader and statistical aggregators
|   |   `-- transforms.py           # Standard transforms (Resize 256, CenterCrop 224, Normalize)
|   |-- models/
|   |   |-- autoencoder.py          # ConvAutoencoder and ConvVAE architectures
|   |   |-- coreset.py              # Greedy Farthest-Point Sampler and JL Random Projection
|   |   |-- feature_extractor.py    # ResNetExtractor (Hooks + 3x3 Pooling) and FeatureMemoryBank
|   |   |-- patchcore.py            # Complete PatchCore module (fit, predict, reweight, save, load)
|   |   `-- losses.py               # SSIMLoss and Composite Reconstruction Objectives
|   `-- utils/
|       |-- metrics.py              # Image AUROC, Pixel AUROC, and PRO Score computation
|       `-- visualization.py        # Heatmap colorization, binary overlays, and galleries
|-- tests/                          # Automated Pytest suite (33 tests covering all modules)
|   `-- models/
|-- requirements.txt                # Python dependencies
|-- pytest.ini                      # Optimized Pytest configuration
`-- README.md                       # Comprehensive project documentation
```

---

## 7. Installation and Getting Started

### Step 1: Environment Setup
```bash
# Clone repository and create virtual environment
python -m venv .env

# Activate virtual environment (Windows PowerShell):
.\.env\Scripts\Activate.ps1

# Activate virtual environment (Linux/macOS):
# source .env/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests (33 Tests)
```bash
# Execute test suite across all model modules
python -m pytest tests/models/ -v
```

### Step 3: Train and Evaluate Reconstruction Baselines (ConvAE and ConvVAE)
```bash
# Train Autoencoder on 'bottle'
python scripts/train_autoencoder.py --model ae --category bottle --epochs 50 --batch_size 16 --lr 1e-3

# Train VAE on 'bottle'
python scripts/train_autoencoder.py --model vae --category bottle --epochs 50 --batch_size 16 --lr 1e-3

# Evaluate trained Autoencoder and generate localization heatmaps
python scripts/evaluate_autoencoder.py --category bottle --model ae
```

### Step 4: Run SOTA PatchCore Pipeline (1% Coreset Subsampling)
```bash
# Run PatchCore on 'bottle' (Object) with ResNet-18 backbone and 1% Coreset
python scripts/run_patchcore.py --category bottle --backbone resnet18 --coreset_ratio 0.01

# Run PatchCore on 'carpet' (Texture) with 1% Coreset
python scripts/run_patchcore.py --category carpet --backbone resnet18 --coreset_ratio 0.01

# Evaluate using an existing saved PatchCore checkpoint (skipping feature extraction)
python scripts/run_patchcore.py --category bottle --load_checkpoint checkpoints/patchcore/bottle_resnet18_c0.01.pt
```

---

## 8. References

1. **Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P.** (2022). *Towards Total Recall in Industrial Anomaly Detection*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022).
2. **Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C.** (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. IEEE/CVF CVPR 2019.
3. **Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C.** (2020). *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. International Journal of Computer Vision (IJCV).
4. **Kingma, D. P., & Welling, M.** (2013). *Auto-Encoding Variational Bayes*. arXiv preprint arXiv:1312.6114.
5. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE Transactions on Image Processing (TIP).
