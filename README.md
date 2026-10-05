# ANOMALY-VISION

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep Learning-Based Industrial Visual Anomaly Detection with PyTorch, exploring unsupervised & one-class classification on the **MVTec AD** dataset.

---

## 📌 Project Architecture

```text
Anomaly Detection/
├── configs/               # Hyperparameters and experiment configurations (.yaml)
├── data/                  # MVTec AD dataset directory
├── notebooks/             # Step-by-step Jupyter notebooks for exploration
├── scripts/               # Training, evaluation, and application scripts
├── src/                   # Reusable source code modules
│   ├── data/              # Dataset loaders and transforms
│   ├── models/            # Autoencoder, Feature Extractors, PatchCore
│   └── utils/             # Metric calculations (AUROC, PRO) and Heatmap visualizations
├── .gitignore
├── proposal.md            # Project proposal and roadmap
├── requirements.txt       # Project dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset
Follow the guide in [`data/README.md`](data/README.md) to download and extract the MVTec AD dataset.

---

## 📖 Roadmap & Methodology
See [`proposal.md`](proposal.md) for full context and week-by-week implementation milestones.
