# ANOMALY-VISION

## Deep Learning-Based Industrial Visual Anomaly Detection

## 1. Project Overview

**ANOMALY-VISION** là một personal project xây dựng hệ thống **Visual Anomaly Detection** cho kiểm tra sản phẩm công nghiệp bằng Deep Learning.

Dự án sử dụng **MVTec AD** và tập trung vào việc học cách phát hiện các sản phẩm bất thường khi mô hình chủ yếu được học từ **normal images**.

Không hướng đến việc đề xuất phương pháp nghiên cứu mới, project tập trung vào:

- Hiểu các kỹ thuật cốt lõi của Anomaly Detection.
- Học và ứng dụng nhiều mô hình Computer Vision hiện đại.
- Xây dựng pipeline từ dữ liệu → feature extraction → anomaly detection → localization → visualization.
- Xây dựng một demo nhỏ để đưa model vào ứng dụng thực tế.

---

## 2. Main Idea

Thay vì huấn luyện classifier cho từng loại defect, hệ thống học đặc trưng của sản phẩm bình thường:

```text
Normal Images
      ↓
Visual Feature Learning
      ↓
Learn Normal Representation
      ↓
New Image
      ↓
Compare with Normal Features
      ↓
Normal / Anomaly
      ↓
Anomaly Localization
```

Mục tiêu cuối cùng là hệ thống có thể trả lời:

1. **Ảnh có bất thường không?**
2. **Mức độ bất thường là bao nhiêu?**
3. **Bất thường nằm ở đâu trên ảnh?**

---

## 3. Dataset

### MVTec AD

MVTec AD là dataset dành cho Industrial Anomaly Detection, bao gồm nhiều loại sản phẩm và vật liệu như:

- Bottle
- Cable
- Capsule
- Hazelnut
- Metal Nut
- Pill
- Screw
- Tile
- Wood
- Zipper

Các defect bao gồm scratch, crack, hole, contamination, deformation,...

Dataset cung cấp cả **image-level labels** và **pixel-level masks**, phù hợp để đánh giá cả detection và localization.

---

## 4. Learning & Implementation Pipeline

Project sẽ được triển khai từ đơn giản đến nâng cao.

### Step 1 — Computer Vision & Dataset

- Image preprocessing
- DataLoader với PyTorch
- Image augmentation
- Visualization
- Normal / anomaly data analysis

### Step 2 — CNN Autoencoder

Xây dựng Convolutional Autoencoder:

```text
Image
  ↓
CNN Encoder
  ↓
Latent Representation
  ↓
CNN Decoder
  ↓
Reconstructed Image
```

Sử dụng **reconstruction error** để phát hiện anomaly.

Mục tiêu: hiểu cách reconstruction-based anomaly detection hoạt động.

### Step 3 — Pretrained Vision Models

Sử dụng các pretrained models để học cách trích xuất visual features:

- ResNet
- Vision Transformer (ViT)

Pipeline:

```text
Image
  ↓
Pretrained Vision Model
  ↓
Feature Embedding
  ↓
Normal Feature Representation
```

Qua bước này, project cũng đóng vai trò như một mini Computer Vision project để học **CNN và Transformer-based vision models**.

### Step 4 — Feature-based Anomaly Detection

Sử dụng feature của pretrained model kết hợp với:

- k-Nearest Neighbors
- Feature distance
- Normal feature memory

Ý tưởng:

$$
AnomalyScore(x)
=
\min_{z\in\mathcal{M}} d(f(x),z)
$$

Trong đó \(\mathcal{M}\) là tập feature của normal images.

### Step 5 — Patch-based Detection

Chuyển từ image-level features sang **patch-level features**:

```text
Image
  ↓
Vision Encoder
  ↓
Feature Map
  ↓
Patch Features
  ↓
Memory Bank
  ↓
Nearest Neighbor
  ↓
Anomaly Score
```

Triển khai một hệ thống theo hướng **PatchCore**, giúp phát hiện anomaly ở mức local.

Đây là thành phần chính của project.

### Step 6 — Anomaly Localization

Từ anomaly score của từng patch, xây dựng **anomaly heatmap**:

```text
Original Image
      ↓
Patch Features
      ↓
Patch Anomaly Scores
      ↓
Anomaly Heatmap
      ↓
Overlay
```

Hệ thống cuối cùng có thể vừa phát hiện anomaly vừa chỉ ra vị trí defect.

---

## 5. Evaluation

Đánh giá hệ thống ở 2 mức:

### Image-level

**AUROC** cho khả năng phân biệt:

```text
Normal vs Anomaly
```

### Pixel-level

Đánh giá chất lượng anomaly localization bằng các metric phù hợp như:

- Pixel AUROC
- PRO

Ngoài metric, trực quan hóa:

```text
Original Image
Ground-truth Mask
Predicted Anomaly Map
Overlay
```

---

## 6. Final Application

Xây dựng một demo đơn giản bằng **Gradio/Streamlit**:

```text
Upload Image
      ↓
Anomaly Detection
      ↓
┌──────────────────────┐
│ Prediction: Anomaly  │
│ Score: 0.91          │
└──────────────────────┘
      ↓
Anomaly Heatmap
      ↓
Defect Localization
```

---

## 7. Technology Stack

- **Python**
- **PyTorch**
- **torchvision / timm**
- **OpenCV**
- **scikit-learn**
- **NumPy / Pandas**
- **Matplotlib**
- **Gradio / Streamlit**

---

## 8. Timeline

### Week 1

Dataset, preprocessing, PyTorch pipeline, Computer Vision fundamentals.

### Week 2

CNN Autoencoder + reconstruction-based anomaly detection.

### Week 3

ResNet / ViT feature extraction + kNN anomaly detection.

### Week 4

Patch-level features + Memory Bank + PatchCore-style detection.

### Week 5

Anomaly localization + evaluation + visualization.

### Week 6

Gradio/Streamlit demo + GitHub documentation.

---

## 9. Final Deliverables

```text
ANOMALY-VISION
│
├── Data & preprocessing pipeline
├── CNN Autoencoder
├── ResNet feature extractor
├── ViT feature extractor
├── kNN anomaly detection
├── PatchCore-style detector
├── Anomaly heatmap localization
├── Evaluation pipeline
├── Interactive demo
└── GitHub documentation
```

### Main Learning Outcomes

Sau project, có thể nắm được pipeline quan trọng của **Visual Anomaly Detection**:

```text
Image
  ↓
CNN / ViT
  ↓
Visual Representation
  ↓
Feature Distance
  ↓
Memory Bank
  ↓
Patch-level Detection
  ↓
Anomaly Localization
```

Đồng thời có thêm kinh nghiệm thực hành với **CNN, Vision Transformer, Autoencoder, feature extraction, kNN và patch-based vision models**.
