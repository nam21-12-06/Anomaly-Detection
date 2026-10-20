# ANOMALY-VISION — Kế Hoạch 6 Tuần (Advanced Learning Plan)

> **Mục tiêu tổng quan**: Xây dựng sự hiểu biết sâu về Computer Vision — từ CNN cổ điển đến Vision Transformer (ViT) — thông qua bài toán thực tế: Industrial Visual Anomaly Detection trên MVTec AD.

---

## 📊 Đánh Giá Tổ Chức Project Hiện Tại

### ✅ Điểm Mạnh

| Thành phần | Nhận xét |
|---|---|
| **Cấu trúc thư mục** | Rất tốt. `src/`, `scripts/`, `configs/`, `tests/`, `notebooks/` tách biệt rõ ràng theo vai trò |
| **Test suite** | 26 tests phân cấp theo `tests/data/`, `tests/models/`, `tests/utils/` — professional |
| **ConvAE & ConvVAE** | Code sạch, rõ ràng. Encoder/Decoder đối xứng, VAE có reparameterization trick đúng chuẩn |
| **Loss functions** | L1 + SSIM composite loss + KL divergence — lựa chọn tốt hơn MSE thuần túy |
| **Anomaly scoring** | Top-5% pixel strategy thay vì global mean — thông minh, tránh dilution |
| **Resume training** | `last.pth` + `best.pth` + `epoch_N.pth` — pipeline chắc chắn |
| **Config YAML** | Đã có slot sẵn cho `patchcore` với backbone + layers — thiết kế nhìn xa |
| **Documentation** | README có lý thuyết toán học (SSIM, KL, reparameterization) |

### ⚠️ Điểm Cần Cải Thiện

| Thành phần | Vấn đề | Đề xuất |
|---|---|---|
| **Evaluation script** | `evaluate_autoencoder.py` tồn tại nhưng chưa tích hợp vào main workflow | Tạo unified evaluation notebook |
| **Results** | Chỉ có `eda_gallery.png` — chưa có AUROC plots từ baseline | Cần chạy evaluate + lưu results có hệ thống |
| **Notebooks** | Chỉ có 1 notebook EDA — thiếu training monitoring & result analysis | Thêm notebook cho mỗi phase |
| **Không có `src/models/feature_extractor.py`** | Phase 3+ chưa được code | Đây là next step chính |
| **`config.yaml` image_size** | `[256, 256]` crop về `[224, 224]` — nhưng ViT standard cần `224x224` native | Cần xử lý khi dùng ViT |
| **Chưa có logging** | Training chỉ dùng tqdm, không có TensorBoard/W&B | Thêm ít nhất TensorBoard |
| **Chỉ train trên `bottle`** | Baseline chỉ validate 1 category | Sẽ mở rộng sang multi-category |

### 🏗️ Kiến Trúc Tổng Thể — Đánh Giá

```
AE/VAE (Reconstruction)  ←── ✅ Đã có
     ↓
Feature Extractor CNN     ←── 🔲 Tuần 3 (ResNet)
Feature Extractor ViT     ←── 🔲 Tuần 3-4 (ViT — điểm học quan trọng nhất)
     ↓
kNN / Feature Distance    ←── 🔲 Tuần 3
     ↓
PatchCore Memory Bank     ←── 🔲 Tuần 4 (SOTA approach)
     ↓
Anomaly Heatmap           ←── 🔲 Tuần 5
     ↓
Gradio Demo               ←── 🔲 Tuần 6
```

**Kết luận**: Tổ chức project **rất tốt cho level học tập**. Foundation code quality tốt. Vấn đề chính là chưa có kết quả được visualize & so sánh, và chưa bước sang các phase quan trọng hơn. Plan 6 tuần dưới đây sẽ giải quyết điều đó.

---

## 📅 KẾ HOẠCH 6 TUẦN CHI TIẾT

---

## 🗓️ Tuần 1 — Hoàn Thiện Baseline & Đánh Giá Định Lượng

> **Thực tế hiện tại**: AE & VAE đã train trên `bottle`. Nhưng chưa có kết quả AUROC cụ thể, chưa có visual so sánh reconstruction.

### 🎯 Mục tiêu tuần này
- Đánh giá đầy đủ 2 baseline đã train: Image-AUROC + Pixel-AUROC
- Visualize reconstruction quality & anomaly heatmaps
- Hiểu tại sao reconstruction-based methods có giới hạn

### 📚 Lý Thuyết — Anomaly Detection Metrics

#### AUROC là gì?

AUROC (Area Under the ROC Curve) là metric chuẩn để đánh giá binary anomaly detection:

```
ROC Curve: TPR (True Positive Rate) vs FPR (False Positive Rate)
  tại mọi possible threshold t

TPR = TP / (TP + FN) = sensitivity
FPR = FP / (FP + TN) = 1 - specificity

AUROC = 1.0  →  perfect separation
AUROC = 0.5  →  random guess
```

**Tại sao dùng AUROC thay vì Accuracy?**  
- Anomaly detection thường có **class imbalance** nặng (95% normal, 5% anomaly)  
- Accuracy bị bias về class majority  
- AUROC đo khả năng **ranking** — model có xếp hạng ảnh anomaly cao hơn normal không?

#### Pixel-AUROC

Áp dụng AUROC ở mức pixel:
```
- Ground truth: binary mask (0 = normal pixel, 1 = defect pixel)
- Prediction: anomaly heatmap (continuous score per pixel)
- AUROC đo: pixel-level discrimination capacity
```

#### PRO (Per-Region Overlap)

PRO là metric tinh tế hơn Pixel-AUROC:
```
Ý tưởng: với mỗi connected defect region R_i,
tính overlap giữa predicted anomaly mask và R_i.

PRO = average(overlap_i) tại FPR = 0.3

→ Defects lớn và nhỏ được weight bằng nhau
→ Pixel-AUROC bị bias bởi large defects
```

### 🔧 Tasks thực hành

- [ ] Chạy `evaluate_autoencoder.py` trên `bottle` — ghi lại Image-AUROC & Pixel-AUROC cho AE và VAE
- [ ] Tạo notebook `02_baseline_evaluation.ipynb`:
  - Grid của ảnh test: Original | Reconstruction | Anomaly Map | GT Mask
  - ROC curve plot cho AE vs VAE
  - Bảng kết quả so sánh
- [ ] Thêm TensorBoard logging vào `train_autoencoder.py`
- [ ] Extend training sang thêm 1-2 categories (e.g., `hazelnut`, `capsule`)

### 📊 Expected Results (Lý thuyết)

| Model | Image-AUROC (bottle) | Nhận xét |
|---|---|---|
| ConvAE | ~65-75% | Reconstruction quá smooth, anomaly signal yếu |
| ConvVAE | ~60-70% | VAE blurry hơn → anomaly map kém hơn đôi khi |

> **Key insight học được**: Reconstruction-based methods bị giới hạn bởi "generalization paradox" — model tốt quá thì reconstruct được cả anomaly, model kém quá thì không detect được gì.

---

## 🗓️ Tuần 2 — CNN Feature Extraction & Transfer Learning

> **Bước nhảy quan trọng**: Từ "train từ đầu" → "dùng pre-trained CNN backbone".

### 📚 Lý Thuyết — CNN & Transfer Learning

#### 1. CNN Hierarchy of Features

ResNet được train trên ImageNet học ra một hierarchy of features:

```
Layer 1-2 (shallow):  Edges, colors, textures, Gabor-like filters
Layer 3-4 (middle):   Parts, patterns, shapes  
Layer 5+ (deep):      Semantic concepts, object parts
```

**Tại sao điều này quan trọng cho anomaly detection?**

```
Khi ta extract features ở layer2, layer3:
→ Features vẫn spatial (feature map, không phải global pooling)
→ Encode local texture & shape information
→ Anomaly = deviation from normal texture/shape patterns
```

#### 2. ResNet Architecture — Hiểu Cơ Bản

```
Input (3, 224, 224)
    ↓
Stem conv (7x7, stride 2)  → (64, 112, 112)
    ↓
MaxPool (3x3, stride 2)    → (64, 56, 56)
    ↓
Layer1: [Conv3x3, Conv3x3] × 3  → (256, 56, 56)
    ↓
Layer2: [Conv3x3, Conv3x3] × 4  → (512, 28, 28)
    ↓
Layer3: [Conv3x3, Conv3x3] × 6  → (1024, 14, 14)
    ↓
Layer4: [Conv3x3, Conv3x3] × 3  → (2048, 7, 7)
    ↓
GlobalAvgPool → FC(1000) → Softmax
```

**Residual Connection** (điểm cốt lõi của ResNet):
```python
# Thay vì học H(x), học residual F(x) = H(x) - x
output = F(x) + x  # skip connection
# Gradient highway: gradient từ deep layers về shallow layers dễ hơn
# Giải quyết vanishing gradient problem
```

#### 3. Feature Extraction cho Anomaly Detection

```python
# Không fine-tune, chỉ extract features
model = resnet50(pretrained=True)
model.eval()  # BatchNorm ở eval mode

hooks = []
features = {}
def make_hook(name):
    def hook(module, input, output):
        features[name] = output
    return hook

model.layer2.register_forward_hook(make_hook('layer2'))
model.layer3.register_forward_hook(make_hook('layer3'))

# Forward pass
with torch.no_grad():
    model(image)

# features['layer2']: (B, 512, 28, 28)
# features['layer3']: (B, 1024, 14, 14)
```

#### 4. kNN Anomaly Detection

```
Idea đơn giản nhưng effective:
1. Extract features của TẤT CẢ normal training images → Memory Bank M
2. Tại test time: extract feature của ảnh test → f_test
3. AnomalyScore = distance(f_test, nearest_neighbor(f_test, M))

→ Nếu ảnh test normal: feature nằm gần memory bank
→ Nếu ảnh test anomaly: feature xa memory bank
```

### 🔧 Tasks thực hành

- [ ] Tạo `src/models/feature_extractor.py`:
  - `ResNetExtractor` với hook-based feature extraction
  - Support `layer2`, `layer3`, `avgpool`
  - `FeatureMemoryBank` class để lưu & query features
- [ ] Tạo `scripts/build_feature_bank.py`:
  - Extract features từ toàn bộ training set → save `.npy`
- [ ] Tạo `scripts/evaluate_feature_knn.py`:
  - Load bank, run kNN inference, tính AUROC
- [ ] Notebook `03_feature_extraction.ipynb`:
  - Visualize feature maps ở các layer khác nhau
  - t-SNE plot của features: normal vs anomaly
  - AUROC comparison: AE vs kNN-ResNet

### 📊 Expected Improvement

| Method | Image-AUROC (bottle) |
|---|---|
| ConvAE | ~65-75% |
| kNN + ResNet (layer2+3) | ~85-92% |

> **Key insight**: Pre-trained features encode prior knowledge từ ImageNet. Chúng "hiểu" texture, structure tốt hơn features learned from scratch trên dataset nhỏ.

---

## 🗓️ Tuần 3 — Vision Transformer (ViT) — Deep Dive

> **Đây là tuần học quan trọng nhất về lý thuyết**. ViT là paradigm shift trong Computer Vision.

### 📚 Lý Thuyết — Vision Transformer từ Gốc

#### 1. Tại Sao Transformer Cho Vision?

NLP → Vision: Transformer được thiết kế cho sequences. Làm thế nào apply cho ảnh?

**Ý tưởng core của ViT** (Dosovitskiy et al., 2020):  
*"Treat an image as a sequence of patches"*

#### 2. Kiến Trúc ViT — Chi Tiết

```
Input Image: (3, 224, 224)
    ↓
① Patch Embedding
   - Split thành P×P patches (P=16): 14×14 = 196 patches
   - Mỗi patch: (3, 16, 16) → flatten → Linear(768) → 196 tokens
   
    ↓
② Add [CLS] Token
   - Thêm learnable [CLS] token: tổng 197 tokens
   - CLS token sẽ aggregate global information
   
    ↓
③ Add Positional Encoding
   - Learnable position embeddings (1D)
   - Cho model biết mỗi token ở vị trí nào
   
    ↓
④ Transformer Encoder × L layers
   - MSA: Multi-Head Self-Attention
   - MLP: Feed-Forward Network
   - LayerNorm + Residual connections
   
    ↓
⑤ [CLS] Token Output → MLP Head → Classification
```

#### 3. Self-Attention — Cơ Chế Quan Trọng Nhất

```
Với mỗi token, Self-Attention hỏi:
"Token này nên attend đến tokens nào khác?"

Cho input sequence X ∈ R^(N × d):

Q = X · W_Q  (Query)
K = X · W_K  (Key)  
V = X · W_V  (Value)

Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

→ Mỗi token nhận thông tin từ tất cả tokens khác
→ Weighted by attention scores (how "relevant" each token is)
```

**Trực quan hóa**:
```
Patch 1 (góc trái trên):  attend nhiều đến patch lân cận
Patch defect:              pattern attention khác so với normal patch
→ Self-attention có khả năng capture long-range dependencies
→ CNN chỉ capture local neighborhood (kernel size)
```

#### 4. Multi-Head Self-Attention (MSA)

```
Thay vì 1 attention head, dùng H=12 heads:

head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
MSA(X) = Concat(head_1, ..., head_H) · W_O

→ Mỗi head học 1 "aspect" khác nhau:
   - Head 1: attend to nearby patches (local)
   - Head 2: attend to similar texture patches
   - Head 3: attend to global structure
   ...
```

#### 5. ViT Patch Features cho Anomaly Detection

**Khác biệt core với CNN**:

| CNN (ResNet) | ViT |
|---|---|
| Local receptive field | Global context từ layer đầu |
| Feature map (spatial grid) | Token sequence |
| Inductive bias: locality | Không có inductive bias |
| Cần ít data hơn | Cần nhiều data (hoặc pre-training) |
| layer2/3 → patch features | Intermediate transformer blocks |

**Cách extract patch features từ ViT**:
```python
# ViT-B/16: 196 patches + 1 CLS token
# Tại mỗi transformer block output: 197 × 768

# Lấy patch tokens (bỏ CLS), reshape về spatial
patch_tokens = vit_output[1:]           # 196 × 768
spatial_features = patch_tokens.reshape(14, 14, 768)  # 14×14 grid

# → Patch features giống feature map của CNN
# → Dùng được cho PatchCore!
```

#### 6. DINO — Self-Supervised ViT (Rất Phù Hợp Cho Anomaly Detection)

DINO là self-supervised pretraining cho ViT, đặc biệt tốt cho anomaly detection vì:

```
DINO features:
- Không cần labels khi pretrain
- CLS token attention maps → segmentation-like properties
- Patch features encode fine-grained local texture tốt
- AUROC trên MVTec với DINO features thường > supervised ViT
```

### 🔧 Tasks thực hành

- [ ] Extend `feature_extractor.py`:
  - `ViTExtractor` class sử dụng `timm.create_model('vit_base_patch16_224', pretrained=True)`
  - Hook ở intermediate transformer blocks (block 6, 9, 11)
  - Reshape patch tokens về spatial grid
- [ ] Notebook `04_vit_features.ipynb`:
  - Visualize attention maps (DINO-style)
  - So sánh attention maps: normal vs anomaly patches  
  - t-SNE: ResNet features vs ViT features
- [ ] Benchmark: kNN với ViT features vs ResNet features
- [ ] Viết learning note về Self-Attention mechanism trong notebook

### 📊 Expected Results

| Method | Image-AUROC (bottle) |
|---|---|
| kNN + ResNet50 | ~87% |
| kNN + ViT-B/16 | ~90-93% |
| kNN + DINO-ViT | ~93-96% |

---

## 🗓️ Tuần 4 — PatchCore: SOTA Memory Bank Method

> **PatchCore** (Roth et al., CVPR 2022) đạt 99.1% Image-AUROC trên MVTec. Đây là phương pháp SOTA quan trọng nhất cần implement.

### 📚 Lý Thuyết — PatchCore Deep Dive

#### 1. Tại Sao Feature-Level Beats Reconstruction-Level?

```
Reconstruction-based (AE/VAE):
  - Phụ thuộc khả năng reconstruct của decoder
  - "Generalization paradox": model tốt → reconstruct anomaly
  - Blurry reconstructions → noisy anomaly maps

Feature-based (PatchCore):
  - Pre-trained features: rất discriminative ngay cả without fine-tuning
  - Memory bank: explicit representation của normality
  - Không cần training! (inference only)
```

#### 2. PatchCore Algorithm

**Bước 1: Memory Bank Construction (offline, training set)**
```python
for image in train_set:  # all normal images
    features = backbone(image)  # (C, H, W) feature map
    # Lấy neighborhood-aware patch features
    for each patch (h, w):
        # Aggregrate feature từ 3x3 neighborhood
        patch_feat = adaptive_avg_pool(features[:, h-1:h+2, w-1:w+2])
        memory_bank.append(patch_feat)

# memory_bank: N_train × H × W × C vectors
```

**Bước 2: Coreset Subsampling (quan trọng!)**
```
Memory bank có thể rất lớn: 1000 images × 196 patches = 196,000 vectors

Coreset subsampling: giữ lại subset đại diện nhất
→ Giảm còn ~1-2% vectors
→ Giữ coverage của feature space

Thuật toán: Greedy farthest-point sampling
1. Chọn điểm đầu tiên ngẫu nhiên
2. Mỗi bước: chọn điểm xa nhất với all selected points
3. Repeat đến khi đủ quota
```

**Bước 3: Inference (test time)**
```python
# Test image
features = backbone(test_image)  # patch features

for each patch:
    score = min_distance(patch_feat, memory_bank)
    # → Small: patch giống normal patches
    # → Large: patch khác normal patches (anomaly!)

# Image-level score: max patch score
image_score = max(patch_scores)

# Pixel-level heatmap: upsample patch scores to image size
anomaly_heatmap = upsample(patch_scores, to=224×224)
```

#### 3. Vì Sao PatchCore Đạt SOTA?

3 yếu tố chính:
1. **Rich pretrained features**: Wide ResNet50 (ImageNet pretrained) → rất strong features
2. **Patch-level**: local defect detection tốt hơn global image-level
3. **Memory bank**: non-parametric, không cần assume distribution
4. **Coreset**: efficient & representative coverage

#### 4. Coreset Subsampling — Chi Tiết Toán Học

```
Cho tập memory bank M và coreset target C:
Mục tiêu: min over C ⊂ M:
    max_{m∈M} min_{c∈C} ||m - c||_2

→ C phải "cover" M tốt nhất có thể
→ Greedy approximate: O(|M| × |C|)
```

#### 5. Anomaly Scoring — Image Level & Pixel Level

```
Patch score: s(p) = min_{m∈C} ||f(p) - m||_2

Image score: S = max_{p} s(p)
           (maximum patch anomaly score)

Pixel heatmap:
1. Reshape patch scores → spatial grid (14×14 for ViT-B/16)
2. Upsample với bilinear interpolation → (224×224)
3. Gaussian smoothing để loại noise
```

### 🔧 Tasks thực hành

- [ ] Tạo `src/models/patchcore.py`:
  - `PatchCore` class: `fit(train_loader)` + `predict(image)` + `get_heatmap(image)`
  - Coreset sampling với greedy farthest-point
  - Support cả ResNet backbone lẫn ViT backbone
- [ ] Tạo `scripts/run_patchcore.py`:
  - `--backbone wide_resnet50_2 | vit_base_patch16_224`
  - `--category bottle | capsule | ...`
  - Output: AUROC + heatmap gallery
- [ ] Notebook `05_patchcore.ipynb`:
  - Visualize memory bank coverage (PCA projection)
  - Coreset vs full bank: speed vs accuracy tradeoff
  - Qualitative: heatmaps trên nhiều defect types
- [ ] Chạy trên **ít nhất 3 categories** và ghi kết quả

### 📊 Expected Results

| Method | Image-AUROC | Pixel-AUROC |
|---|---|---|
| ConvAE | ~70% | ~55% |
| kNN + ResNet | ~88% | ~75% |
| PatchCore (WR50) | ~97-99% | ~90-95% |
| PatchCore (ViT) | ~95-98% | ~88-93% |

---

## 🗓️ Tuần 5 — Anomaly Localization, Full Evaluation & Visualization

> Tuần này focus vào **chất lượng kết quả** và **hiểu sâu hơn về failure cases**.

### 📚 Lý Thuyết — Anomaly Localization

#### 1. Heatmap Generation Pipeline

```
Patch Anomaly Scores (14×14 grid for ViT-B/16)
    ↓
① Bilinear Upsample → (224×224)
    ↓
② Gaussian Blur (σ=4)
   - Smoothing: anomaly không bao giờ hard-edge
   - Loại bỏ high-frequency noise artifacts
    ↓
③ Normalize [0, 1]
    ↓
④ Colormap (e.g., viridis, hot)
    ↓
⑤ Blend với original image (alpha=0.5)
```

#### 2. Threshold Selection

```
AUROC đo performance independent of threshold.
Nhưng để deploy, cần chọn threshold t:

Strategies:
1. Percentile: t = percentile(train_scores, 95%)
   → 95% normal images classified as normal

2. ROC-based: t = argmax_{t} (TPR - FPR)
   → Youden's J statistic

3. F1-based: t = argmax_{t} F1(t)
   → Optimize F1 score
```

#### 3. PRO Metric — Implementation

```python
def compute_pro(anomaly_map, gt_mask, fpr_limit=0.3):
    """
    Per-Region Overlap:
    - Tách ground truth thành connected regions
    - Tính overlap của từng region với predicted mask
    - Average over regions
    """
    regions = label(gt_mask)  # scipy connected components
    per_region_overlaps = []
    
    for region_id in unique(regions):
        region_mask = (regions == region_id)
        # Sweep threshold, tính overlap tại mỗi threshold
        overlap = intersection(predicted_mask, region_mask) / region_mask.sum()
        per_region_overlaps.append(overlap)
    
    # Tính PRO tại FPR ≤ 0.3
    pro = average_over_fpr_range(per_region_overlaps, fpr_limit)
    return pro
```

#### 4. Multi-Category Evaluation

```
MVTec có 15 categories với đặc điểm khác nhau:

Textures: carpet, grid, leather, tile, wood
  → Defects là texture irregularities
  → ViT patch features đặc biệt tốt ở đây

Objects: bottle, cable, capsule, metal_nut, ...
  → Defects là structural anomalies (crack, hole, deformation)
  → CNN features cũng compete được

→ So sánh trên multi-category → hiểu strength/weakness của mỗi approach
```

### 🔧 Tasks thực hành

- [ ] Implement `PRO score` trong `src/utils/metrics.py`
- [ ] Tạo `scripts/full_evaluation.py`:
  - Chạy tất cả methods trên multiple categories
  - Output: comparison table CSV + plots
- [ ] Notebook `06_full_results.ipynb`:
  - **Master comparison table**: AE vs VAE vs kNN-ResNet vs PatchCore-WR50 vs PatchCore-ViT
  - Heatmap gallery: mỗi method × mỗi defect type
  - Failure case analysis: khi nào methods fail?
  - Confusion cases: FP và FN patterns
- [ ] Ablation study nhỏ:
  - PatchCore: `coreset_ratio` effect (0.001 → 0.05)
  - PatchCore: `n_neighbors` (1, 3, 9)
  - ViT layer depth effect (shallow vs deep blocks)

### 📊 Target: Complete Comparison Table

| Method | Bottle AUROC | Capsule AUROC | Hazelnut AUROC | Mean AUROC |
|---|---|---|---|---|
| ConvAE | ? | ? | ? | ? |
| ConvVAE | ? | ? | ? | ? |
| kNN-ResNet50 | ? | ? | ? | ? |
| PatchCore-WR50 | ? | ? | ? | ? |
| PatchCore-ViT | ? | ? | ? | ? |

---

## 🗓️ Tuần 6 — Gradio Demo & Documentation

> Tuần cuối: đưa model vào ứng dụng thực tế và tổng kết learning.

### 📚 Lý Thuyết — Model Deployment Considerations

#### 1. Inference Latency Profile

```
Method          | Memory     | Inference Time | AUROC
AE/VAE          | ~50MB      | ~5ms/image     | ~70%
kNN-ResNet      | ~500MB     | ~50ms/image    | ~88%
PatchCore-WR50  | ~1-5GB     | ~100ms/image   | ~98%
PatchCore-ViT   | ~1-3GB     | ~80ms/image    | ~95%

→ Accuracy vs Speed vs Memory tradeoff
→ Production deployment: thường dùng coreset nhỏ + fast backbone
```

#### 2. Gradio Interface Design

```
Upload Image (drag & drop)
    ↓
Model Selection (dropdown): AE | kNN-ResNet | PatchCore
Category Selection: bottle | capsule | ...
    ↓
┌─────────────────────────────────────────┐
│  Original    │  Heatmap    │  Overlay   │
│  Image       │  (viridis)  │  (blended) │
├─────────────────────────────────────────┤
│  Prediction: ANOMALY (confidence: 0.87) │
│  Anomaly Score: 0.234                   │
│  Processing time: 124ms                 │
└─────────────────────────────────────────┘
```

### 🔧 Tasks thực hành

- [ ] Tạo `app/demo.py` với Gradio:
  - Multi-model inference
  - Upload + heatmap overlay display
  - Anomaly score + confidence display
  - Inference time display
- [ ] Update README.md hoàn chỉnh:
  - Full results table
  - Architecture diagrams
  - GIF demo của Gradio interface
- [ ] Tạo `notebooks/07_learning_summary.ipynb`:
  - Tóm tắt toàn bộ pipeline đã học
  - Key insights từ mỗi method
  - Reflection: nếu làm lại sẽ thay đổi gì?
- [ ] (Optional) HuggingFace Spaces deployment

---

## 📐 TỔNG KẾT CẤU TRÚC PROJECT SAU 6 TUẦN

```
Anomaly Detection/
├── configs/
│   └── config.yaml              ← ✅ Có sẵn
├── data/                        ← ✅ MVTec AD
├── src/
│   ├── data/
│   │   ├── mvtec.py             ← ✅ Có sẵn
│   │   └── transforms.py        ← ✅ Có sẵn
│   ├── models/
│   │   ├── autoencoder.py       ← ✅ AE + VAE
│   │   ├── losses.py            ← ✅ SSIM + KL
│   │   ├── feature_extractor.py ← 🔲 Tuần 2-3 (ResNet + ViT)
│   │   └── patchcore.py         ← 🔲 Tuần 4
│   └── utils/
│       ├── metrics.py           ← ⚡ Extend với PRO
│       └── visualization.py     ← ⚡ Extend với heatmap overlay
├── scripts/
│   ├── train_autoencoder.py     ← ✅ + TensorBoard
│   ├── evaluate_autoencoder.py  ← ✅ Có sẵn
│   ├── build_feature_bank.py    ← 🔲 Tuần 2
│   ├── evaluate_feature_knn.py  ← 🔲 Tuần 2
│   ├── run_patchcore.py         ← 🔲 Tuần 4
│   └── full_evaluation.py       ← 🔲 Tuần 5
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   ← ✅
│   ├── 02_baseline_evaluation.ipynb   ← 🔲 Tuần 1
│   ├── 03_feature_extraction.ipynb    ← 🔲 Tuần 2
│   ├── 04_vit_features.ipynb          ← 🔲 Tuần 3
│   ├── 05_patchcore.ipynb             ← 🔲 Tuần 4
│   ├── 06_full_results.ipynb          ← 🔲 Tuần 5
│   └── 07_learning_summary.ipynb      ← 🔲 Tuần 6
├── app/
│   └── demo.py                        ← 🔲 Tuần 6 (Gradio)
├── results/
│   ├── eda_gallery.png          ← ✅
│   ├── baseline_evaluation/     ← 🔲 Tuần 1
│   ├── feature_knn/             ← 🔲 Tuần 2-3
│   ├── patchcore/               ← 🔲 Tuần 4
│   └── full_comparison.csv      ← 🔲 Tuần 5
└── tests/                       ← ✅ 26 tests
```

---

## 🧠 KEY LEARNING MILESTONES

Sau 6 tuần, bạn sẽ nắm được:

### Computer Vision Fundamentals
- [x] Convolutional Neural Networks — convolution, pooling, receptive field
- [x] Residual connections — skip connections, vanishing gradient
- [ ] Vision Transformer — patches, positional encoding, self-attention
- [ ] Transfer learning — feature extraction vs fine-tuning

### Anomaly Detection Techniques
- [x] Reconstruction-based: AE, VAE, SSIM loss
- [ ] Feature-based: kNN với pretrained features
- [ ] Memory bank: PatchCore, coreset sampling
- [ ] Localization: patch-level heatmaps, PRO metric

### ML Engineering Skills
- [x] Modular PyTorch pipeline (dataset, transforms, dataloader)
- [x] Training with resume, checkpointing
- [ ] Feature extraction hooks
- [ ] Efficient inference (batch processing, memory management)
- [ ] Gradio deployment

---

## 📖 TÀI LIỆU ĐỌC THÊM (Theo Tuần)

| Tuần | Paper | Link |
|---|---|---|
| 1 | MVTec AD Paper (Bergmann 2019) | CVPR 2019 |
| 2 | ResNet (He et al., 2015) | arXiv:1512.03385 |
| 3 | ViT: "An Image is Worth 16x16 Words" | arXiv:2010.11929 |
| 3 | DINO: Self-Supervised ViT | arXiv:2104.14294 |
| 4 | PatchCore Paper (Roth 2022) | arXiv:2106.08265 |
| 5 | WinCLIP (anomaly with CLIP) | arXiv:2303.14814 |
| 6 | Anomalib Library | GitHub |
