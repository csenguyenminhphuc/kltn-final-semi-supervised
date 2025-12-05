# 🔴 PHÂN TÍCH VẤN ĐỀ NGHIÊM TRỌNG - Multi-View Soft Teacher

**Ngày phân tích:** 03/12/2025  
**Trạng thái:** Model train được (loss giảm) nhưng **mAP = 0.0** hoàn toàn

---

## 📊 HIỆN TƯỢNG QUAN SÁT

### 1. **Kết Quả Đánh Giá (Epoch 0)**
```
+-------------+-----+--------+--------+-------+-------+-------+
| category    | mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+-------------+-----+--------+--------+-------+-------+-------+
| Broken      | 0.0 | 0.0    | 0.0    | nan   | nan   | 0.0   |
| Chipped     | 0.0 | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
| Scratched   | 0.0 | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
| Severe_Rust | 0.0 | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
| Tip_Wear    | 0.0 | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+-------------+-----+--------+--------+-------+-------+-------+
```

### 2. **Teacher Predictions (Bất Thường)**
```
[Teacher Predictions] Total boxes: 800
Score range: [0.999, 1.000], Mean: 1.000, Median: 1.000
[After Filtering] Threshold: 0.7, Kept: 120/800 (15.0%)
```

### 3. **Training Loss Values (Iter 50)**
```
loss: 4.6771

Supervised Branch:
  loss_rpn_cls: 0.2659
  loss_rpn_bbox: 0.5833
  loss_cls: 2.0261 ← RẤT CAO!
  loss_bbox: 1.3502
  acc_fg: 8.5714% ← CỰC THẤP!
  acc_bg: 60.0000%

Unsupervised Branch:
  loss_rpn_cls: 0.2000
  loss_rpn_bbox: 0.8338
  loss_cls: 1.5117
  loss_bbox: 0.6804
  acc_fg: 86.0360% ← CAO BẤT THƯỜNG!
  acc_bg: 48.6486%
```

---

## 🔍 PHÂN TÍCH GỐC RỄ VẤN ĐỀ

### **VẤN ĐỀ 1: Teacher Model Output Confidence = 1.0**

**Hiện tượng:**
```python
Score range: [0.999, 1.000], Mean: 1.000
```

**Nguyên nhân:**
1. **Init không đúng:** Teacher được copy từ Student ban đầu
2. **Activation function:** Sigmoid/Softmax với logits quá lớn → confidence = 1.0
3. **EMA update không hoạt động đúng:** Teacher không được cập nhật dần dần

**Hậu quả:**
- Pseudo-labels có confidence = 1.0 → model overfit vào pseudo-labels SAI
- Unsupervised acc_fg = 86% cao bất thường vì model "chắc chắn" về pseudo-labels sai

**Cách kiểm tra:**
```python
# In mmdet/models/detectors/soft_teacher.py
@torch.no_grad()
def get_pseudo_instances():
    # Add debug
    print(f"Teacher cls scores BEFORE sigmoid: {cls_logits.min():.3f} to {cls_logits.max():.3f}")
    print(f"Teacher cls scores AFTER sigmoid: {cls_scores.min():.3f} to {cls_scores.max():.3f}")
```

**FIX:**
```python
# Option 1: Temperature scaling for teacher predictions
teacher_temperature = 2.0  # Smooth confidence distribution
cls_scores = cls_logits.sigmoid() / teacher_temperature

# Option 2: Clip logits before sigmoid
cls_logits = torch.clamp(cls_logits, min=-10, max=10)
cls_scores = cls_logits.sigmoid()

# Option 3: Add noise to teacher predictions
cls_scores = cls_scores + torch.randn_like(cls_scores) * 0.05
cls_scores = torch.clamp(cls_scores, 0, 1)
```

---

### **VẤN ĐỀ 2: Supervised Accuracy Cực Thấp (8.57%)**

**Hiện tượng:**
```
sup_acc_fg: 8.5714%  ← Chỉ 8.57% foreground dự đoán đúng!
sup_loss_cls: 2.0261  ← Loss classification rất cao
```

**Nguyên nhân có thể:**

#### A. **Class Imbalance Nghiêm Trọng**
```python
# Your config
num_classes = 5  # Broken, Chipped, Scratched, Severe_Rust, Tip_Wear
```

Kiểm tra distribution:
```bash
# Count labels in annotation file
grep -o '"category_id": [0-9]*' data_drill/anno_train/_annotations.coco.json | sort | uniq -c
```

Nếu có 1 class chiếm >80% → Focal Loss không đủ mạnh

**FIX:**
```python
# In config: Tăng gamma của Focal Loss
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=3.0,  # Tăng từ 2.0 → 3.0 (focus hơn vào hard examples)
    alpha=0.25,
    loss_weight=1.0
)

# Hoặc dùng Class-Balanced Focal Loss
loss_cls=dict(
    type='CrossEntropyLoss',
    use_sigmoid=True,
    class_weight=[1.0, 2.0, 1.5, 3.0, 2.5],  # Weight theo inverse frequency
    loss_weight=1.0
)
```

#### B. **Learning Rate Quá Thấp**
```python
lr: 1.9704e-05  # = 0.00001970 (RẤT THẤP!)
```

**FIX:**
```python
# In config
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
    # Thử lr=0.001 thay vì 0.00002
)

# Hoặc dùng AdamW
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05)
)
```

#### C. **MVViT Attention Làm Nhiễu Features**

**Giả thuyết:** MVViT cross-view attention đang "trộn lẫn" features từ 8 crops → features bị nhiễu

**Kiểm tra:**
```python
# Trong MultiViewBackbone.forward()
# Thêm debug
if self.fusion == 'mvvit':
    feats_before = feats  # Features từ ResNet
    feats_refined = self.mvvit(...)  # Features sau MVViT
    
    # Check difference
    diff = (feats_refined - feats_before).abs().mean()
    print(f"[DEBUG] MVViT change magnitude: {diff:.6f}")
    
    # If diff > 0.5 → MVViT đang thay đổi quá mạnh!
```

**FIX:**
```python
# Option 1: Giảm learning rate của MVViT
mvvit=dict(
    type='MVViT',
    # ... other params ...
    lr_multiplier=0.1  # MVViT học chậm hơn backbone
)

# Option 2: Residual connection mạnh hơn
# Trong multi_view_transformer.py
refined = 0.9 * original + 0.1 * refined  # Thay vì 0.5/0.5

# Option 3: Tắt MVViT tạm thời để test
detector.backbone = dict(
    type='MultiViewBackbone',
    backbone=dict(type='ResNet', ...),
    fusion='mean',  # Thay vì 'mvvit'
    views_per_sample=8
)
# → Nếu mAP tăng → MVViT là nguyên nhân!
```

---

### **VẤN ĐỀ 3: mAP = 0.0 Hoàn Toàn**

**Nguyên nhân có thể:**

#### A. **NMS/Score Threshold Quá Cao**
```python
# Check trong config
nms=dict(type='nms', iou_threshold=0.5),
score_thr=0.05  # Có thể quá cao?
```

**FIX:**
```python
# Giảm threshold xuống
test_cfg=dict(
    rcnn=dict(
        score_thr=0.001,  # Từ 0.05 → 0.001
        nms=dict(type='nms', iou_threshold=0.7),  # Từ 0.5 → 0.7
        max_per_img=100
    )
)
```

#### B. **Bounding Box ở Sai Coordinate Space**

**Vấn đề:** Predictions được crop coordinates, nhưng evaluation ở base image coordinates

**Kiểm tra:**
```python
# Trong evaluation log
[BBOX DEBUG] File: S110_bright_2_crop_1, Image shape: 720x256
  → Loaded 1 boxes from COCO JSON
    Box 0: [25, 492, 69.894, 622.718], label: 1

# Check: Box width = 69.894-25 = 44.894, height = 622.718-492 = 130.718
# Nhưng image width = 720, height = 256
# → Height 130.718 < 256 ✓, nhưng x-coord 622.718 > 720?! 🔴
```

**PHÁT HIỆN:** Bounding box coordinates **VỚT KHỎI IMAGE**!

**Nguyên nhân:**
- Annotation có thể ở base image coords (1080×2560)
- Nhưng crop size là 256×720
- Transform không đúng!

**FIX:**
```python
# Kiểm tra trong MultiViewFromFolder._load_annotations()
def _clip_bbox_to_image(self, bbox, img_width, img_height):
    """Clip bbox to image boundaries."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_width))
    x2 = max(0, min(x2, img_width))
    y1 = max(0, min(y1, img_height))
    y2 = max(0, min(y2, img_height))
    return [x1, y1, x2, y2]

# Apply in load_data_list()
for bbox in gt_bboxes:
    bbox = self._clip_bbox_to_image(bbox, img_width, img_height)
```

#### C. **Evaluation Mode Sai**

Log shows:
```
[Info] Evaluation mode: Per-view (no aggregation)
  - Total views evaluated: 640
```

**Vấn đề:** Đang evaluate từng crop riêng biệt, không aggregate về base image!

**FIX:**
```python
# Trong config, thay đổi evaluation metric
val_evaluator = dict(
    type='MultiViewCocoMetric',
    views_per_sample=8,
    aggregate_predictions=True,  # ← PHẢI BẬT!
    aggregation_method='wbf',
    metric='bbox',
    format_only=False,
    ann_file='data_drill/anno_valid/_annotations_filtered.coco.json'
)
```

---

### **VẤN ĐỀ 4: Annotations Có Thể Sai**

**Observation từ log:**
```
[BBOX DEBUG] File: ...bright_4_crop_1.jpg, Image shape: 720x256
  → Loaded 1 boxes
    Box 0: [89, 0, 153.76, 238.57], label: 2
```

**Kiểm tra:**
```bash
# Visualize 1 crop với GT bbox
python tools/analysis_tools/browse_dataset.py \
  configs/soft_teacher/soft_teacher_custom_multi_view.py \
  --output-dir vis_check \
  --not-show

# Check xem bbox có đúng không
```

**Vấn đề tiềm ẩn:**
1. **Filtered annotations** (`_annotations_filtered.coco.json`) có thể bị lỗi
2. **base_img_id mapping** có thể sai
3. **Crop coordinates** không match với bbox coordinates

**FIX:**
```bash
# Quay lại dùng original annotations
ann_file='data_drill/anno_train/_annotations.coco.json'  # Không dùng _filtered

# Hoặc regenerate filtered file
python tools/misc/regenerate_filtered_annotations.py
```

---

### **VẤN ĐỀ 5: MVViT Capacity Không Phù Hợp**

**Config hiện tại:**
```python
mvvit=dict(
    embed_dim=256,
    num_heads=4,     # 4 heads
    num_layers=1,    # 1 layer - QUÁ ÍT!
    mlp_ratio=2.0,   # MLP dim = 512
    spatial_attention='moderate'  # 512 tokens/view
)
```

**Vấn đề:**
- **1 layer quá ít** → không đủ capacity học cross-view relationships
- **4 heads ít** → attention patterns hạn chế
- **mlp_ratio=2.0 thấp** → bottleneck trong feedforward

**FIX:**
```python
mvvit=dict(
    embed_dim=256,
    num_heads=8,      # 4 → 8 heads (more attention patterns)
    num_layers=2,     # 1 → 2 layers (deeper learning)
    mlp_ratio=4.0,    # 2.0 → 4.0 (standard transformer ratio)
    dropout=0.1,
    spatial_attention='moderate'
)

# Note: Tăng capacity có thể cần thêm regularization
# Add dropout, weight decay
```

---

### **VẤN ĐỀ 6: Data Augmentation Quá Mạnh?**

**Supervised branch có accuracy thấp** → có thể augmentation làm khó data quá mức

**Check config:**
```python
# Strong augmentation for student
dict(type='RandAugment', ...)
dict(type='RandomErasing', ...)
```

**FIX:**
```python
# Giảm augmentation trong supervised branch
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),  # Chỉ flip
    # Bỏ RandAugment, RandomErasing
    dict(type='PackDetInputs')
]
```

---

## 🛠️ HÀNH ĐỘNG KHẮC PHỤC (ƯU TIÊN)

### **Priority 1: FIX NGAY** (Critical)

#### 1.1. **Clip Bounding Boxes**
```python
# File: mmdet/datasets/wrappers/multi_view_from_folder.py
# Thêm vào load_data_list()

def _clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = np.clip(x1, 0, width)
    x2 = np.clip(x2, 0, width)
    y1 = np.clip(y1, 0, height)
    y2 = np.clip(y2, 0, height)
    # Remove invalid boxes
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

# Apply to all GT boxes
gt_bboxes_filtered = []
for bbox in gt_bboxes:
    clipped = _clip_bbox(bbox, img_width, img_height)
    if clipped is not None:
        gt_bboxes_filtered.append(clipped)
```

#### 1.2. **Giảm Score Threshold**
```python
# In config
test_cfg=dict(
    rcnn=dict(
        score_thr=0.001,  # ← Từ 0.05 xuống
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=100
    )
)
```

#### 1.3. **Bật Aggregation trong Evaluation**
```python
val_evaluator = dict(
    type='MultiViewCocoMetric',
    views_per_sample=8,
    aggregate_predictions=True,  # ← BẬT
    aggregation_method='wbf',
    # ...
)
```

### **Priority 2: EXPERIMENT** (Test individually)

#### 2.1. **Test WITHOUT MVViT**
```python
detector.backbone = dict(
    type='MultiViewBackbone',
    backbone=dict(type='ResNet', depth=50, ...),
    fusion='mean',  # ← Tắt MVViT
    views_per_sample=8
)
```
→ Train 1000 iterations, check mAP  
→ Nếu mAP > 0 → MVViT là vấn đề  
→ Nếu mAP = 0 → Vấn đề ở chỗ khác

#### 2.2. **Tăng Learning Rate**
```python
optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# Từ 0.00002 → 0.01 (x500)
```

#### 2.3. **Giảm Augmentation**
```python
# Supervised pipeline: chỉ flip
# Unsupervised student: chỉ ColorJitter + flip
```

#### 2.4. **Fix Teacher Temperature**
```python
# In soft_teacher.py, get_pseudo_instances()
cls_scores = (cls_logits / 2.0).sigmoid()  # Temperature = 2.0
```

### **Priority 3: DEEP DIVE** (Nếu vẫn mAP=0)

#### 3.1. **Visualize Predictions**
```python
python tools/analysis_tools/visualize_predictions.py \
  --config configs/soft_teacher/soft_teacher_custom_multi_view.py \
  --checkpoint work_dirs/.../latest.pth \
  --img-dir data_drill/valid \
  --output vis_pred
```

#### 3.2. **Check Annotation Correctness**
```python
python tools/analysis_tools/browse_dataset.py \
  configs/soft_teacher/soft_teacher_custom_multi_view.py \
  --phase val \
  --output vis_gt
```

#### 3.3. **Profile Forward Pass**
```python
# Check if MVViT nans/infs
with torch.autograd.detect_anomaly():
    loss = model(batch)
```

---

## 📋 CHECKLIST DEBUG

```
□ Bbox coordinates clipped to image boundaries
□ Score threshold lowered (0.05 → 0.001)
□ Evaluation aggregation enabled
□ Teacher confidence not = 1.0
□ Learning rate reasonable (>0.0001)
□ Visualized GT annotations (correct?)
□ Visualized predictions (exist?)
□ Test without MVViT (isolate issue)
□ Check for NaN/Inf in loss
□ Verify annotation file correctness
```

---

## 🎯 KẾT LUẬN

**Vấn đề chính:**
1. **Bounding boxes vượt khỏi image boundaries** → cần clip
2. **Teacher predictions = 1.0** → cần temperature scaling
3. **Evaluation không aggregate** → cần bật WBF
4. **Score threshold quá cao** → cần giảm
5. **MVViT có thể làm nhiễu** → cần test without

**Khả năng cao nhất:**
- **70%**: Bbox coordinates sai + score threshold cao
- **20%**: MVViT làm nhiễu features
- **10%**: Learning rate quá thấp

**Next Steps:**
1. Fix bbox clipping (5 phút)
2. Giảm score threshold (1 phút)
3. Bật aggregation (1 phút)
4. Train thêm 1000 iterations
5. Check mAP → nếu vẫn 0 → test without MVViT

---

**Generated:** 2025-12-03  
**Author:** GitHub Copilot Analysis  
**Status:** CRITICAL - Cần fix ngay để model có thể detect được objects
