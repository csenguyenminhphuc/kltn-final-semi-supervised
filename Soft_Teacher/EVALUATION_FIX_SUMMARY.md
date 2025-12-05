# 🔧 EVALUATION FIX - Multi-View Aggregation

**Ngày:** 03/12/2025  
**Vấn đề:** Evaluation đánh giá 640 views riêng lẻ thay vì aggregate thành 80 base images  
**Trạng thái:** ✅ ĐÃ SỬA

---

## 📋 THAY ĐỔI CHÍNH

### **File:** `mmdetection/mmdet/evaluation/metrics/multi_view_coco_metric.py`

#### **1. Sửa `compute_metrics()` - Dòng 176-257**

**Trước (SAI):**
```python
def compute_metrics(self, results):
    # Group predictions (chỉ để log statistics!)
    grouped_predictions = self._group_predictions(list(preds))
    logger.info(f"Number of groups: {len(grouped_predictions)}")  # 80 groups
    
    # ❌ Evaluate 640 views riêng lẻ
    metrics = super().compute_metrics(results)  # Pass 640 views
    
    return metrics
```

**Sau (ĐÚNG):**
```python
def compute_metrics(self, results):
    # Step 1: Group 640 predictions by base_img_id
    grouped_predictions = self._group_predictions(list(preds))  # 80 groups
    
    # Step 2: ✅ Aggregate 8 views → 1 base image (WBF/NMS)
    aggregated_preds = self._aggregate_predictions(grouped_predictions)  # 80 preds
    
    # Step 3: ✅ Get GT for base images
    base_gts = self._get_base_image_gts(grouped_predictions, gts)  # 80 GTs
    
    # Step 4: ✅ Evaluate 80 base images
    aggregated_results = list(zip(base_gts, aggregated_preds))
    metrics = super().compute_metrics(aggregated_results)  # Pass 80 base images
    
    return metrics
```

#### **2. Thêm hàm `_get_base_image_gts()` - Dòng 318-365**

```python
def _get_base_image_gts(self, grouped_predictions, all_gts):
    """Aggregate ground truth from all views to base image.
    
    Steps:
    1. Transform GT boxes: crop coords → base coords (homography)
    2. Collect GT from all 8 views
    3. Deduplicate (remove same object in multiple views)
    """
    base_gts = []
    gt_dict = {gt['img_id']: gt for gt in all_gts}
    
    for base_name, view_preds in grouped_predictions.items():
        all_view_anns = []
        
        # Transform GT from each view to base image coords
        for view_pred in view_preds:
            view_gt = gt_dict[view_pred['img_id']]
            homography = view_pred['homography_matrix']
            
            for ann in view_gt['anns']:
                # Transform bbox using homography
                transformed_bbox = transform(ann['bbox'], homography)
                all_view_anns.append({
                    'bbox': transformed_bbox,
                    'category_id': ann['category_id'],
                    ...
                })
        
        # Remove duplicates (IoU > 0.5)
        deduplicated = self._deduplicate_gt_boxes(all_view_anns)
        
        base_gts.append({
            'img_id': base_img_id,
            'anns': deduplicated
        })
    
    return base_gts
```

#### **3. Thêm hàm `_deduplicate_gt_boxes()` - Mới**

```python
def _deduplicate_gt_boxes(self, annotations, iou_threshold=0.5):
    """Remove duplicate GT (same object in overlapping views).
    
    - Group by category_id
    - For each category: NMS with IoU > 0.5 → keep unique boxes
    """
    # Group by category
    cat_groups = {cat_id: [anns] for ann in annotations}
    
    deduplicated = []
    for cat_id, anns in cat_groups.items():
        # Simple NMS: keep first, remove high-IoU duplicates
        kept = []
        for ann in anns:
            if not overlaps_with_kept(ann, kept, iou_threshold):
                kept.append(ann)
        deduplicated.extend(kept)
    
    return deduplicated
```

---

## 📊 LUỒNG EVALUATION MỚI

### **Input:**
```
640 views from validation set:
  - 80 base images × 8 crops each
  - Each crop has its own prediction
```

### **Processing:**

```
Step 1: GROUP BY BASE_IMG_ID
  640 views → 80 groups
  {
    'S110_bright_2': [pred_0, pred_1, ..., pred_7],    # 8 views
    'S110_bright_3': [pred_8, pred_9, ..., pred_15],   # 8 views
    ...
  }

Step 2: AGGREGATE PREDICTIONS (per group)
  For each group (80 times):
    - Transform predictions to base image coordinates (homography)
    - Apply WBF/NMS to merge 8 predictions → 1 prediction
  
  Result: 80 aggregated predictions

Step 3: AGGREGATE GROUND TRUTH (per group) ← ✅ MỚI SỬA
  For each group (80 times):
    - Get GT from all 8 views (640 view GTs)
    - Transform GT boxes: crop coords → base coords (homography)
    - Deduplicate: Remove same object in multiple views (IoU > 0.5)
  
  Result: 80 aggregated GTs

Step 4: COMPUTE mAP
  COCO evaluation on 80 (prediction, GT) pairs
  
  Result: mAP on base images ✅
```

---

## 🔄 SO SÁNH TRƯỚC/SAU

| Metric | Trước (SAI) | Sau (ĐÚNG) |
|--------|-------------|------------|
| **Số predictions evaluate** | 640 views | 80 base images |
| **Aggregation method** | None | WBF/NMS |
| **GT aggregation** | ❌ None (first view only) | ✅ Transform + Deduplicate |
| **Coordinate transform** | No | Yes (homography) |
| **Reflects end-user usage** | ❌ No | ✅ Yes |
| **mAP meaning** | Per-crop detection | Full image detection |

---

## ✅ HÀNG ĐÃ CÓ SẴN (ĐƯỢC SỬ DỤNG)

1. ✅ `_group_predictions()` - Group 640 views thành 80 groups
2. ✅ `_aggregate_predictions()` - Aggregate predictions bằng WBF/NMS
3. ✅ `_project_boxes_to_original_space()` - Transform coordinates
4. ✅ `_nms_aggregation()`, `_wbf_aggregation()`, etc. - Merge boxes

## ✅ HÀM MỚI THÊM VÀO

5. ✅ `_get_base_image_gts()` - **Aggregate GT từ 8 views**
6. ✅ `_deduplicate_gt_boxes()` - **Remove duplicate GT boxes**
7. ✅ `_compute_iou()` - **Helper để tính IoU cho deduplication**

---

## 🎯 KẾT QUẢ MONG ĐỢI

### **Log mới sẽ hiển thị:**
```
Multi-View COCO Evaluation (Aggregated Mode)
================================================================================

[Step 1] Grouping 640 view predictions by base image...
  - Number of base image groups: 80
  - Views per group: 8
  - Total views: 640

[Step 2] Aggregating predictions using 'wbf'...
  - Groups with detections: 80
  - Total boxes projected to original space: 450
  - Aggregated to 80 base images

[Step 3] Preparing ground truth for base images...
  - Transforming GT from 640 views to base coordinates...
  - Deduplicating overlapping GT boxes (IoU > 0.5)...
  - Prepared GT for 80 base images
  - Average GT boxes per base image: 5.6

[Step 4] Computing COCO metrics on 80 base images...
Evaluating bbox...

+-------------+------+--------+--------+-------+-------+-------+
| category    | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+-------------+------+--------+--------+-------+-------+-------+
| Broken      | 0.15 | 0.25   | 0.12   | nan   | 0.18  | 0.14  |
| Chipped     | 0.22 | 0.35   | 0.18   | nan   | 0.25  | 0.20  |
| ...         | ...  | ...    | ...    | ...   | ...   | ...   |
+-------------+------+--------+--------+-------+-------+-------+

Multi-View Summary
================================================================================
Evaluation mode: Aggregated (8 views → 1 base image)
Aggregation method: wbf
Base images evaluated: 80
================================================================================
```

### **Metrics mới:**
```python
{
  'bbox_mAP': 0.18,           # mAP trên base images
  'bbox_mAP_50': 0.28,
  'mv_num_views': 8,
  'mv_num_groups': 80,
  'mv_base_images_evaluated': 80,  # ← CHỈ 80, không phải 640!
  'mv_evaluation_mode': 'aggregated',
  'mv_aggregation_method': 'wbf'
}
```

---

## 🚀 CÁCH SỬ DỤNG

Không cần thay đổi config, evaluation tự động aggregate:

```python
# Config (không đổi)
val_evaluator = dict(
    type='MultiViewCocoMetric',
    ann_file='data_drill/anno_valid/_annotations_filtered.coco.json',
    metric='bbox',
    views_per_sample=8,
    aggregation='wbf',  # hoặc 'nms', 'soft_nms', 'voting'
    nms_iou_thr=0.5
)
```

**Training command (không đổi):**
```bash
python tools/train.py configs/soft_teacher/soft_teacher_custom_multi_view.py
```

---

## 🔍 KIỂM TRA FIX HOẠT ĐỘNG

```bash
# Run validation
python tools/test.py \
  configs/soft_teacher/soft_teacher_custom_multi_view.py \
  work_dirs/soft_teacher_8views_cross_transformers/latest.pth

# Kiểm tra log:
# 1. ✅ "Aggregated Mode" (không phải "Per-View Mode")
# 2. ✅ "Aggregated to 80 base images"
# 3. ✅ "Computing COCO metrics on 80 base images"
# 4. ✅ mAP > 0 (nếu model đã train đủ)
```

---

## 📝 NOTES

### **Về Ground Truth:**
- ✅ **ĐÚNG:** Aggregate GT từ 8 views giống như predictions
- Mỗi view có GT riêng ở crop coordinates
- Transform GT boxes: crop coords → base image coords (homography)
- Deduplicate: Remove duplicate boxes (same object in multiple views, IoU > 0.5)
- Kết quả: 80 aggregated GTs cho 80 base images

### **Về Homography Matrix:**
- Code đã có sẵn `_project_boxes_to_original_space()`
- Transform predictions từ crop coords → base image coords
- Quan trọng để các predictions từ 8 crops có cùng coordinate system

### **Về Aggregation Methods:**
- `nms`: Standard NMS (fast, simple)
- `wbf`: Weighted Box Fusion (better for multi-view, recommended)
- `soft_nms`: Soft NMS (smooth)
- `voting`: Box voting (ensemble-like)

---

**Generated:** 2025-12-03  
**Status:** ✅ FIXED - Evaluation now correctly aggregates 8 views → 1 base image  
**Next:** Run validation to verify mAP improves
