# Kiến Trúc Multi-View Soft Teacher

## Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-VIEW SOFT TEACHER ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    INPUT STAGE                                       │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  MultiViewFromFolder Dataset                                                 │  │
│  │  • Nhóm 8 crops từ 1 base image thành 1 sample                              │  │
│  │  • Labeled data: 298 groups × 8 = 2384 images (60%)                        │  │
│  │  • Unlabeled data: 199 groups × 8 = 1592 images (40%)                      │  │
│  │                                                                              │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐       ┌───────────┐          │  │
│  │  │ Crop 0    │  │ Crop 1    │  │ Crop 2    │  ...  │ Crop 7    │          │  │
│  │  │ (256×720) │  │ (256×720) │  │ (256×720) │       │ (256×720) │          │  │
│  │  └───────────┘  └───────────┘  └───────────┘       └───────────┘          │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                         ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  multi_view_collate_flatten                                                  │  │
│  │  • Flatten: (B groups, 8 views, C, H, W) → (B×8, C, H, W)                  │  │
│  │  • Batch_size=4 → 32 images per batch                                      │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       TEACHER-STUDENT FRAMEWORK (Semi-Supervised)                    │
│                                                                                      │
│  ┌──────────────────────────────┐      ┌──────────────────────────────┐            │
│  │     TEACHER BRANCH           │      │     STUDENT BRANCH           │            │
│  │   (EMA, frozen=True)         │      │   (Trained by gradient)      │            │
│  │                              │      │                              │            │
│  │  Weak augmentation:          │      │  Strong augmentation:        │            │
│  │  • Resize + Flip             │      │  • Resize + Flip             │            │
│  │  • No color jitter           │      │  • RandAugment (color)       │            │
│  │                              │      │  • RandAugment (geometric)   │            │
│  │                              │      │  • RandomErasing             │            │
│  └──────────────────────────────┘      └──────────────────────────────┘            │
│              ↓                                      ↓                               │
│  ┌──────────────────────────────┐      ┌──────────────────────────────┐            │
│  │   Pseudo Label Generation    │      │   Learning from Labels       │            │
│  │   • Score threshold: 0.7     │ ───→ │   • Supervised loss (GT)     │            │
│  │   • Max 15 boxes/image       │      │   • Unsupervised loss (PL)   │            │
│  │   • Uncertainty filtering    │      │   • Weight: 1.0 vs 0.5       │            │
│  └──────────────────────────────┘      └──────────────────────────────┘            │
│              ↓                                      ↓                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                          SHARED ARCHITECTURE                                 │  │
│  │                       (Both Teacher & Student use same)                      │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            MULTI-VIEW BACKBONE STAGE                                 │
│                                                                                      │
│  Input: (B×8, 3, H, W) - Flattened batch                                           │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  MultiViewBackbone Wrapper                                                   │  │
│  │  • Reshapes: (B×8, C, H, W) → (B, 8, C, H, W) internally                   │  │
│  │  • views_per_sample = 8                                                     │  │
│  │  • fusion = 'mvvit'                                                         │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                         ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  ResNet-50 Backbone (Caffe pretrained)                                      │  │
│  │  • Frozen Stage 1 (conv1, bn1)                                              │  │
│  │  • BatchNorm frozen (eval mode)                                             │  │
│  │                                                                              │  │
│  │  Input: (B×8, 3, 256, 720) - Process all views as independent samples      │  │
│  │         ↓                                                                    │  │
│  │  ┌─────────────┐                                                            │  │
│  │  │   Conv1     │  (B×8, 64, 128, 360)                                       │  │
│  │  │   Stage 1   │                                                            │  │
│  │  └─────────────┘                                                            │  │
│  │         ↓                                                                    │  │
│  │  ┌─────────────┐                                                            │  │
│  │  │   Stage 2   │  (B×8, 256, 64, 180)   → P2                                │  │
│  │  └─────────────┘                                                            │  │
│  │         ↓                                                                    │  │
│  │  ┌─────────────┐                                                            │  │
│  │  │   Stage 3   │  (B×8, 512, 32, 90)    → P3                                │  │
│  │  └─────────────┘                                                            │  │
│  │         ↓                                                                    │  │
│  │  ┌─────────────┐                                                            │  │
│  │  │   Stage 4   │  (B×8, 1024, 16, 45)   → P4                                │  │
│  │  └─────────────┘                                                            │  │
│  │         ↓                                                                    │  │
│  │  ┌─────────────┐                                                            │  │
│  │  │   Stage 5   │  (B×8, 2048, 8, 22)    → P5                                │  │
│  │  └─────────────┘                                                            │  │
│  │                                                                              │  │
│  │  Output: 4 feature levels [P2, P3, P4, P5]                                 │  │
│  │  • Each level: (B×8, C_level, H_level, W_level)                            │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-VIEW TRANSFORMER FUSION (MVViT)                             │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  MVViT - Cross-View Attention Module                                        │  │
│  │  • embed_dim: 256                                                           │  │
│  │  • num_heads: 4                                                             │  │
│  │  • num_layers: 1                                                            │  │
│  │  • mlp_ratio: 2.0                                                           │  │
│  │  • dropout: 0.1                                                             │  │
│  │  • spatial_attention: 'moderate' (512 tokens/view)                          │  │
│  │  • use_gradient_checkpointing: True                                         │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  Process each FPN level independently:                                               │
│                                                                                      │
│  For each level (P2, P3, P4, P5):                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 1: Reshape & Project                                                  │  │
│  │  (B×8, C_level, H, W) → (B, 8, C_level, H, W) → (B, 8, 256, H, W)         │  │
│  │  • 1×1 Conv projection if C_level ≠ 256                                    │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                         ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 2: Attention Pooling (Spatial Reduction)                              │  │
│  │  (B, 8, 256, H, W) → (B×8, H×W, 256)                                       │  │
│  │                  ↓                                                          │  │
│  │  • K = min(512, H×W) learnable pooling queries                             │  │
│  │  • Attention: Q=queries(K,256), KV=spatial_tokens(H×W,256)                 │  │
│  │  • Output: (B×8, K, 256) - Pooled spatial features per view                │  │
│  │                                                                              │  │
│  │  Memory saved: O(H×W) → O(512) per view                                    │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                         ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 3: Cross-View Transformer                                             │  │
│  │  (B×8, K, 256) → (B, 8×K, 256)                                             │  │
│  │                  ↓                                                          │  │
│  │  ┌────────────────────────────────────────────────┐                        │  │
│  │  │  Transformer Encoder Layer × 1                 │                        │  │
│  │  │  ┌──────────────────────────────────────────┐  │                        │  │
│  │  │  │  Multi-Head Self-Attention (4 heads)     │  │                        │  │
│  │  │  │  • Views attend to each other            │  │                        │  │
│  │  │  │  • Q,K,V: (B, 8×K, 256)                  │  │                        │  │
│  │  │  │  • Attention weights: (B, 4, 8×K, 8×K)   │  │                        │  │
│  │  │  │  • Learns cross-view relationships       │  │                        │  │
│  │  │  └──────────────────────────────────────────┘  │                        │  │
│  │  │                  ↓                              │                        │  │
│  │  │  ┌──────────────────────────────────────────┐  │                        │  │
│  │  │  │  Feedforward Network (MLP)               │  │                        │  │
│  │  │  │  • Linear: 256 → 512 → 256               │  │                        │  │
│  │  │  │  • GELU activation                        │  │                        │  │
│  │  │  │  • Dropout: 0.1                           │  │                        │  │
│  │  │  └──────────────────────────────────────────┘  │                        │  │
│  │  │                  ↓                              │                        │  │
│  │  │  • Residual connections + LayerNorm           │                        │  │
│  │  │  • Gradient checkpointing enabled             │                        │  │
│  │  └────────────────────────────────────────────────┘                        │  │
│  │                                                                              │  │
│  │  Output: (B, 8×K, 256) - Refined cross-view features                       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                         ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 4: Attention Upsampling (Spatial Reconstruction)                      │  │
│  │  (B, 8×K, 256) → (B×8, K, 256)                                             │  │
│  │                  ↓                                                          │  │
│  │  • H×W learnable upsample queries                                           │  │
│  │  • Attention: Q=queries(H×W,256), KV=pooled_tokens(K,256)                  │  │
│  │  • Output: (B×8, H×W, 256) → (B, 8, 256, H, W)                            │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                         ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 5: Residual & Project Back                                            │  │
│  │  • Residual connection: refined = 0.5×refined + 0.5×original                │  │
│  │  • Project back: 256 → C_level (if needed)                                  │  │
│  │  • Output: (B×8, C_level, H, W) - Refined per-view features                │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  Final Output: 4 refined FPN levels [P2', P3', P4', P5']                           │
│  • Each level: (B×8, C_level, H_level, W_level)                                    │
│  • Features now contain cross-view context from all 8 crops                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION FRAMEWORK (Semi-Supervised)                      │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Main Loss Function: SemiBaseDetector.loss()                                 │  │
│  │  Combines supervised and unsupervised losses                                 │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌─────────────────────────────────┐      ┌─────────────────────────────────┐      │
│  │   SUPERVISED BRANCH             │      │   UNSUPERVISED BRANCH           │      │
│  │   (Labeled data: 60%)           │      │   (Unlabeled data: 40%)         │      │
│  ├─────────────────────────────────┤      ├─────────────────────────────────┤      │
│  │                                 │      │                                 │      │
│  │  loss_by_gt_instances()         │      │  loss_by_pseudo_instances()     │      │
│  │  • Ground-truth labels          │      │  • Pseudo labels from teacher   │      │
│  │  • Standard detection losses    │      │  • Filtered by confidence       │      │
│  │  • Weight: sup_weight = 1.0     │      │  • Weight: unsup_weight = 0.5   │      │
│  │                                 │      │                                 │      │
│  └─────────────────────────────────┘      └─────────────────────────────────┘      │
│              ↓                                      ↓                               │
│  ┌─────────────────────────────────┐      ┌─────────────────────────────────┐      │
│  │   Student.loss()                │      │   Pseudo Label Generation       │      │
│  │   • Input: (B×8, C, H, W)       │      │   ┌─────────────────────────┐   │      │
│  │   • Multi-view features         │      │   │ 1. Teacher Forward      │   │      │
│  │   • Cross-view context          │      │   │    (weak augmentation)  │   │      │
│  │                                 │      │   │ 2. RPN predict          │   │      │
│  │   ↓                             │      │   │ 3. ROI Head predict     │   │      │
│  │   Compute 3 types of losses:    │      │   │ 4. Score filtering      │   │      │
│  │   1. RPN Loss                   │      │   │    threshold = 0.7      │   │      │
│  │   2. RCNN Classification Loss   │      │   │ 5. Max 15 boxes/image   │   │      │
│  │   3. RCNN Regression Loss       │      │   │ 6. Uncertainty filter   │   │      │
│  └─────────────────────────────────┘      │   │    (jitter_times = 5)   │   │      │
│                                            │   └─────────────────────────┘   │      │
│                                            │              ↓                  │      │
│                                            │   Student learns from:          │      │
│                                            │   • High-conf pseudo labels     │      │
│                                            │   • Strong augmentation         │      │
│                                            └─────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         DETAILED LOSS COMPONENTS                                     │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  1. RPN LOSS (Region Proposal Network)                                       │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │                                                                              │  │
│  │  A. RPN Classification Loss (Objectness)                                    │  │
│  │     ┌────────────────────────────────────────────────────────────────────┐ │  │
│  │     │  Type: Focal Loss (combat class imbalance)                         │ │  │
│  │     │  Formula: FL(p) = -α(1-p)^γ log(p)                                 │ │  │
│  │     │                                                                      │ │  │
│  │     │  Parameters:                                                        │ │  │
│  │     │  • gamma = 2.0 (focus factor - down-weight easy examples)          │ │  │
│  │     │  • alpha = 0.5 (class weight - balance FG/BG)                      │ │  │
│  │     │  • loss_weight = 3.0 (emphasize classification vs regression)      │ │  │
│  │     │                                                                      │ │  │
│  │     │  Training Config (Anchor Assignment):                               │ │  │
│  │     │  • Positive IoU threshold: 0.5 (relaxed for narrow objects)        │ │  │
│  │     │  • Negative IoU threshold: 0.3                                      │ │  │
│  │     │  • num_samples: 256 per image (reduced from 512)                   │ │  │
│  │     │  • pos_fraction: 0.8 (80% positive - handle sparse data!)          │ │  │
│  │     │  • neg_pos_ub: 1 (max 1:1 neg:pos ratio - not 3:1!)               │ │  │
│  │     │                                                                      │ │  │
│  │     │  Why Focal Loss?                                                    │ │  │
│  │     │  • Background anchors dominate (thousands vs few objects)          │ │  │
│  │     │  • Easy negatives overwhelm loss signal                            │ │  │
│  │     │  • Focal Loss: (1-p)^2 reduces loss from easy examples by 100x     │ │  │
│  │     │  • Forces model to focus on hard-to-classify anchors               │ │  │
│  │     └────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                              │  │
│  │  B. RPN Regression Loss (Anchor → Proposal refinement)                      │  │
│  │     ┌────────────────────────────────────────────────────────────────────┐ │  │
│  │     │  Type: L1 Loss (simple and effective)                              │ │  │
│  │     │  Formula: L1(pred, target) = |pred - target|                       │ │  │
│  │     │                                                                      │ │  │
│  │     │  Parameters:                                                        │ │  │
│  │     │  • loss_weight = 1.0                                                │ │  │
│  │     │                                                                      │ │  │
│  │     │  Target Encoding (Delta XYWHBBoxCoder):                            │ │  │
│  │     │  • dx = (x_gt - x_anchor) / w_anchor                               │ │  │
│  │     │  • dy = (y_gt - y_anchor) / h_anchor                               │ │  │
│  │     │  • dw = log(w_gt / w_anchor)                                       │ │  │
│  │     │  • dh = log(h_gt / h_anchor)                                       │ │  │
│  │     │  • Target stds: [1.0, 1.0, 1.0, 1.0] (no normalization)           │ │  │
│  │     │                                                                      │ │  │
│  │     │  Only computed for positive anchors (IoU > 0.5)                    │ │  │
│  │     └────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                              │  │
│  │  RPN Output per image:                                                       │  │
│  │  • loss_rpn_cls: ~0.1-0.5 (supervised), ~0.05-0.2 (unsupervised)           │  │
│  │  • loss_rpn_bbox: ~0.05-0.15                                                │  │
│  │  • ~1000 proposals per image (after NMS)                                    │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  2. RCNN CLASSIFICATION LOSS (Final class prediction)                       │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │                                                                              │  │
│  │  A. Supervised Loss (Ground-truth labels)                                   │  │
│  │     ┌────────────────────────────────────────────────────────────────────┐ │  │
│  │     │  Type: Focal Loss                                                   │ │  │
│  │     │  Formula: FL(p) = -α(1-p)^γ log(p)                                 │ │  │
│  │     │                                                                      │ │  │
│  │     │  Parameters:                                                        │ │  │
│  │     │  • gamma = 2.0 (focus on hard examples)                            │ │  │
│  │     │  • alpha = 0.5 (increased from 0.25 for sparse data)               │ │  │
│  │     │  • loss_weight = 2.0 (increased from 1.0)                          │ │  │
│  │     │  • use_sigmoid = True (required by mmdet)                          │ │  │
│  │     │                                                                      │ │  │
│  │     │  Training Config (RoI Assignment):                                  │ │  │
│  │     │  • Positive IoU threshold: 0.5                                      │ │  │
│  │     │  • num_samples: 256 per image                                       │ │  │
│  │     │  • pos_fraction: 0.75 (75% positive samples)                       │ │  │
│  │     │  • neg_pos_ub: 1 (max 1:1 ratio)                                   │ │  │
│  │     │                                                                      │ │  │
│  │     │  Classes: 5 defect types + 1 background                            │ │  │
│  │     │  • 0: Broken                                                        │ │  │
│  │     │  • 1: Chipped                                                       │ │  │
│  │     │  • 2: Scratched                                                     │ │  │
│  │     │  • 3: Severe_Rust                                                   │ │  │
│  │     │  • 4: Tip_Wear                                                      │ │  │
│  │     │  • 5: Background (implicit)                                         │ │  │
│  │     └────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                              │  │
│  │  B. Unsupervised Loss (Pseudo labels from teacher)                          │  │
│  │     ┌────────────────────────────────────────────────────────────────────┐ │  │
│  │     │  Special Handling: Soft Teacher's key innovation                   │ │  │
│  │     │                                                                      │ │  │
│  │     │  1. Pseudo Label Filtering:                                         │ │  │
│  │     │     • Initial threshold: 0.7 (high confidence only)                │ │  │
│  │     │     • Max 15 boxes per image (prevent overfit)                     │ │  │
│  │     │     • Per-class threshold: cls_pseudo_thr = 0.6                    │ │  │
│  │     │                                                                      │ │  │
│  │     │  2. Background Score Reweighting:                                   │ │  │
│  │     │     • Teacher predicts background score for negative samples       │ │  │
│  │     │     • Replace hard 0/1 labels with soft scores                     │ │  │
│  │     │     • Formula: label_weight[neg] = teacher_bg_score[neg]           │ │  │
│  │     │     • Prevents overconfident predictions on ambiguous regions      │ │  │
│  │     │                                                                      │ │  │
│  │     │  3. Loss Normalization:                                             │ │  │
│  │     │     • Scale = num_samples / sum(label_weights)                     │ │  │
│  │     │     • loss_cls = loss_cls * scale                                  │ │  │
│  │     │     • Ensures consistent gradient magnitude                        │ │  │
│  │     │                                                                      │ │  │
│  │     │  Why this works?                                                    │ │  │
│  │     │  • Avoids confirmation bias (student copying teacher errors)       │ │  │
│  │     │  • Soft labels provide richer supervision signal                   │ │  │
│  │     │  • High threshold prevents noisy pseudo-labels                     │ │  │
│  │     └────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                              │  │
│  │  Output:                                                                     │  │
│  │  • loss_cls: ~0.2-0.8 (supervised), ~0.1-0.4 (unsupervised)                │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  3. RCNN REGRESSION LOSS (Bbox refinement)                                  │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │                                                                              │  │
│  │  A. Supervised Loss (Ground-truth boxes)                                    │  │
│  │     ┌────────────────────────────────────────────────────────────────────┐ │  │
│  │     │  Type: Smooth L1 Loss (robust to outliers)                         │ │  │
│  │     │  Formula:                                                           │ │  │
│  │     │    SmoothL1(x) = 0.5 * x^2           if |x| < 1                   │ │  │
│  │     │                  |x| - 0.5            otherwise                     │ │  │
│  │     │                                                                      │ │  │
│  │     │  Parameters:                                                        │ │  │
│  │     │  • beta = 1.0 (transition point)                                   │ │  │
│  │     │  • loss_weight = 1.0                                                │ │  │
│  │     │                                                                      │ │  │
│  │     │  Target Encoding: Same as RPN (DeltaXYWHBBoxCoder)                │ │  │
│  │     │  Only computed for positive RoIs (foreground objects)              │ │  │
│  │     └────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                              │  │
│  │  B. Unsupervised Loss (Pseudo labels with uncertainty filtering)            │  │
│  │     ┌────────────────────────────────────────────────────────────────────┐ │  │
│  │     │  Uncertainty-based Filtering (Soft Teacher's innovation):          │ │  │
│  │     │                                                                      │ │  │
│  │     │  1. Box Jittering for Uncertainty Estimation:                      │ │  │
│  │     │     • jitter_times = 5 (augment each box 5 times)                  │ │  │
│  │     │     • jitter_scale = 0.03 (3% of box size)                         │ │  │
│  │     │     • offset = random_normal(0, 0.03 * box_scale)                  │ │  │
│  │     │                                                                      │ │  │
│  │     │  2. Teacher Prediction on Jittered Boxes:                          │ │  │
│  │     │     • Feed 5 jittered versions to teacher                          │ │  │
│  │     │     • Get 5 predictions per box                                     │ │  │
│  │     │                                                                      │ │  │
│  │     │  3. Uncertainty Computation:                                        │ │  │
│  │     │     • Mean bbox: μ = mean(pred_1, ..., pred_5)                    │ │  │
│  │     │     • Std deviation: σ = std(pred_1, ..., pred_5)                 │ │  │
│  │     │     • Box size: wh = [w, h] of predicted box                       │ │  │
│  │     │     • Normalized uncertainty: reg_unc = mean(σ / wh)              │ │  │
│  │     │                                                                      │ │  │
│  │     │  4. Filtering:                                                      │ │  │
│  │     │     • Keep only boxes with reg_unc < reg_pseudo_thr (0.02)        │ │  │
│  │     │     • Removes unreliable pseudo-labels                             │ │  │
│  │     │                                                                      │ │  │
│  │     │  Why this works?                                                    │ │  │
│  │     │  • High uncertainty = teacher unsure about exact location          │ │  │
│  │     │  • Filtering prevents student from learning noisy boxes            │ │  │
│  │     │  • More reliable than simple confidence threshold                  │ │  │
│  │     └────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                              │  │
│  │  Output:                                                                     │  │
│  │  • loss_bbox: ~0.05-0.2 (supervised), ~0.02-0.1 (unsupervised)             │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  FINAL LOSS AGGREGATION                                                      │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │                                                                              │  │
│  │  Supervised Losses (weight = 1.0):                                          │  │
│  │  • sup_loss_rpn_cls     = weight × loss_rpn_cls                             │  │
│  │  • sup_loss_rpn_bbox    = weight × loss_rpn_bbox                            │  │
│  │  • sup_loss_cls         = weight × loss_cls                                 │  │
│  │  • sup_loss_bbox        = weight × loss_bbox                                │  │
│  │                                                                              │  │
│  │  Unsupervised Losses (weight = 0.5):                                        │  │
│  │  • unsup_loss_rpn_cls   = weight × loss_rpn_cls                             │  │
│  │  • unsup_loss_rpn_bbox  = weight × loss_rpn_bbox                            │  │
│  │  • unsup_loss_cls       = weight × loss_cls                                 │  │
│  │  • unsup_loss_bbox      = weight × loss_bbox                                │  │
│  │                                                                              │  │
│  │  Total Loss:                                                                 │  │
│  │  loss = Σ(sup_losses) + Σ(unsup_losses)                                     │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │  │
│  │  │  MULTI-VIEW LOSS COMPUTATION (Key Innovation)                       │   │  │
│  │  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │   │  │
│  │  │                                                                      │   │  │
│  │  │  Input Structure: 8 crops per base image                            │   │  │
│  │  │  ┌────────┬────────┬────────┬─────┬────────┐                       │   │  │
│  │  │  │ Crop 0 │ Crop 1 │ Crop 2 │ ... │ Crop 7 │  = 1 Group           │   │  │
│  │  │  └────────┴────────┴────────┴─────┴────────┘                       │   │  │
│  │  │  Batch Size = 4 groups → 32 images total                           │   │  │
│  │  │                                                                      │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │  Step 1: FEATURE EXTRACTION with Cross-View Attention              │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │                                                                      │   │  │
│  │  │  Input: (B×V, C, H, W) = (32, 3, 256, 720)                        │   │  │
│  │  │         ↓                                                           │   │  │
│  │  │  MultiViewBackbone:                                                 │   │  │
│  │  │  • Reshape internally: (32, 3, 256, 720) → (4, 8, 3, 256, 720)    │   │  │
│  │  │  • ResNet extracts per-crop features                               │   │  │
│  │  │  • MVViT applies cross-view attention at each FPN level            │   │  │
│  │  │    - Crop 0 attends to Crops 1-7 in same group                    │   │  │
│  │  │    - Learns: "What info from other views helps this crop?"        │   │  │
│  │  │  • Output: (32, 256, H', W') per FPN level                        │   │  │
│  │  │                                                                      │   │  │
│  │  │  KEY: Features now contain CROSS-VIEW CONTEXT                      │   │  │
│  │  │       Each crop's features aware of other 7 crops!                 │   │  │
│  │  │                                                                      │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │  Step 2: PER-CROP LOSS COMPUTATION (Independent)                   │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │                                                                      │   │  │
│  │  │  For each of 32 crops:                                              │   │  │
│  │  │  1. RPN predicts proposals on enriched features                    │   │  │
│  │  │  2. ROI Head predicts boxes on enriched features                   │   │  │
│  │  │  3. Compute loss vs GT/pseudo-labels of THIS crop only             │   │  │
│  │  │                                                                      │   │  │
│  │  │  Example for Crop 0 in Group 0:                                    │   │  │
│  │  │  ┌──────────────────────────────────────────────────────┐         │   │  │
│  │  │  │ Features: f_0 (contains info from Crops 0-7)         │         │   │  │
│  │  │  │ GT Labels: gt_0 (only for Crop 0's visible region)   │         │   │  │
│  │  │  │                                                        │         │   │  │
│  │  │  │ RPN Loss:                                              │         │   │  │
│  │  │  │   loss_rpn_cls_0  = FocalLoss(rpn_pred_0, gt_0)      │         │   │  │
│  │  │  │   loss_rpn_bbox_0 = L1Loss(rpn_bbox_0, gt_0)         │         │   │  │
│  │  │  │                                                        │         │   │  │
│  │  │  │ ROI Loss:                                              │         │   │  │
│  │  │  │   loss_cls_0  = FocalLoss(roi_cls_0, gt_0)           │         │   │  │
│  │  │  │   loss_bbox_0 = SmoothL1(roi_bbox_0, gt_0)           │         │   │  │
│  │  │  │                                                        │         │   │  │
│  │  │  │ Total for Crop 0:                                      │         │   │  │
│  │  │  │   loss_0 = loss_rpn_cls_0 + loss_rpn_bbox_0 +        │         │   │  │
│  │  │  │            loss_cls_0 + loss_bbox_0                   │         │   │  │
│  │  │  └──────────────────────────────────────────────────────┘         │   │  │
│  │  │                                                                      │   │  │
│  │  │  Repeat for all 32 crops → get 32 individual losses                │   │  │
│  │  │                                                                      │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │  Step 3: LOSS AGGREGATION (Standard PyTorch)                       │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │                                                                      │   │  │
│  │  │  PyTorch's loss functions automatically average over batch:        │   │  │
│  │  │                                                                      │   │  │
│  │  │  loss_rpn_cls  = mean(loss_rpn_cls_0, ..., loss_rpn_cls_31)       │   │  │
│  │  │  loss_rpn_bbox = mean(loss_rpn_bbox_0, ..., loss_rpn_bbox_31)     │   │  │
│  │  │  loss_cls      = mean(loss_cls_0, ..., loss_cls_31)               │   │  │
│  │  │  loss_bbox     = mean(loss_bbox_0, ..., loss_bbox_31)             │   │  │
│  │  │                                                                      │   │  │
│  │  │  This is equivalent to:                                             │   │  │
│  │  │  • Average over 8 crops within each group                          │   │  │
│  │  │  • Then average over 4 groups                                       │   │  │
│  │  │                                                                      │   │  │
│  │  │  Mathematically:                                                    │   │  │
│  │  │    loss = (1/32) × Σ(all 32 crops)                                │   │  │
│  │  │         = (1/4) × Σ_groups[(1/8) × Σ_crops_in_group]              │   │  │
│  │  │         = mean over groups of (mean over crops)                    │   │  │
│  │  │                                                                      │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │  Why This Approach Works for Multi-View Learning:                  │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │                                                                      │   │  │
│  │  │  ✓ Cross-view learning happens in FEATURES (MVViT attention)       │   │  │
│  │  │    • Before loss computation, features already contain              │   │  │
│  │  │      information from all 8 views                                   │   │  │
│  │  │    • Model learns: "Which other views help predict this crop?"     │   │  │
│  │  │                                                                      │   │  │
│  │  │  ✓ Per-crop losses respect independent ground truth                │   │  │
│  │  │    • Each crop has different visible region & GT annotations       │   │  │
│  │  │    • Loss for Crop 0 only uses GT for Crop 0                       │   │  │
│  │  │    • Prevents incorrect supervision from other crops' GTs          │   │  │
│  │  │                                                                      │   │  │
│  │  │  ✓ Gradient flow maintains group structure                         │   │  │
│  │  │    • Backprop through MVViT connects all crops in same group       │   │  │
│  │  │    • ∂Loss/∂MVViT depends on all 8 crops' losses                   │   │  │
│  │  │    • Model optimizes for COLLECTIVE performance across views       │   │  │
│  │  │                                                                      │   │  │
│  │  │  ✓ Standard averaging is mathematically correct                    │   │  │
│  │  │    • No need for special multi-view loss weighting                 │   │  │
│  │  │    • PyTorch's mean() gives equal importance to all crops          │   │  │
│  │  │    • Group structure preserved through feature interactions        │   │  │
│  │  │                                                                      │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │  Code Implementation (from multi_view_soft_teacher.py):            │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │                                                                      │   │  │
│  │  │  ```python                                                          │   │  │
│  │  │  # Step 1: Extract features with cross-view attention              │   │  │
│  │  │  x = self.student.extract_feat(batch_inputs)                       │   │  │
│  │  │  # x shape: (B*V, C, H, W) = (32, 256, H, W)                      │   │  │
│  │  │  # MVViT inside backbone created cross-view relationships          │   │  │
│  │  │                                                                      │   │  │
│  │  │  # Step 2: Compute per-crop losses (standard detection)            │   │  │
│  │  │  rpn_losses, rpn_results = self.rpn_loss_by_pseudo_instances(      │   │  │
│  │  │      x, batch_data_samples)  # 32 crops processed                  │   │  │
│  │  │  losses.update(**rpn_losses)                                        │   │  │
│  │  │                                                                      │   │  │
│  │  │  losses.update(**self.rcnn_cls_loss_by_pseudo_instances(           │   │  │
│  │  │      x, rpn_results, batch_data_samples, batch_info))              │   │  │
│  │  │                                                                      │   │  │
│  │  │  losses.update(**self.rcnn_reg_loss_by_pseudo_instances(           │   │  │
│  │  │      x, rpn_results, batch_data_samples))                          │   │  │
│  │  │                                                                      │   │  │
│  │  │  # Step 3: PyTorch already averaged over 32 crops                  │   │  │
│  │  │  # losses = {                                                       │   │  │
│  │  │  #   'loss_rpn_cls': mean over 32 crops,                           │   │  │
│  │  │  #   'loss_cls': mean over 32 crops,                               │   │  │
│  │  │  #   ...                                                            │   │  │
│  │  │  # }                                                                 │   │  │
│  │  │                                                                      │   │  │
│  │  │  # Note: Group structure maintained through feature graph          │   │  │
│  │  │  # Gradient ∂Loss/∂θ flows through MVViT to all 8 crops            │   │  │
│  │  │  ```                                                                 │   │  │
│  │  │                                                                      │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │  Comparison: Multi-View vs Single-View Loss                        │   │  │
│  │  │  ──────────────────────────────────────────────────────────────    │   │  │
│  │  │                                                                      │   │  │
│  │  │  Single-View (baseline):                                            │   │  │
│  │  │  • Features: f_i independent for each image                        │   │  │
│  │  │  • Loss: mean(loss_0, loss_1, ..., loss_31)                       │   │  │
│  │  │  • No relationship between images                                   │   │  │
│  │  │                                                                      │   │  │
│  │  │  Multi-View (ours):                                                 │   │  │
│  │  │  • Features: f_i depends on f_j (j in same group) via MVViT       │   │  │
│  │  │  • Loss: mean(loss_0, loss_1, ..., loss_31)  [same formula!]     │   │  │
│  │  │  • BUT gradient ∂loss_i/∂f_j ≠ 0 for same group                   │   │  │
│  │  │  • Model learns to leverage complementary views                     │   │  │
│  │  │                                                                      │   │  │
│  │  └─────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                              │  │
│  │  Typical Loss Values:                                                        │  │
│  │  • Initial: total_loss ≈ 2.0-3.0                                            │  │
│  │  • Converged: total_loss ≈ 0.5-1.0                                          │  │
│  │  • Supervised contributes ~60-70% of total loss                             │  │
│  │  • Unsupervised contributes ~30-40% of total loss                           │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FPN NECK STAGE                                          │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Feature Pyramid Network (FPN)                                               │  │
│  │  • Top-down pathway with lateral connections                                 │  │
│  │  • All levels unified to 256 channels                                        │  │
│  │                                                                              │  │
│  │  Input: [P2', P3', P4', P5'] from MVViT                                     │  │
│  │         • P2': (B×8, 256, 64, 180)                                          │  │
│  │         • P3': (B×8, 512, 32, 90)                                           │  │
│  │         • P4': (B×8, 1024, 16, 45)                                          │  │
│  │         • P5': (B×8, 2048, 8, 22)                                           │  │
│  │                                                                              │  │
│  │  Output: 5 FPN levels [FPN_P2, FPN_P3, FPN_P4, FPN_P5, FPN_P6]             │  │
│  │         • All with 256 channels                                              │  │
│  │         • FPN_P2: (B×8, 256, 64, 180)   [stride=4]                          │  │
│  │         • FPN_P3: (B×8, 256, 32, 90)    [stride=8]                          │  │
│  │         • FPN_P4: (B×8, 256, 16, 45)    [stride=16]                         │  │
│  │         • FPN_P5: (B×8, 256, 8, 22)     [stride=32]                         │  │
│  │         • FPN_P6: (B×8, 256, 4, 11)     [stride=64]                         │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RPN HEAD STAGE                                          │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  Region Proposal Network (RPN)                                               │  │
│  │  • Per-view proposal generation                                              │  │
│  │                                                                              │  │
│  │  Anchor Configuration:                                                       │  │
│  │  • Scales: [4, 8, 16] - Multiple sizes                                      │  │
│  │  • Ratios: [0.2, 0.35, 0.5, 1.0, 2.0] - Handle narrow objects              │  │
│  │  • Strides: [4, 8, 16, 32, 64] - Multi-scale                                │  │
│  │  • Total anchors per location: 3×5 = 15                                     │  │
│  │                                                                              │  │
│  │  Loss Functions:                                                             │  │
│  │  • Classification: Focal Loss                                                │  │
│  │    - gamma=2.0 (focus on hard examples)                                     │  │
│  │    - alpha=0.5 (weight foreground)                                          │  │
│  │    - loss_weight=3.0 (emphasize classification)                             │  │
│  │  • Regression: L1 Loss (loss_weight=1.0)                                    │  │
│  │                                                                              │  │
│  │  Training Config:                                                            │  │
│  │  • Positive IoU threshold: 0.5 (relaxed for narrow objects)                │  │
│  │  • Negative IoU threshold: 0.3                                              │  │
│  │  • num_samples: 256 per image                                               │  │
│  │  • pos_fraction: 0.8 (80% positive samples)                                 │  │
│  │  • neg_pos_ub: 1 (max 1:1 ratio)                                           │  │
│  │                                                                              │  │
│  │  Output per image: ~1000 proposals                                          │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            ROI HEAD STAGE                                            │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  RoI Align + Bbox Head (Faster R-CNN style)                                 │  │
│  │  • Per-view detection on each crop independently                            │  │
│  │                                                                              │  │
│  │  RoI Align:                                                                  │  │
│  │  • Pool proposals to fixed size: 7×7                                        │  │
│  │  • Bilinear sampling                                                         │  │
│  │                                                                              │  │
│  │  Bbox Head:                                                                  │  │
│  │  • 2 FC layers: 1024 dims each                                              │  │
│  │  • Classification head: 5 classes (defects)                                 │  │
│  │    - Broken, Chipped, Scratched, Severe_Rust, Tip_Wear                     │  │
│  │  • Regression head: 4 values (bbox refinement)                              │  │
│  │                                                                              │  │
│  │  Loss Functions:                                                             │  │
│  │  • Classification: Focal Loss                                                │  │
│  │    - gamma=2.0, alpha=0.5, loss_weight=2.0                                  │  │
│  │  • Regression: Smooth L1 Loss                                               │  │
│  │                                                                              │  │
│  │  Training Config:                                                            │  │
│  │  • Positive IoU threshold: 0.5                                              │  │
│  │  • num_samples: 256 per image                                               │  │
│  │  • pos_fraction: 0.75 (75% positive samples)                                │  │
│  │                                                                              │  │
│  │  Test Config:                                                                │  │
│  │  • Score threshold: 0.05                                                     │  │
│  │  • NMS IoU threshold: 0.5                                                    │  │
│  │  • Max detections per image: 100                                            │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  Output per image (B×8 total):                                                      │
│  • Bboxes: (N, 4) - [x1, y1, x2, y2]                                               │
│  • Scores: (N,) - Confidence scores                                                 │
│  • Labels: (N,) - Class predictions (0-4)                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                         ↓

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION & AGGREGATION STAGE                                │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  MultiViewCocoMetric                                                         │  │
│  │  • Aggregate predictions from 8 crops back to base image                    │  │
│  │                                                                              │  │
│  │  Per Crop Predictions:                                                       │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐       ┌────────────┐       │  │
│  │  │ Crop 0     │  │ Crop 1     │  │ Crop 2     │  ...  │ Crop 7     │       │  │
│  │  │ • Boxes: N0│  │ • Boxes: N1│  │ • Boxes: N2│       │ • Boxes: N7│       │  │
│  │  │ • Scores   │  │ • Scores   │  │ • Scores   │       │ • Scores   │       │  │
│  │  └────────────┘  └────────────┘  └────────────┘       └────────────┘       │  │
│  │                                                                              │  │
│  │                                ↓                                             │  │
│  │                                                                              │  │
│  │  ┌────────────────────────────────────────────────────────────────┐         │  │
│  │  │  Aggregation Method: Weighted Box Fusion (WBF)                 │         │  │
│  │  │  • Transform boxes to base image coordinates                   │         │  │
│  │  │  • Cluster overlapping boxes (IoU > 0.5)                       │         │  │
│  │  │  • Fuse boxes in each cluster:                                 │         │  │
│  │  │    - Weighted average of coordinates by confidence scores      │         │  │
│  │  │    - Average scores across all views                           │         │  │
│  │  │  • Utilizes information from all 8 views (better than NMS)     │         │  │
│  │  └────────────────────────────────────────────────────────────────┘         │  │
│  │                                                                              │  │
│  │                                ↓                                             │  │
│  │                                                                              │  │
│  │  Final Predictions per Base Image:                                          │  │
│  │  • Aggregated bboxes with fused coordinates                                 │  │
│  │  • Consensus scores from multiple views                                     │  │
│  │  • More robust and accurate than single-view                                │  │
│  │                                                                              │  │
│  │  COCO Metrics:                                                               │  │
│  │  • AP, AP50, AP75 (IoU thresholds)                                          │  │
│  │  • AP_small, AP_medium, AP_large (object sizes)                             │  │
│  │  • Per-class AP (5 defect types)                                            │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Chi Tiết Kiến Trúc MVViT (Multi-View Vision Transformer)

### 1. Kiến Trúc Tổng Thể

```
MVViT Module (moderate mode - Optimized for 32GB GPU)
├── Spatial Attention: 'moderate' → 512 tokens per view
├── Number of Transformer Layers: 1
├── Attention Heads: 4
├── Embedding Dimension: 256
├── MLP Ratio: 2.0 (feedforward = 512 dims)
├── Gradient Checkpointing: Enabled
└── Memory Usage: ~8-16GB
```

### 2. Chi Tiết Attention Pooling Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  MODERATE MODE: Attention Pooling + Cross-View Attention        │
│  Best trade-off between memory and accuracy                     │
└─────────────────────────────────────────────────────────────────┘

Input: (B, V=8, C=256, H, W) - Per-view features from ResNet

Step 1: Flatten Spatial Dimensions
  (B, 8, 256, H, W) → (B×8, H×W, 256)

Step 2: Add Spatial Positional Embeddings
  pos_embed: (H×W, 256) - Learned positional encoding
  tokens = tokens + pos_embed

Step 3: Attention Pooling (SPATIAL REDUCTION)
  ┌────────────────────────────────────────────────────┐
  │  K = min(512, H×W) learnable pooling queries       │
  │  Q: (B×8, K=512, 256) - Learnable queries         │
  │  K, V: (B×8, H×W, 256) - Spatial tokens           │
  │                                                    │
  │  Attention Score: Q @ K^T / √256                   │
  │  → (B×8, 512, H×W)                                 │
  │                                                    │
  │  Attention Output: Attention @ V                   │
  │  → (B×8, 512, 256)                                 │
  │                                                    │
  │  Result: Each view compressed to 512 tokens        │
  │  Memory Saved: H×W (e.g., 2880) → 512 tokens      │
  └────────────────────────────────────────────────────┘

Step 4: Reshape for Cross-View Processing
  (B×8, 512, 256) → (B, 8×512=4096, 256)

Step 5: Add View Positional Embeddings
  view_pos: (8, 256) - Learned view-specific encoding
  Broadcast to all K tokens per view

Step 6: Transformer Encoder (CROSS-VIEW ATTENTION)
  ┌────────────────────────────────────────────────────┐
  │  Input: (B, 4096, 256) - All views' pooled tokens │
  │                                                    │
  │  Multi-Head Self-Attention (4 heads):              │
  │  • Q, K, V: (B, 4096, 256)                        │
  │  • Attention: (B, 4, 4096, 4096)                  │
  │  • Each token can attend to ALL views' tokens     │
  │  • Learns: Which views provide complementary info │
  │                                                    │
  │  Feedforward Network:                              │
  │  • Linear: 256 → 512 (MLP)                        │
  │  • GELU activation                                 │
  │  • Linear: 512 → 256                              │
  │  • Dropout: 0.1                                    │
  │                                                    │
  │  Residual + LayerNorm after each sublayer         │
  │  Gradient Checkpointing: Saves ~50% memory        │
  └────────────────────────────────────────────────────┘

Step 7: Attention Upsampling (SPATIAL RECONSTRUCTION)
  ┌────────────────────────────────────────────────────┐
  │  Reconstruct full spatial resolution               │
  │  H×W learnable upsample queries: (H×W, 256)       │
  │  Q: (B×8, H×W, 256) - Spatial queries             │
  │  K, V: (B×8, 512, 256) - Pooled tokens            │
  │                                                    │
  │  Attention Score: Q @ K^T / √256                   │
  │  → (B×8, H×W, 512)                                 │
  │                                                    │
  │  Attention Output: Attention @ V                   │
  │  → (B×8, H×W, 256)                                 │
  │                                                    │
  │  Reshape: (B, 8, 256, H, W)                       │
  └────────────────────────────────────────────────────┘

Step 8: Residual Connection
  refined = 0.5 × refined + 0.5 × original
  Output: (B, 8, 256, H, W)
```

### 3. Complexity Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  Computational Complexity Comparison                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Full Attention (NOT USED - OOM on 32GB):                       │
│  • Tokens per batch: B × V × H × W = 4 × 8 × 45 × 16 = 23,040  │
│  • Attention complexity: O(N²) = O(23,040²) ≈ 531M operations   │
│  • Memory: ~20-30GB just for attention weights                  │
│  ✗ OOM on 32GB GPU!                                             │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Moderate Mode (USED):                                          │
│  • Attention Pooling: O(H×W × K) = O(720 × 512) ≈ 368K ops     │
│  • Cross-View Attention: O((V×K)²) = O((8×512)²) ≈ 16M ops     │
│  • Attention Upsampling: O(H×W × K) = O(720 × 512) ≈ 368K ops  │
│  • Total: ~17M operations (31× less than full!)                 │
│  • Memory: ~8-16GB (with gradient checkpointing)                │
│  ✓ Fits comfortably on 32GB GPU                                 │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Memory Breakdown (Batch Size=4, V=8, P4 level):                │
│  • Input features: 4×8×256×16×45 × 4 bytes ≈ 2.4 MB            │
│  • Pooling queries: 512×256 × 4 bytes ≈ 0.5 MB                 │
│  • Attention weights (pooling): 32×512×720 × 4 bytes ≈ 47 MB   │
│  • Cross-view tokens: 4×4096×256 × 4 bytes ≈ 16 MB             │
│  • Transformer attention: 4×4×4096×4096 × 4 bytes ≈ 1 GB       │
│  • Gradients (with checkpointing): ~4 GB                        │
│  • Total per forward-backward: ~6-8 GB                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Điểm Khác Biệt Chính

### 1. So với Soft Teacher gốc:
- **Multi-view processing**: Xử lý 8 crops từ 1 ảnh base thay vì 1 ảnh đơn
- **Cross-view attention (MVViT)**: Học mối quan hệ giữa các views
- **Aggregation**: Fuse predictions từ nhiều views bằng WBF

### 2. So với các multi-view methods khác:
- **End-to-end learning**: Không cần post-processing phức tạp
- **Attention-based fusion**: Thay vì simple averaging
- **Semi-supervised**: Tận dụng unlabeled data với pseudo-labeling

### 3. Tối ưu hóa cho 32GB GPU:
- **Gradient checkpointing**: Giảm 50% memory
- **Attention pooling**: Giảm spatial tokens trước transformer
- **Moderate mode**: Cân bằng giữa accuracy và memory

---

## Từ Soft Teacher Gốc → Multi-View Soft Teacher: Các Thay Đổi Cần Thiết

### **OVERVIEW: 5 Thành Phần Chính Cần Thay Đổi**

```
┌────────────────────────────────────────────────────────────────────┐
│  Soft Teacher (Gốc)              →    Multi-View Soft Teacher      │
├────────────────────────────────────────────────────────────────────┤
│  1. Dataset                      →    MultiViewFromFolder          │
│  2. Collate Function             →    multi_view_collate_flatten   │
│  3. Backbone                     →    MultiViewBackbone + MVViT    │
│  4. Detector                     →    MultiViewSoftTeacher         │
│  5. Evaluation Metric            →    MultiViewCocoMetric          │
└────────────────────────────────────────────────────────────────────┘
```

---

### **1. DATASET LAYER (Thay đổi QUAN TRỌNG nhất)**

#### **Soft Teacher Gốc:**
```python
# configs/soft_teacher/soft_teacher_baseline.py
dataset_type = 'CocoDataset'

labeled_dataset = dict(
    type='CocoDataset',
    data_root='data/',
    ann_file='annotations/instances_train.json',
    data_prefix=dict(img='train/'),
    pipeline=sup_pipeline
)

unlabeled_dataset = dict(
    type='CocoDataset',
    data_root='data/',
    ann_file='annotations/instances_unlabeled.json',
    data_prefix=dict(img='train/'),
    pipeline=unsup_pipeline
)

train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='pseudo_collate'),  # Standard collate
    dataset=dict(
        type='ConcatDataset',
        datasets=[labeled_dataset, unlabeled_dataset]
    )
)
```

**Output mỗi sample:**
- `inputs`: (C, H, W) - Ảnh đơn
- `data_samples`: DetDataSample - GT cho 1 ảnh

---

#### **Multi-View Soft Teacher (Thay đổi):**
```python
# configs/soft_teacher/soft_teacher_custom_multi_view.py
# FILE MỚI: mmdet/datasets/wrappers/multi_view_from_folder.py

labeled_dataset = dict(
    type='MultiViewFromFolder',  # ← THAY ĐỔI 1: Dataset mới
    data_root=data_root,
    views_per_sample=8,  # ← THÊM: Số crops per group
    ann_file=data_root + 'semi_anno_multiview/_annotations.coco.labeled.grouped@60.json',
    data_prefix=dict(img='train/'),
    pipeline=sup_pipeline
)

unlabeled_dataset = dict(
    type='MultiViewFromFolder',  # ← THAY ĐỔI 1
    data_root=data_root,
    views_per_sample=8,
    ann_file=data_root + 'semi_anno_multiview/_annotations.coco.unlabeled.grouped@60.json',
    data_prefix=dict(img='train/'),
    pipeline=unsup_pipeline
)

train_dataloader = dict(
    batch_size=4,  # 4 groups × 8 = 32 images
    collate_fn=dict(
        type='mmdet.datasets.wrappers.multi_view_from_folder.multi_view_collate_flatten'
    ),  # ← THAY ĐỔI 2: Collate function mới
    dataset=dict(
        type='ConcatDataset',
        datasets=[labeled_dataset, unlabeled_dataset]
    )
)
```

**Output mỗi sample:**
- `inputs`: (V, C, H, W) = (8, 3, 256, 720) - 8 crops
- `data_samples`: List[8] of DetDataSample - GT cho từng crop

**Yêu cầu annotation JSON:**
- Thêm field `base_img_id` trong `images` để group crops
- Example:
```json
{
  "images": [
    {"id": 1, "file_name": "S1_bright_0_crop_0.jpg", "base_img_id": "S1_bright_0"},
    {"id": 2, "file_name": "S1_bright_0_crop_1.jpg", "base_img_id": "S1_bright_0"},
    ...
  ]
}
```

---

### **2. COLLATE FUNCTION (Xử lý Multi-Branch)**

#### **Soft Teacher Gốc:**
```python
# mmdet/datasets/pseudo_collate.py (simplified)
def pseudo_collate(batch):
    # batch: list of {'inputs': (C,H,W), 'data_samples': DetDataSample}
    
    # Stack into batches
    inputs_list = [item['inputs'] for item in batch]
    data_samples = [item['data_samples'] for item in batch]
    
    return {
        'inputs': inputs_list,  # List[(C,H,W)] length B
        'data_samples': data_samples  # List[DetDataSample] length B
    }
```

**Output:**
- `inputs`: List của B ảnh đơn
- `data_samples`: List của B DetDataSample

---

#### **Multi-View Soft Teacher (Thay đổi):**
```python
# FILE MỚI: mmdet/datasets/wrappers/multi_view_from_folder.py
def multi_view_collate_flatten(batch):
    # batch: list of {'inputs': (V,C,H,W), 'data_samples': List[V]}
    
    # Separate labeled/unlabeled
    sup_items = []
    unsup_items = []
    for item in batch:
        # Check if has GT labels
        if has_labels(item):
            sup_items.append(item)
        else:
            unsup_items.append(item)
    
    # Stack and flatten: (B, V, C, H, W) → (B×V, C, H, W)
    def _pack_and_flatten(items):
        stacked = torch.stack([it['inputs'] for it in items])  # (B, V, C, H, W)
        B, V, C, H, W = stacked.shape
        flat = stacked.reshape(B * V, C, H, W)
        flat_list = list(torch.unbind(flat, dim=0))  # List[B×V]
        
        # Flatten data_samples too
        flat_ds = []
        for it in items:
            flat_ds.extend(it['data_samples'])  # List[V] → flat list
        
        return flat_list, flat_ds
    
    sup_inputs, sup_ds = _pack_and_flatten(sup_items)
    unsup_inputs, unsup_ds = _pack_and_flatten(unsup_items)
    
    # Multi-branch output
    return {
        'inputs': {
            'sup': sup_inputs,  # List[(C,H,W)] length B×V
            'unsup_teacher': unsup_inputs,
            'unsup_student': unsup_inputs
        },
        'data_samples': {
            'sup': sup_ds,  # List[DetDataSample] length B×V
            'unsup_teacher': unsup_ds,
            'unsup_student': unsup_ds
        }
    }
```

**Output:**
- `inputs`: Dict với 3 branches, mỗi branch là List[(C,H,W)] length B×V
- `data_samples`: Dict với 3 branches, mỗi branch là List[DetDataSample] length B×V

**Lý do flatten:**
- DetDataPreprocessor expects List[(C,H,W)]
- MultiViewBackbone sẽ reshape lại (B×V, C, H, W) → (B, V, ...)

---

### **3. BACKBONE (Thêm Multi-View Fusion)**

#### **Soft Teacher Gốc:**
```python
# configs/soft_teacher/soft_teacher_baseline.py
detector = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'
    ),
    neck=dict(type='FPN', ...),
    rpn_head=dict(...),
    roi_head=dict(...)
)
```

**Feature flow:**
```
Input: (B, C, H, W) 
  ↓
ResNet: Per-image features independently
  ↓
Output: (B, 256, H', W') per FPN level
```

---

#### **Multi-View Soft Teacher (Thay đổi):**
```python
# FILE MỚI: mmdet/models/utils/multi_view.py
# FILE MỚI: mmdet/models/utils/multi_view_transformer.py

detector.backbone = dict(
    type='MultiViewBackbone',  # ← THAY ĐỔI 3: Wrapper mới
    backbone=dict(  # Inner backbone giữ nguyên
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'
    ),
    fusion='mvvit',  # ← THÊM: Fusion strategy
    views_per_sample=8,  # ← THÊM
    mvvit=dict(  # ← THÊM: MVViT config
        type='MVViT',
        embed_dim=256,
        num_heads=4,
        num_layers=1,
        spatial_attention='moderate',
        use_gradient_checkpointing=True
    )
)
```

**Feature flow:**
```
Input: (B×V, C, H, W) = (32, 3, 256, 720)
  ↓
MultiViewBackbone.forward():
  1. Reshape: (32, ...) → (4, 8, ...)  [infer B, V]
  2. ResNet: Extract per-crop features → (32, 256, H', W')
  3. Reshape for MVViT: (32, ...) → (4, 8, 256, H', W')
  4. MVViT: Cross-view attention
     • Crop 0 attends to Crops 1-7 in same group
     • Output: (4, 8, 256, H', W') with cross-view context
  5. Flatten back: (4, 8, ...) → (32, 256, H', W')
  ↓
Output: (B×V, 256, H', W') per FPN level
  → Features now contain cross-view information!
```

**Code structure:**
```python
# mmdet/models/utils/multi_view.py
class MultiViewBackbone(nn.Module):
    def forward(self, x):
        # x: (B*V, C, H, W) flattened input
        
        if self.fusion == 'mvvit':
            # Infer B and V
            BV, C, H, W = x.shape
            V = self._views_per_sample
            B = BV // V
            
            # Extract per-crop features
            feats = self.backbone(x)  # ResNet
            # feats: (B*V, 256, H', W') per FPN level
            
            # Apply MVViT fusion
            feats = self._apply_mvvit_fusion(feats, B, V)
            # feats: (B*V, 256, H', W') with cross-view context
            
        return feats

# mmdet/models/utils/multi_view_transformer.py
class MVViT(nn.Module):
    def forward(self, features_views):
        # features_views: List[V] of (B, C, H, W)
        
        for each FPN level:
            # 1. Attention Pooling: H×W → K tokens
            # 2. Cross-view Transformer on V×K tokens
            # 3. Attention Upsampling: K → H×W tokens
            # 4. Residual connection
        
        return refined_features_views
```

---

### **4. DETECTOR (Xử lý Multi-View Loss)**

#### **Soft Teacher Gốc:**
```python
# mmdet/models/detectors/soft_teacher.py
@MODELS.register_module()
class SoftTeacher(SemiBaseDetector):
    def loss_by_pseudo_instances(self, batch_inputs, batch_data_samples, batch_info):
        # batch_inputs: (B, C, H, W)
        # batch_data_samples: List[B] of DetDataSample
        
        # Extract features
        x = self.student.extract_feat(batch_inputs)
        # x: (B, 256, H', W') per FPN level
        
        # Compute losses (standard)
        losses = {}
        rpn_losses, rpn_results = self.rpn_loss_by_pseudo_instances(x, batch_data_samples)
        losses.update(**rpn_losses)
        losses.update(**self.rcnn_cls_loss_by_pseudo_instances(...))
        losses.update(**self.rcnn_reg_loss_by_pseudo_instances(...))
        
        # Apply unsup weight
        return rename_loss_dict('unsup_', reweight_loss_dict(losses, 0.5))
```

**Loss computation:**
- Each image processed independently
- Loss averaged over B images
- No cross-image relationships

---

#### **Multi-View Soft Teacher (Thay đổi):**
```python
# FILE MỚI: mmdet/models/detectors/multi_view_soft_teacher.py
@MODELS.register_module()
class MultiViewSoftTeacher(SoftTeacher):  # ← THAY ĐỔI 4: Detector mới
    def __init__(self, ..., views_per_sample=8, ...):
        super().__init__(...)
        self.views_per_sample = views_per_sample  # ← THÊM
    
    def loss_by_pseudo_instances(self, batch_inputs, batch_data_samples, batch_info):
        # batch_inputs: (B×V, C, H, W) = (32, 3, 256, 720)
        # batch_data_samples: List[B×V] of DetDataSample = List[32]
        
        # Extract features WITH CROSS-VIEW ATTENTION
        x = self.student.extract_feat(batch_inputs)
        # x: (B×V, 256, H', W') per FPN level
        # ← KEY DIFFERENCE: Features contain cross-view context from MVViT!
        
        # Compute per-crop losses (same as parent)
        # Each crop has its own GT, so losses are independent
        losses = {}
        rpn_losses, rpn_results = self.rpn_loss_by_pseudo_instances(x, batch_data_samples)
        losses.update(**rpn_losses)
        losses.update(**self.rcnn_cls_loss_by_pseudo_instances(...))
        losses.update(**self.rcnn_reg_loss_by_pseudo_instances(...))
        
        # losses = {
        #   'loss_rpn_cls': mean over 32 crops,
        #   'loss_cls': mean over 32 crops,
        #   ...
        # }
        
        # NOTE: No special aggregation needed!
        # PyTorch's mean() automatically averages over B×V crops
        # = (1/4) × Σ_groups[(1/8) × Σ_crops]
        # Group structure preserved through MVViT gradient flow
        
        # Apply unsup weight
        return rename_loss_dict('unsup_', reweight_loss_dict(losses, 0.5))
```

**Loss computation:**
- Each crop processed independently through detection heads
- **BUT** features already contain cross-view context from MVViT
- Loss averaged over B×V crops (standard PyTorch mean)
- Gradient flows through MVViT, connecting all crops in same group

**KEY INSIGHT:**
```python
# Multi-view learning does NOT happen in loss computation!
# It happens in FEATURE EXTRACTION via MVViT attention.

# Gradient flow:
# loss_crop_0 → ∂Loss/∂features_crop_0 → ∂Loss/∂MVViT → ∂Loss/∂features_crop_{1-7}
#                                            ↑
#                                    Cross-view connections!
```

---

### **5. EVALUATION METRIC (Aggregate Predictions)**

#### **Soft Teacher Gốc:**
```python
# configs/soft_teacher/soft_teacher_baseline.py
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/annotations/instances_val.json',
    metric='bbox',
    format_only=False
)
```

**Evaluation:**
- 1 prediction per image
- Direct COCO evaluation

---

#### **Multi-View Soft Teacher (Thay đổi):**
```python
# FILE MỚI: mmdet/evaluation/metrics/multi_view_coco_metric.py

val_evaluator = dict(
    type='MultiViewCocoMetric',  # ← THAY ĐỔI 5: Metric mới
    ann_file=data_root + 'anno_valid/_annotations_filtered.coco.json',
    metric='bbox',
    views_per_sample=8,  # ← THÊM
    aggregation='wbf',  # ← THÊM: Fusion method
    nms_iou_thr=0.5,
    extract_base_name=True
)
```

**Evaluation flow:**
```
Predictions: 8 crops per base image
  ↓
MultiViewCocoMetric.process():
  1. Group predictions by base_img_id
     • Crop 0: [(bbox_0_0, score_0_0), ...]
     • Crop 1: [(bbox_1_0, score_1_0), ...]
     • ...
     • Crop 7: [(bbox_7_0, score_7_0), ...]
  
  2. Transform to base image coordinates
     • Apply inverse crop transform
  
  3. Weighted Box Fusion (WBF)
     • Cluster overlapping boxes (IoU > 0.5)
     • Fuse coordinates: weighted average by scores
     • Fuse scores: average across views
  
  4. Output: 1 set of predictions per base image
  ↓
COCO evaluation on aggregated predictions
```

**Code structure:**
```python
# mmdet/evaluation/metrics/multi_view_coco_metric.py
class MultiViewCocoMetric(CocoMetric):
    def process(self, data_batch, data_samples):
        # data_samples: List[B×V] of DetDataSample with predictions
        
        # Group by base_img_id
        grouped_preds = self._group_predictions(data_samples)
        # grouped_preds: {base_img_id: [pred_crop_0, ..., pred_crop_7]}
        
        # Aggregate predictions
        aggregated = []
        for base_id, crops_preds in grouped_preds.items():
            if self.aggregation == 'wbf':
                fused_pred = self._weighted_box_fusion(crops_preds)
            aggregated.append(fused_pred)
        
        # Standard COCO evaluation
        super().process(data_batch, aggregated)
```

---

## **TÓM TẮT: Checklist Thay Đổi**

### **✅ Files Cần Tạo Mới:**
1. `mmdet/datasets/wrappers/multi_view_from_folder.py`
   - Class `MultiViewFromFolder(Dataset)`
   - Function `multi_view_collate_flatten(batch)`

2. `mmdet/models/utils/multi_view.py`
   - Class `MultiViewBackbone(nn.Module)`

3. `mmdet/models/utils/multi_view_transformer.py`
   - Class `MVViT(nn.Module)`
   - Class `TransformerEncoderLayer(nn.Module)`

4. `mmdet/models/detectors/multi_view_soft_teacher.py`
   - Class `MultiViewSoftTeacher(SoftTeacher)`

5. `mmdet/evaluation/metrics/multi_view_coco_metric.py`
   - Class `MultiViewCocoMetric(CocoMetric)`

### **✅ Config Changes:**
```python
# Soft Teacher Gốc → Multi-View Soft Teacher

# 1. Dataset
- type='CocoDataset'
+ type='MultiViewFromFolder'
+ views_per_sample=8

# 2. Collate
- collate_fn=dict(type='pseudo_collate')
+ collate_fn=dict(type='mmdet.datasets.wrappers.multi_view_from_folder.multi_view_collate_flatten')

# 3. Backbone
- backbone=dict(type='ResNet', ...)
+ backbone=dict(
+     type='MultiViewBackbone',
+     backbone=dict(type='ResNet', ...),
+     fusion='mvvit',
+     views_per_sample=8,
+     mvvit=dict(type='MVViT', ...)
+ )

# 4. Detector
- type='SoftTeacher'
+ type='MultiViewSoftTeacher'
+ views_per_sample=8

# 5. Evaluator
- type='CocoMetric'
+ type='MultiViewCocoMetric'
+ views_per_sample=8
+ aggregation='wbf'
```

### **✅ Annotation Format:**
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "S1_bright_0_crop_0.jpg",
      "base_img_id": "S1_bright_0"  ← THÊM field này
    }
  ]
}
```

### **✅ Registry Registrations:**
```python
# mmdet/datasets/wrappers/__init__.py
from .multi_view_from_folder import MultiViewFromFolder

# mmdet/models/utils/__init__.py
from .multi_view import MultiViewBackbone
from .multi_view_transformer import MVViT

# mmdet/models/detectors/__init__.py
from .multi_view_soft_teacher import MultiViewSoftTeacher

# mmdet/evaluation/metrics/__init__.py
from .multi_view_coco_metric import MultiViewCocoMetric
```

---

## **QUAN TRỌNG: Các Điểm Cần Lưu Ý**

### **1. Input/Output Shape Compatibility:**
```
Dataset → Collate → DataPreprocessor → Backbone → Detection Heads
(V,C,H,W) → (B×V,C,H,W) → (B×V,C,H,W) → (B×V,256,H',W') → Per-crop predictions
              ↑ Flatten                    ↑ Cross-view attention inside
```

### **2. Loss Computation:**
- ❌ KHÔNG cần custom loss aggregation
- ✅ Multi-view learning qua MVViT attention trong features
- ✅ Standard PyTorch mean() over B×V crops

### **3. Gradient Flow:**
```
∂Loss/∂crop_0 → ∂Loss/∂MVViT → affects all 8 crops in group
                     ↑
              Cross-view connections
```

### **4. Memory Management:**
- Gradient checkpointing trong MVViT
- Attention pooling (H×W → 512 tokens)
- Batch size = 4 groups → 32 images total

## Các Thành Phần Chính

### 1. Data Pipeline
- `MultiViewFromFolder`: Group 8 crops thành 1 sample
- `multi_view_collate_flatten`: Flatten (B,V,C,H,W) → (B×V,C,H,W)
- Augmentation: Weak (teacher) vs Strong (student)

### 2. Backbone
- `MultiViewBackbone`: Wrapper reshape và fuse features
- `ResNet-50`: Pretrained Caffe, frozen Stage 1
- `MVViT`: Cross-view transformer fusion

### 3. Detection Heads
- `RPNHead`: Focal Loss, relaxed IoU, high pos_fraction
- `BboxHead`: Focal Loss, 5 classes, smooth L1 regression

### 4. Semi-Supervised Learning
- `MultiViewSoftTeacher`: Teacher-student framework
- EMA update: momentum=0.999
- Pseudo-labeling: threshold=0.7, max 15 boxes/image
- Loss weighting: supervised=1.0, unsupervised=0.5

### 5. Evaluation
- `MultiViewCocoMetric`: Aggregate predictions bằng WBF
- COCO metrics: AP, AP50, AP75, per-class AP

## Tham Số Quan Trọng

```python
# Multi-view
views_per_sample = 8
batch_size = 4  # → 32 images per batch

# MVViT
embed_dim = 256
num_heads = 4
num_layers = 1
spatial_attention = 'moderate'  # 512 tokens/view
use_gradient_checkpointing = True

# Semi-supervised
freeze_teacher = True
sup_weight = 1.0
unsup_weight = 0.5
pseudo_label_initial_score_thr = 0.7
cls_pseudo_thr = 0.6
rpn_pseudo_thr = 0.5
reg_pseudo_thr = 0.02
max_boxes_per_image = 15
jitter_times = 5
jitter_scale = 0.03

# Training
max_iters = 40000
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0001
warmup_iters = 5000

# Loss weights & parameters
# RPN
rpn_cls_loss_weight = 3.0      # Focal Loss
rpn_cls_gamma = 2.0
rpn_cls_alpha = 0.5
rpn_bbox_loss_weight = 1.0     # L1 Loss

# ROI Head
roi_cls_loss_weight = 2.0      # Focal Loss
roi_cls_gamma = 2.0
roi_cls_alpha = 0.5
roi_bbox_loss_weight = 1.0     # Smooth L1 Loss
roi_bbox_beta = 1.0

# Training config
rpn_pos_iou_thr = 0.5
rpn_neg_iou_thr = 0.3
rpn_num_samples = 256
rpn_pos_fraction = 0.8
rpn_neg_pos_ub = 1

rcnn_pos_iou_thr = 0.5
rcnn_num_samples = 256
rcnn_pos_fraction = 0.75
rcnn_neg_pos_ub = 1

# EMA update for teacher
ema_momentum = 0.999
```

## Loss Computation Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  Loss Flow in Multi-View Soft Teacher                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Training Batch: 4 groups × 8 crops = 32 images                │
│                                                                  │
│  1. SUPERVISED (Labeled: 60%, ~2.4 groups/batch)                │
│     ├─ Input: 2.4 × 8 ≈ 19 images with GT labels               │
│     ├─ MVViT: Cross-view attention on features                  │
│     ├─ RPN Loss: Focal Loss (cls) + L1 Loss (bbox)             │
│     ├─ ROI Loss: Focal Loss (cls) + Smooth L1 (bbox)           │
│     └─ Weight: 1.0 × (sum of above losses)                      │
│                                                                  │
│  2. UNSUPERVISED (Unlabeled: 40%, ~1.6 groups/batch)            │
│     ├─ Input: 1.6 × 8 ≈ 13 images without GT                   │
│     │                                                            │
│     ├─ Teacher Branch (Weak Aug):                               │
│     │   ├─ MVViT fusion on teacher features                     │
│     │   ├─ Generate predictions for all 13 images               │
│     │   ├─ Filter by score > 0.7                                │
│     │   ├─ Keep max 15 boxes per image                          │
│     │   ├─ Jitter boxes 5× for uncertainty                      │
│     │   ├─ Compute uncertainty: std / box_size                  │
│     │   └─ Filter by uncertainty < 0.02                         │
│     │                                                            │
│     ├─ Student Branch (Strong Aug):                             │
│     │   ├─ MVViT fusion on student features                     │
│     │   ├─ Learn from teacher's pseudo-labels                   │
│     │   ├─ RPN Loss: Filter by rpn_pseudo_thr (0.5)            │
│     │   ├─ CLS Loss: Filter by cls_pseudo_thr (0.6)            │
│     │   │            + Soft background scores                   │
│     │   └─ REG Loss: Filter by uncertainty < 0.02              │
│     │                                                            │
│     └─ Weight: 0.5 × (sum of pseudo-label losses)              │
│                                                                  │
│  3. TOTAL LOSS                                                   │
│     loss = sup_losses × 1.0 + unsup_losses × 0.5               │
│                                                                  │
│  4. BACKPROPAGATION                                              │
│     ├─ Student: Updated by gradient descent (lr=0.001)          │
│     ├─ Teacher: Updated by EMA (momentum=0.999)                 │
│     │   teacher = 0.999 × teacher + 0.001 × student            │
│     └─ MVViT: Learned cross-view relationships through losses   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Key Loss Design Choices:                                       │
│                                                                  │
│  ✓ Focal Loss: Handles class imbalance (BG >> FG)              │
│  ✓ High pos_fraction (0.8, 0.75): Emphasize sparse positives   │
│  ✓ Low neg_pos_ub (1:1): Prevent BG overwhelming               │
│  ✓ Uncertainty filtering: Remove unreliable pseudo-labels       │
│  ✓ Soft BG scores: Avoid overconfident on ambiguous regions    │
│  ✓ High pseudo threshold (0.7): Only high-conf pseudo-labels    │
│  ✓ Max boxes limit (15): Prevent teacher overgeneration         │
│  ✓ Unsup weight (0.5): Balance supervised vs unsupervised      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
