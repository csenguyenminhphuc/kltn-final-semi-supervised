# BLOCK DIAGRAMS - Multi-View Soft Teacher
## Dành cho vẽ sơ đồ khối (Block Diagrams)

---

## DIAGRAM 1: TỔNG QUAN KIẾN TRÚC (High-Level Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-VIEW SOFT TEACHER                          │
│                   (Semi-Supervised Detection)                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │          INPUT: Base Image                      │
        │     (1 drill bit → 8 crops = 1 group)          │
        └─────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │  LABELED DATA (60%) │           │ UNLABELED DATA (40%)│
    │  298 groups × 8     │           │  199 groups × 8     │
    │  = 2384 images      │           │  = 1592 images      │
    └─────────────────────┘           └─────────────────────┘
                │                                   │
                │                                   │
                ▼                                   ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │  TEACHER BRANCH     │           │  STUDENT BRANCH     │
    │  (EMA, frozen)      │           │  (Gradient descent) │
    │  Weak augmentation  │           │  Strong augmentation│
    └─────────────────────┘           └─────────────────────┘
                │                                   │
                │         Pseudo Labels             │
                └──────────────►────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │       SHARED MULTI-VIEW BACKBONE                │
        │    (ResNet-50 + MVViT Transformer)              │
        │    Cross-view attention learns relationships     │
        └─────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │         DETECTION HEADS (FPN + RPN + ROI)       │
        │    Per-crop predictions with cross-view context │
        └─────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │    LOSS COMPUTATION (Supervised + Unsupervised) │
        │    Multi-view learning via MVViT attention      │
        └─────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │   PREDICTION AGGREGATION (Weighted Box Fusion)  │
        │   8 crop predictions → 1 base image result      │
        └─────────────────────────────────────────────────┘
```

---

## DIAGRAM 2: DATA FLOW (Luồng Dữ Liệu Chi Tiết)

```
┌────────────────────────────────────────────────────────────────────┐
│                          INPUT STAGE                               │
└────────────────────────────────────────────────────────────────────┘

    Base Image: S1_bright_0.jpg (Original drill bit image)
                        │
                        ▼
            ┌───────────────────────┐
            │  Crop into 8 views    │
            │  (3×3 grid - 1 view)  │
            └───────────────────────┘
                        │
        ┌───────┬───────┼───────┬───────┬───────┬───────┬───────┐
        ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
    Crop_0  Crop_1  Crop_2  Crop_3  Crop_4  Crop_5  Crop_6  Crop_7
    256×720 256×720 256×720 256×720 256×720 256×720 256×720 256×720
        │       │       │       │       │       │       │       │
        └───────┴───────┴───────┴───────┴───────┴───────┴───────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Dataset: Load + GT   │
            │  MultiViewFromFolder  │
            └───────────────────────┘
                        │
                        ▼
        Shape: (8, 3, 256, 720) per sample
        GT: List[8] DetDataSample (per-crop annotations)
                        │
                        ▼
            ┌───────────────────────┐
            │  Collate: Batch + Flatten │
            │  multi_view_collate_flatten │
            └───────────────────────┘
                        │
                        ▼
        Shape: (B×8, 3, 256, 720) = (32, 3, 256, 720)
        GT: List[32] DetDataSample (flattened)
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────┐
│                     DATA PREPROCESSOR                              │
│  Normalize: mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0]  │
│  Pad to divisible by 8                                             │
└────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
        Input to backbone: (32, 3, 256, 720)
```

---

## DIAGRAM 3: MULTI-VIEW BACKBONE (Feature Extraction)

```
┌────────────────────────────────────────────────────────────────────┐
│                    MULTI-VIEW BACKBONE                             │
└────────────────────────────────────────────────────────────────────┘

Input: (B×V, 3, H, W) = (32, 3, 256, 720)
                │
                ▼
    ┌───────────────────────────────┐
    │  MultiViewBackbone.forward()  │
    │  • Infer B=4, V=8             │
    │  • Reshape internally         │
    └───────────────────────────────┘
                │
                ▼
        Internal: (4, 8, 3, 256, 720)
                │
                ▼
    ┌───────────────────────────────┐
    │      RESNET-50 BACKBONE       │
    │  Process all 32 crops         │
    │  independently                │
    └───────────────────────────────┘
                │
        ┌───────┴───────┬───────┬───────┐
        ▼               ▼       ▼       ▼
    ┌─────┐         ┌─────┐ ┌─────┐ ┌─────┐
    │ P2  │         │ P3  │ │ P4  │ │ P5  │
    │256ch│         │512ch│ │1024 │ │2048 │
    └─────┘         └─────┘ └─────┘ └─────┘
    (32,256,        (32,512, (32,1024, (32,2048,
     64,180)         32,90)   16,45)    8,22)
        │               │       │       │
        └───────┬───────┴───────┴───────┘
                ▼
    ┌───────────────────────────────┐
    │      MVVIT TRANSFORMER        │
    │  Cross-View Attention         │
    │  (Applied to each FPN level)  │
    └───────────────────────────────┘
                │
        For each level L:
                │
                ▼
    ┌───────────────────────────────┐
    │  1. Reshape: (32,...) →       │
    │     (4, 8, C_L, H_L, W_L)     │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  2. Project to embed_dim=256  │
    │     if C_L ≠ 256              │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  3. Attention Pooling         │
    │     H_L×W_L → 512 tokens      │
    │     per view                  │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  4. Cross-View Transformer    │
    │     • Multi-head attention    │
    │     • Crop_i attends to       │
    │       Crop_{0-7} in group     │
    │     • 4 heads, 1 layer        │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  5. Attention Upsampling      │
    │     512 tokens → H_L×W_L      │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  6. Residual Connection       │
    │     refined = 0.5×refined +   │
    │               0.5×original    │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  7. Flatten back to           │
    │     (32, C_L, H_L, W_L)       │
    └───────────────────────────────┘
                │
                ▼
        Refined FPN levels with cross-view context:
        • P2': (32, 256, 64, 180)
        • P3': (32, 512, 32, 90)
        • P4': (32, 1024, 16, 45)
        • P5': (32, 2048, 8, 22)
```

---

## DIAGRAM 4: MVVIT ATTENTION MECHANISM (Moderate Mode)

```
┌────────────────────────────────────────────────────────────────────┐
│              MVVIT: MODERATE MODE (512 tokens/view)                │
└────────────────────────────────────────────────────────────────────┘

Input: List[8] of (B, C, H, W) features (one FPN level)
                │
                ▼
    ┌───────────────────────────────┐
    │  Stack: (B, 8, C, H, W)       │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  Project: C → 256 (embed_dim) │
    │  1×1 Conv if needed           │
    └───────────────────────────────┘
                │
                ▼
        (B, 8, 256, H, W)
                │
                ▼
    ┌───────────────────────────────┐
    │  STEP 1: Flatten Spatial      │
    │  (B×8, H×W, 256)              │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  STEP 2: Add Spatial Pos      │
    │  Learned: (H×W, 256)          │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────┐
    │  STEP 3: ATTENTION POOLING                │
    │  (Reduce spatial dimension)               │
    │                                           │
    │  K = 512 learnable queries                │
    │  Q: (B×8, 512, 256)                       │
    │  K, V: (B×8, H×W, 256)                    │
    │                                           │
    │  Attention(Q, K, V):                      │
    │    scores = QK^T / √256                   │
    │    weights = softmax(scores)              │
    │    output = weights @ V                   │
    │                                           │
    │  Output: (B×8, 512, 256)                  │
    │  → Each view now 512 tokens               │
    └───────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  STEP 4: Reshape for Cross-View │
    │  (B, 8×512, 256) = (B, 4096, 256) │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  STEP 5: Add View Positional  │
    │  Learned: (8, 256)            │
    │  Broadcast to all 512 tokens  │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────┐
    │  STEP 6: CROSS-VIEW TRANSFORMER           │
    │  (Multi-head self-attention)              │
    │                                           │
    │  Input: (B, 4096, 256)                    │
    │                                           │
    │  Multi-Head Attention (4 heads):          │
    │    Q = K = V = input                      │
    │    Attention weights: (B, 4, 4096, 4096)  │
    │    → Each token attends to ALL views      │
    │                                           │
    │  Feedforward Network:                     │
    │    Linear: 256 → 512                      │
    │    GELU activation                        │
    │    Linear: 512 → 256                      │
    │                                           │
    │  Residual + LayerNorm                     │
    │  Gradient Checkpointing enabled           │
    │                                           │
    │  Output: (B, 4096, 256)                   │
    │  → Features contain cross-view context    │
    └───────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  STEP 7: Reshape               │
    │  (B, 8, 512, 256)             │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────┐
    │  STEP 8: ATTENTION UPSAMPLING             │
    │  (Restore spatial dimension)              │
    │                                           │
    │  H×W learnable queries                    │
    │  Q: (B×8, H×W, 256)                       │
    │  K, V: (B×8, 512, 256)                    │
    │                                           │
    │  Attention(Q, K, V):                      │
    │    scores = QK^T / √256                   │
    │    weights = softmax(scores)              │
    │    output = weights @ V                   │
    │                                           │
    │  Output: (B×8, H×W, 256)                  │
    │  → Back to spatial resolution             │
    └───────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  STEP 9: Reshape + Residual   │
    │  (B, 8, 256, H, W)            │
    │  refined = 0.5×refined +      │
    │            0.5×original       │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  Project back: 256 → C if needed │
    └───────────────────────────────┘
                │
                ▼
Output: List[8] of (B, C, H, W) refined features
```

---

## DIAGRAM 5: DETECTION HEADS (RPN + ROI Head)

```
┌────────────────────────────────────────────────────────────────────┐
│                       DETECTION HEADS                              │
└────────────────────────────────────────────────────────────────────┘

Input: FPN features with cross-view context
    • P2': (32, 256, 64, 180)   [stride=4]
    • P3': (32, 256, 32, 90)    [stride=8]
    • P4': (32, 256, 16, 45)    [stride=16]
    • P5': (32, 256, 8, 22)     [stride=32]
    • P6': (32, 256, 4, 11)     [stride=64]

┌────────────────────────────────────────────────────────────────────┐
│                       RPN HEAD                                     │
└────────────────────────────────────────────────────────────────────┘

        All 5 FPN levels
                │
                ▼
    ┌───────────────────────────────┐
    │  3×3 Conv (256 → 256)         │
    └───────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
    ┌─────────┐   ┌─────────┐
    │ Cls Head│   │Bbox Head│
    │ 1×1 Conv│   │ 1×1 Conv│
    └─────────┘   └─────────┘
        │               │
        ▼               ▼
    Objectness      Delta (dx,dy,dw,dh)
    (32, 15, H, W)  (32, 60, H, W)
        │               │
        └───────┬───────┘
                ▼
    ┌───────────────────────────────┐
    │  Generate Anchors             │
    │  • Scales: [4, 8, 16]         │
    │  • Ratios: [0.2,0.35,0.5,1,2] │
    │  • 15 anchors per location    │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  RPN Loss (if training)       │
    │  • Focal Loss (objectness)    │
    │  • L1 Loss (bbox regression)  │
    └───────────────────────────────┘
                │
                ▼
        ~1000 proposals per image
        Shape: (32, ~1000, 4)

┌────────────────────────────────────────────────────────────────────┐
│                       ROI HEAD                                     │
└────────────────────────────────────────────────────────────────────┘

        Proposals + FPN features
                │
                ▼
    ┌───────────────────────────────┐
    │  RoI Align (7×7)              │
    │  • Pool from appropriate FPN  │
    │    level based on proposal    │
    │    size                       │
    └───────────────────────────────┘
                │
                ▼
        (N_proposals, 256, 7, 7)
                │
                ▼
    ┌───────────────────────────────┐
    │  Flatten + FC Layers          │
    │  • FC1: 256×7×7 → 1024        │
    │  • ReLU                       │
    │  • FC2: 1024 → 1024           │
    │  • ReLU                       │
    └───────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
    ┌─────────┐   ┌─────────┐
    │ Cls Head│   │Bbox Head│
    │FC:1024→5│   │FC:1024→4│
    └─────────┘   └─────────┘
        │               │
        ▼               ▼
    Class scores    Bbox deltas
    (N, 5)          (N, 4)
        │               │
        │               │
    (Training: Compute losses)
        │
        ▼
    ┌───────────────────────────────┐
    │  RCNN Loss                    │
    │  • Focal Loss (classification)│
    │  • Smooth L1 (bbox regression)│
    └───────────────────────────────┘
        │
        │
    (Inference: Post-processing)
        │
        ▼
    ┌───────────────────────────────┐
    │  NMS (IoU threshold = 0.5)    │
    │  Score threshold = 0.05       │
    │  Max 100 boxes per image      │
    └───────────────────────────────┘
                │
                ▼
        Final predictions per crop
        • Bboxes: (N, 4)
        • Scores: (N,)
        • Labels: (N,)
```

---

## DIAGRAM 6: LOSS COMPUTATION (Supervised + Unsupervised)

```
┌────────────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION FLOW                           │
└────────────────────────────────────────────────────────────────────┘

Training Batch: 4 groups × 8 crops = 32 images
        │
        ├─────────────────────┬─────────────────────┐
        ▼                     ▼                     ▼
    Labeled (60%)       Unlabeled (40%)       Unlabeled (40%)
    ~19 images          ~13 images            ~13 images
    with GT             no GT                 no GT
        │                     │                     │
        │              ┌──────┴──────┐              │
        │              ▼             ▼              │
        │         TEACHER        TEACHER            │
        │      (weak aug)      (weak aug)           │
        │      Extract feat    Generate             │
        │                      pseudo-labels         │
        │                           │                │
        │              ┌────────────┘                │
        │              ▼                             │
        │      ┌─────────────────┐                  │
        │      │ Pseudo Filtering│                  │
        │      │ • Score > 0.7   │                  │
        │      │ • Max 15 boxes  │                  │
        │      │ • Uncertainty   │                  │
        │      │   < 0.02        │                  │
        │      └─────────────────┘                  │
        │              │                             │
        │              └─────────────────────────────┘
        │                                            │
        ▼                                            ▼
    ┌─────────────────┐                    ┌─────────────────┐
    │   STUDENT       │                    │   STUDENT       │
    │ (supervised)    │                    │ (unsupervised)  │
    │ Strong aug      │                    │ Strong aug      │
    └─────────────────┘                    └─────────────────┘
        │                                            │
        ▼                                            ▼
    ┌─────────────────┐                    ┌─────────────────┐
    │ Extract Features│                    │ Extract Features│
    │ with MVViT      │                    │ with MVViT      │
    └─────────────────┘                    └─────────────────┘
        │                                            │
        ▼                                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │         DETECTION HEADS (RPN + ROI Head)                │
    │         Predict on enriched features                    │
    └─────────────────────────────────────────────────────────┘
        │                                            │
        ▼                                            ▼
    ┌─────────────────┐                    ┌─────────────────┐
    │  Compute Losses │                    │  Compute Losses │
    │  vs GT labels   │                    │  vs Pseudo labels│
    └─────────────────┘                    └─────────────────┘
        │                                            │
        ▼                                            ▼
    ┌─────────────────┐                    ┌─────────────────┐
    │ RPN Loss:       │                    │ RPN Loss:       │
    │ • Focal (cls)   │                    │ • Focal (cls)   │
    │ • L1 (bbox)     │                    │ • L1 (bbox)     │
    │                 │                    │ (filtered > 0.5)│
    ├─────────────────┤                    ├─────────────────┤
    │ RCNN Loss:      │                    │ RCNN Loss:      │
    │ • Focal (cls)   │                    │ • Focal (cls)   │
    │ • Smooth L1     │                    │   + soft BG     │
    │   (bbox)        │                    │ • Smooth L1     │
    │                 │                    │   (filtered)    │
    └─────────────────┘                    └─────────────────┘
        │                                            │
        │                                            │
        ▼                                            ▼
    sup_loss × 1.0                        unsup_loss × 0.5
        │                                            │
        └────────────────┬───────────────────────────┘
                         ▼
                ┌─────────────────┐
                │   TOTAL LOSS    │
                │                 │
                │ loss = sup_loss │
                │        + unsup  │
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  BACKPROPAGATION│
                │                 │
                │ Student: SGD    │
                │ lr = 0.001      │
                │                 │
                │ Teacher: EMA    │
                │ momentum=0.999  │
                └─────────────────┘
```

---

## DIAGRAM 7: MULTI-VIEW LOSS COMPUTATION (Key Innovation)

```
┌────────────────────────────────────────────────────────────────────┐
│          HOW MULTI-VIEW LEARNING WORKS IN LOSS                     │
└────────────────────────────────────────────────────────────────────┘

Batch: 4 groups × 8 crops = 32 images

┌────────────────────────────────────────────────────────────────────┐
│  STEP 1: FEATURE EXTRACTION WITH CROSS-VIEW ATTENTION              │
└────────────────────────────────────────────────────────────────────┘

Input: (32, 3, 256, 720)
    │
    ▼
MultiViewBackbone:
    │
    ├─ Infer: B=4, V=8
    │
    ├─ ResNet-50: Extract per-crop features
    │   → (32, 256, H', W') per FPN level
    │
    └─ MVViT: Cross-view attention
        │
        For each group:
        ┌──────────────────────────────────┐
        │  Group 0: [Crop_0 ... Crop_7]   │
        │                                  │
        │  Crop_0 features attend to:     │
        │  • Crop_1, Crop_2, ..., Crop_7  │
        │  (via transformer attention)    │
        │                                  │
        │  Learning: Which other views     │
        │  provide useful context?         │
        └──────────────────────────────────┘
    │
    ▼
Features: (32, 256, H', W')
→ Each crop's features contain info from all 8 views in its group!

┌────────────────────────────────────────────────────────────────────┐
│  STEP 2: PER-CROP LOSS COMPUTATION                                 │
└────────────────────────────────────────────────────────────────────┘

For each of 32 crops INDEPENDENTLY:
    │
    ├─ RPN predicts on enriched features
    │   • Objectness: Is there an object?
    │   • Bbox: Where is it?
    │
    ├─ ROI Head refines on enriched features
    │   • Classification: What class?
    │   • Bbox regression: Exact location?
    │
    └─ Compute loss vs THIS crop's GT only
        │
        Example for Crop_0:
        ┌──────────────────────────────────┐
        │  Features: f_0 (has cross-view  │
        │             context from MVViT)  │
        │  GT: Only Crop_0's annotations   │
        │                                  │
        │  loss_0 = RPN_loss(f_0, gt_0) + │
        │           ROI_loss(f_0, gt_0)   │
        └──────────────────────────────────┘

Repeat for all 32 crops → get 32 losses

┌────────────────────────────────────────────────────────────────────┐
│  STEP 3: LOSS AGGREGATION (Standard PyTorch Mean)                  │
└────────────────────────────────────────────────────────────────────┘

    loss_total = mean(loss_0, loss_1, ..., loss_31)
               = (1/32) × Σ(all 32 crops)
               = (1/4) × Σ_groups[(1/8) × Σ_crops]

    ← No special multi-view weighting needed!
    ← Group structure preserved through MVViT connections

┌────────────────────────────────────────────────────────────────────┐
│  STEP 4: GRADIENT FLOW (Where Multi-View Learning Happens!)        │
└────────────────────────────────────────────────────────────────────┘

Backpropagation:

    ∂loss_total/∂θ = Σ(∂loss_i/∂θ) for i=0..31

For MVViT parameters:

    ∂loss_0/∂MVViT ≠ 0  (direct, through f_0)
    ∂loss_1/∂MVViT ≠ 0  (direct, through f_1)
    ...
    ∂loss_7/∂MVViT ≠ 0  (direct, through f_7)
    
    ← All 8 crops in Group 0 contribute gradient to MVViT!

For crop features via MVViT:

    ∂loss_0/∂f_1 ≠ 0  ← Crop_0's loss affects Crop_1's features!
    ∂loss_0/∂f_2 ≠ 0
    ...
    ∂loss_0/∂f_7 ≠ 0
    
    ← MVViT attention creates cross-crop dependencies

Result:
    • Model learns to use complementary views
    • Gradient encourages helpful cross-view patterns
    • Loss on one crop improves features for all crops in group

┌────────────────────────────────────────────────────────────────────┐
│  KEY INSIGHT                                                        │
└────────────────────────────────────────────────────────────────────┘

    Multi-view learning does NOT happen in loss formula!
    (Still just mean over 32 crops)
    
    It happens in FEATURES via MVViT attention:
    • Features depend on multiple views
    • Gradient flows through MVViT to all views
    • Model optimizes for COLLECTIVE performance
```

---

## DIAGRAM 8: EVALUATION & AGGREGATION

```
┌────────────────────────────────────────────────────────────────────┐
│              EVALUATION: 8 CROPS → 1 BASE IMAGE                    │
└────────────────────────────────────────────────────────────────────┘

Inference on validation set:

Base Image: S1_bright_0 (8 crops)
    │
    ├─ Crop_0: [(bbox_0_0, score_0_0, class_0_0), ...]
    ├─ Crop_1: [(bbox_1_0, score_1_0, class_1_0), ...]
    ├─ Crop_2: [(bbox_2_0, score_2_0, class_2_0), ...]
    ├─ Crop_3: [(bbox_3_0, score_3_0, class_3_0), ...]
    ├─ Crop_4: [(bbox_4_0, score_4_0, class_4_0), ...]
    ├─ Crop_5: [(bbox_5_0, score_5_0, class_5_0), ...]
    ├─ Crop_6: [(bbox_6_0, score_6_0, class_6_0), ...]
    └─ Crop_7: [(bbox_7_0, score_7_0, class_7_0), ...]
        │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 1: Transform to Base Image Coordinates                       │
└────────────────────────────────────────────────────────────────────┘

    For each bbox in each crop:
        • Apply inverse crop transform
        • Convert local coords → base image coords
        
    All boxes now in same coordinate space!

┌────────────────────────────────────────────────────────────────────┐
│  STEP 2: Weighted Box Fusion (WBF)                                 │
└────────────────────────────────────────────────────────────────────┘

    Input: All boxes from 8 crops (50-200 total boxes)
    
    Algorithm:
    ┌────────────────────────────────────┐
    │ 1. Sort by confidence descending   │
    └────────────────────────────────────┘
            │
            ▼
    ┌────────────────────────────────────┐
    │ 2. For each box B:                 │
    │    • Find overlapping boxes        │
    │      (IoU > 0.5)                   │
    │    • Form cluster C                │
    └────────────────────────────────────┘
            │
            ▼
    ┌────────────────────────────────────┐
    │ 3. Fuse cluster C:                 │
    │                                    │
    │    Coordinates:                    │
    │    x_fused = Σ(x_i × score_i) /   │
    │              Σ(score_i)            │
    │                                    │
    │    Score:                          │
    │    score_fused = mean(scores)     │
    │                                    │
    │    Class:                          │
    │    class_fused = majority vote     │
    └────────────────────────────────────┘
            │
            ▼
    ┌────────────────────────────────────┐
    │ 4. Remove boxes in cluster         │
    │    Keep fused box                  │
    └────────────────────────────────────┘
            │
            ▼
    Repeat until all boxes processed

┌────────────────────────────────────────────────────────────────────┐
│  STEP 3: Post-Processing                                           │
└────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────┐
    │ • Remove low confidence boxes      │
    │   (score < 0.05)                   │
    │                                    │
    │ • Apply NMS within same class      │
    │   (IoU threshold = 0.5)            │
    │                                    │
    │ • Keep max 100 boxes               │
    └────────────────────────────────────┘
            │
            ▼
    Final predictions for base image:
    • Bboxes: (N, 4)
    • Scores: (N,)
    • Labels: (N,)

┌────────────────────────────────────────────────────────────────────┐
│  STEP 4: COCO Evaluation                                           │
└────────────────────────────────────────────────────────────────────┘

    Compare with ground truth annotations:
    
    ┌────────────────────────────────────┐
    │ • AP @ IoU=0.50:0.05:0.95          │
    │ • AP @ IoU=0.50 (AP50)             │
    │ • AP @ IoU=0.75 (AP75)             │
    │                                    │
    │ • AP for small objects             │
    │ • AP for medium objects            │
    │ • AP for large objects             │
    │                                    │
    │ • Per-class AP:                    │
    │   - Broken                         │
    │   - Chipped                        │
    │   - Scratched                      │
    │   - Severe_Rust                    │
    │   - Tip_Wear                       │
    └────────────────────────────────────┘

Advantages of WBF over NMS:
✓ Uses ALL 8 views (NMS keeps only highest)
✓ More robust to occlusion (info from multiple views)
✓ Better localization (averaged coordinates)
✓ Higher recall (captures objects seen in multiple crops)
```

---

## DIAGRAM 9: SOFT TEACHER → MULTI-VIEW (Summary of Changes)

```
┌────────────────────────────────────────────────────────────────────┐
│         FROM SOFT TEACHER TO MULTI-VIEW SOFT TEACHER               │
└────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┬─────────────────────────────────────────┐
│    Soft Teacher (Gốc)    │    Multi-View Soft Teacher (Ours)       │
├──────────────────────────┼─────────────────────────────────────────┤
│                          │                                         │
│  INPUT:                  │  INPUT:                                 │
│  • 1 image per sample    │  • 8 crops per sample (1 group)        │
│  • (B, C, H, W)          │  • (B, 8, C, H, W) → flatten to (B×8,) │
│                          │                                         │
├──────────────────────────┼─────────────────────────────────────────┤
│                          │                                         │
│  DATASET:                │  DATASET:                               │
│  • CocoDataset           │  • MultiViewFromFolder [NEW]            │
│  • Standard annotations  │  • Grouped annotations (base_img_id)    │
│                          │                                         │
├──────────────────────────┼─────────────────────────────────────────┤
│                          │                                         │
│  COLLATE:                │  COLLATE:                               │
│  • pseudo_collate        │  • multi_view_collate_flatten [NEW]     │
│  • Output: (B, C, H, W)  │  • Flatten: (B,8,C,H,W)→(B×8,C,H,W)   │
│                          │  • Multi-branch: sup/unsup_T/unsup_S    │
│                          │                                         │
├──────────────────────────┼─────────────────────────────────────────┤
│                          │                                         │
│  BACKBONE:               │  BACKBONE:                              │
│  • ResNet-50             │  • MultiViewBackbone [NEW]              │
│  • Independent features  │    └─ ResNet-50 (same)                  │
│  • (B, 256, H', W')      │    └─ MVViT Transformer [NEW]           │
│                          │       • Cross-view attention            │
│                          │       • 4 heads, 1 layer, 512 tokens    │
│                          │  • Features with cross-view context     │
│                          │  • (B×8, 256, H', W')                   │
│                          │                                         │
├──────────────────────────┼─────────────────────────────────────────┤
│                          │                                         │
│  DETECTOR:               │  DETECTOR:                              │
│  • SoftTeacher           │  • MultiViewSoftTeacher [NEW]           │
│  • Standard loss         │  • Same loss computation                │
│                          │  • BUT: features have cross-view info   │
│                          │  • views_per_sample parameter           │
│                          │                                         │
├──────────────────────────┼─────────────────────────────────────────┤
│                          │                                         │
│  LOSS:                   │  LOSS:                                  │
│  • Mean over B images    │  • Mean over B×8 crops                  │
│  • No cross-image deps   │  • Cross-crop deps via MVViT gradient   │
│                          │                                         │
├──────────────────────────┼─────────────────────────────────────────┤
│                          │                                         │
│  EVALUATION:             │  EVALUATION:                            │
│  • CocoMetric            │  • MultiViewCocoMetric [NEW]            │
│  • 1 prediction per img  │  • 8 predictions → 1 (WBF)             │
│  • Direct COCO eval      │  • Aggregate then COCO eval             │
│                          │                                         │
└──────────────────────────┴─────────────────────────────────────────┘

KEY ADDITIONS:
[NEW] = New module needed
• 5 new Python files
• Config changes in 5 places
• Annotation format: add base_img_id field
```

---

## DIAGRAM 10: TEACHER-STUDENT FRAMEWORK

```
┌────────────────────────────────────────────────────────────────────┐
│             TEACHER-STUDENT SEMI-SUPERVISED LEARNING               │
└────────────────────────────────────────────────────────────────────┘

INITIALIZATION (Iter 0):
    ┌─────────────────┐
    │  Student Model  │  ─copy─►  ┌─────────────────┐
    │  (ResNet+MVViT) │           │  Teacher Model  │
    └─────────────────┘           │  (Same arch)    │
                                   └─────────────────┘

TRAINING LOOP (Each iteration):

    ┌────────────────────────────────────────────────────────────────┐
    │  LABELED DATA (Batch: ~2.4 groups × 8 = ~19 images)           │
    └────────────────────────────────────────────────────────────────┘
            │
            │ Strong Augmentation:
            │ • RandAugment (color)
            │ • RandAugment (geometric)
            │ • RandomErasing
            ▼
    ┌─────────────────┐
    │  Student Model  │
    │  Forward + Loss │
    └─────────────────┘
            │
            ▼
    Supervised Loss vs Ground Truth:
    • RPN Loss (Focal + L1)
    • RCNN Loss (Focal + Smooth L1)
    • Weight: 1.0
    
    
    ┌────────────────────────────────────────────────────────────────┐
    │  UNLABELED DATA (Batch: ~1.6 groups × 8 = ~13 images)         │
    └────────────────────────────────────────────────────────────────┘
            │
            ├─────────────────────┬─────────────────────┐
            │                     │                     │
            │ Weak Aug            │ Strong Aug          │
            │ (flip only)         │ (full pipeline)     │
            ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │  Teacher Model  │   │  Teacher Model  │   │  Student Model  │
    │  (frozen)       │   │  (frozen)       │   │  (trainable)    │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
            │                     │                     │
            │                     │                     │
            ▼                     ▼                     ▼
    Extract features    Generate predictions    Learn from
    for uncertainty     (pseudo-labels)         pseudo-labels
            │                     │                     │
            │                     ▼                     │
            │             ┌───────────────┐             │
            │             │ FILTER        │             │
            │             │ • Score > 0.7 │             │
            │             │ • Max 15 boxes│             │
            │             └───────────────┘             │
            │                     │                     │
            └─────────┬───────────┴─────────┬───────────┘
                      │                     │
                      ▼                     ▼
              Compute uncertainty   Unsupervised Loss:
              (jitter 5×)           • RPN Loss (filtered > 0.5)
                      │             • RCNN Cls (filtered > 0.6)
                      │               + Soft BG scores
                      │             • RCNN Reg (unc < 0.02)
                      │             • Weight: 0.5
                      │                     │
                      └─────────┬───────────┘
                                │
                                ▼
                        Filter pseudo-labels
                        by uncertainty
    
    
    ┌────────────────────────────────────────────────────────────────┐
    │  TOTAL LOSS = Supervised Loss + Unsupervised Loss              │
    └────────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────┐
    │  BACKPROPAGATION│
    └─────────────────┘
            │
            ├─────────────────────┬─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
    Update Student      Update Teacher      Update MVViT
    (SGD, lr=0.001)     (EMA, m=0.999)      (Shared params)
            │                     │                     │
            │                     │                     │
            ▼                     ▼                     ▼
    θ_s ← θ_s - α∇L     θ_t ← 0.999θ_t      θ_mvvit learns
                              + 0.001θ_s      cross-view patterns
    
    
    ┌────────────────────────────────────────────────────────────────┐
    │  EMA UPDATE (Exponential Moving Average)                       │
    └────────────────────────────────────────────────────────────────┘
    
    For each parameter:
        teacher_param = 0.999 × teacher_param + 0.001 × student_param
    
    Effect:
    • Teacher is stable (changes slowly)
    • Generates consistent pseudo-labels
    • Reduces noise in unsupervised learning
    
    
    ┌────────────────────────────────────────────────────────────────┐
    │  KEY HYPERPARAMETERS                                           │
    └────────────────────────────────────────────────────────────────┘
    
    • EMA momentum: 0.999 (teacher stability)
    • Pseudo threshold: 0.7 (high confidence only)
    • Unsup weight: 0.5 (balance sup/unsup)
    • RPN pseudo thr: 0.5 (moderate filtering)
    • CLS pseudo thr: 0.6 (stricter filtering)
    • REG uncertainty: 0.02 (bbox quality)
    • Jitter times: 5 (uncertainty estimation)
    • Max boxes: 15 (avoid overgeneration)
```

---

## NOTES FOR DRAWING

### Tool Recommendations:
1. **draw.io / diagrams.net** - Free, web-based
2. **Lucidchart** - Professional
3. **Microsoft Visio** - Standard tool
4. **PlantUML** - Code-based diagrams
5. **Mermaid** - Markdown-based

### Color Scheme Suggestions:
- **Input/Data**: Light Blue (#E3F2FD)
- **Processing**: Light Green (#E8F5E9)
- **Loss/Training**: Light Orange (#FFE0B2)
- **Output/Results**: Light Purple (#F3E5F5)
- **Attention/Important**: Light Red (#FFCDD2)

### Arrow Types:
- **Data flow**: Solid arrow →
- **Gradient flow**: Dashed arrow ⇢
- **Attention**: Double arrow ⇄
- **Copy/Clone**: Dotted arrow ···►

### Box Styles:
- **Main components**: Rounded rectangle
- **Subcomponents**: Rectangle
- **Processes**: Oval/Ellipse
- **Decisions**: Diamond
- **Data**: Parallelogram
