_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', 
    '../_base_/default_runtime.py',
]
# Ensure mmengine/mmcv imports our custom MultiView module so the registry can find
# `MultiViewBackbone`. This avoids KeyError when building model from cfg.
# Import the collate callable directly so MMEngine receives a callable for
# `collate_fn` (it does not accept string import paths for callables).

# Only import modules (not individual callables) via custom_imports so the
# registry decorators run when MMEngine loads the config. Keep module paths
# here; the collate callable is imported above and used directly.
custom_imports = dict(imports=[
    'mmdet.models.utils.multi_view',
    'mmdet.models.utils.multi_view_transformer',
    'mmdet.datasets.wrappers.multi_view_from_folder'
], allow_failed_imports=False)
backend_args = None
metainfo = {
    'classes': ('Broken','Chipped','Scratched','Severe_Rust','Tip_Wear'),
    'palette': [
        (134, 34, 255),
        (0, 255, 206),
        (255, 128, 0),
        (254, 0, 86),
        (199, 252, 0)
    ]
}

# Multi-view configuration (optimized for 32GB GPU)
views_per_sample = 8
batch_size = 2
num_workers = 4  # Reduced from 4 to save memory

# model settings
detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],  # Caffe/BGR
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=8
)
detector.backbone = dict(
    type='MultiViewBackbone',
    # inner backbone config is the original ResNet
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    fusion='mvvit',  # Use MVViT for multi-view feature fusion
    concat_reduce=False,
    views_per_sample=8,  # Number of views per sample for reshaping flattened inputs
    # MVViT configuration for cross-view attention (optimized for 32GB GPU)
    # Memory-performance trade-off options:
    # 
    # Option 1: 'efficient' mode (memory-constrained)
    #   - spatial_attention='efficient' (256 tokens/view)
    #   - Memory: ~4-8GB, Speed: Fast, Accuracy: Good
    # 
    # Option 2: 'moderate' mode (RECOMMENDED for 32GB GPU)
    #   - spatial_attention='moderate' (512 tokens/view)
    #   - Memory: ~8-16GB, Speed: Medium, Accuracy: Better
    #   - Best balance between performance and memory
    # 
    # Option 3: Disable MVViT (baseline comparison)
    #   - Change fusion='mean' and remove mvvit config
    #   - Memory: ~1GB saved, Accuracy: Lower (no cross-view learning)
    # 
    # UPGRADED MVViT FOR HARD CLASSES (Severe_Rust: 0.006 mAP, Chipped: 0.032 mAP)
    # Analysis: These classes have MOST data (32.8%, 18.2%) but WORST performance
    # Root cause: Very subtle visual features that require strong multi-view fusion
    # 
    # BALANCED CONFIG (Option B) - Memory: ~10-16GB, Best trade-off:
    # - num_layers=2: CRITICAL for complex cross-view reasoning (was 1)
    #   * Severe_Rust: Aggregate rust evidence across views with different lighting
    #   * Chipped: Combine chip visibility from multiple angles
    # - num_heads=8: Standard for ViT, diverse attention patterns (was 4)
    #   * More heads → capture different spatial relationships per head
    #   * Better for narrow aspect ratios (median 0.44, 0.51)
    # - mlp_ratio=3.0: Increased capacity for feature transformation (was 2.0)
    #   * More complex patterns → better distinguish hard classes
    # - spatial_attention='moderate': Keep K=2048 tokens (good balance)
    # - use_gradient_checkpointing=True: ~50% memory saving in backward pass
    mvvit=dict(
        type='MVViT',
        embed_dim=256,       # Match FPN channels
        num_heads=4,         # INCREASED from 4 → more diverse attention for hard classes
        num_layers=1,        # INCREASED from 1 → deeper cross-view reasoning (CRITICAL!)
        mlp_ratio=2.0,       # INCREASED from 2.0 → more capacity for subtle features
        dropout=0.2,         # Dropout rate (unchanged)
        use_layer_norm=True, # Use LayerNorm instead of BatchNorm
        spatial_attention='moderate',  # Pool to 2048 tokens BEFORE transformer (saves FLOPs!)
        use_gradient_checkpointing=True  # Enable gradient checkpointing for memory efficiency
    )
)
# Override RPN and ROI head to handle class imbalance
detector.rpn_head = dict(
    type='RPNHead',
    in_channels=256,
    feat_channels=256,
    anchor_generator=dict(
        type='AnchorGenerator',
        scales=[8, 16, 32, 64],  # Multiple scales for objects of different sizes
        # TARGETED RATIOS for Severe_Rust (med=0.45, P25-P75: 0.34-0.61) and Chipped (med=0.51, P25-P75: 0.39-0.65)
        # Analysis shows 60% of Severe_Rust and 48% of Chipped have ratio < 0.5 (narrow boxes)
        ratios=[0.25, 0.5, 1.0, 2.0], 
        strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(
        type='FocalLoss',  # CHANGED from CrossEntropyLoss → focus on hard examples
        use_sigmoid=True,
        gamma=2.0,         # REDUCED from 2.5 → standard ViT/Focal Loss value
        alpha=0.65,         # REDUCED from 0.75 → more balanced FG:BG weighting
        loss_weight=2.0),  # REDUCED from 2.5 → balance with bbox loss
    loss_bbox=dict(type='L1Loss', loss_weight=1.0))

# Override ROI head to use Focal Loss (combat class imbalance)
# NOTE: FocalLoss doesn't support class_weight parameter
# Strategy: VERY HIGH gamma (3.5) focuses HEAVILY on hard examples (Chipped, Severe_Rust)
#           High alpha (0.75) weights all FG classes more than BG
# ANALYSIS: Severe_Rust (0.006 mAP) and Chipped (0.032 mAP) are VERY hard to learn
#           despite having 27.6% and 19.2% of data → need extreme hard example mining!
detector.roi_head.bbox_head.loss_cls = dict(
    type='FocalLoss',     # CHANGED from CrossEntropyLoss
    use_sigmoid=True,     # MUST be True (mmdet implementation requirement)
    gamma=2.0,            # REDUCED from 3.0 → less extreme, more stable
    alpha=0.65,            # REDUCED from 0.75 → more balanced FG:BG weighting
    loss_weight=1.5)      # REDUCED from 2.5 → better balance with bbox loss

# Override train_cfg to use more relaxed IoU thresholds for narrow objects
detector.train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,      # Reduced from 0.7 for narrow objects
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,              # REDUCED from 512 → fewer samples for sparse data
            pos_fraction=0.75,    # INCREASED from 0.7 → VERY strongly prefer FG (75:25)
            neg_pos_ub=1.0,       # REDUCED from 1.5 → max 1.0 BG per FG (1:1 ratio)
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=2000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,              # Match RPN
            pos_fraction=0.75,    # INCREASED from 0.7 → VERY strongly prefer FG (75:25)
            neg_pos_ub=1.0,       # REDUCED from 1.5 → max 1.0 BG per FG (1:1 ratio)
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))

# Test config for RPN: balanced settings to reduce false positives
detector.test_cfg.rpn = dict(
    nms_pre=2000,           # REDUCED from 4000 → faster inference
    max_per_img=1000,       # REDUCED from 2000 → fewer low-quality proposals
    nms=dict(type='nms', iou_threshold=0.5),
    min_bbox_size=8)        # Keep small objects (drill bits can be narrow)

# Test config for RCNN: LOWERED threshold to detect more hard classes (Severe_Rust mAP=3.3%)
detector.test_cfg.rcnn = dict(
    score_thr=0.25,  # LOWERED from 0.5 → allow lower confidence detections for hard classes
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=10)  # REDUCED from 20 → 1.7× GT max (sufficient buffer)


model = dict(
    _delete_=True,
    type='MultiViewSoftTeacher',  # Use MultiViewSoftTeacher instead of SoftTeacher
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True, 
        sup_weight=1.0, 
        unsup_weight=1.0,   # Balance supervised vs unsupervised learning
        # AGGRESSIVE THRESHOLDS for hard class learning (Severe_Rust: 0.006 mAP, Chipped: 0.032 mAP)
        # Analysis: Despite having 27.6% and 19.2% of data, these classes have VERY low performance
        # → Problem is NOT data quantity but QUALITY/DIFFICULTY of examples
        # Solution: LOWER thresholds aggressively to allow teacher to generate more pseudo-labels
        #           even for uncertain predictions → help student learn from harder examples
        # 
        # Threshold hierarchy (progressively relaxed for better data utilization):
        # 1. pseudo_label_initial_score_thr=0.3: LOWERED to allow more hard class detections (Severe_Rust)
        # 2. cls_pseudo_thr=0.25: LOWERED aggressively for hard classes with low confidence
        # 3. rpn_pseudo_thr=0.3: LOWERED to generate more FG proposals
        # Rationale: Severe_Rust mAP=3.3% → need MORE training signal even from lower confidence predictions
        #            Cross-view uncertainty provides quality control → can afford lower thresholds
        pseudo_label_initial_score_thr=0.5,  # LOWERED from 0.4 → allow hard class detections
        cls_pseudo_thr=0.4,                  # LOWERED from 0.35 → more pseudo-labels for Severe_Rust
        rpn_pseudo_thr=0.4,                   # LOWERED from 0.35 → more FG proposals
        reg_pseudo_thr=0.095,                 # Keep unchanged (jitter-based uncertainty)
        jitter_times=5,                      # Augmentation for uncertainty estimation
        jitter_scale=0.03,                   # Scale of jitter augmentation
        min_pseudo_bbox_wh=(8, 8),          # Filter small noisy boxes
        # Multi-view consensus: CROSS-VIEW UNCERTAINTY approach
        # For rotation views (0°, 45°, 90°, etc.): Can't use IoU-based matching
        # Strategy: Measure CLASS-LEVEL agreement across views
        # - If class predicted by multiple views → LOW uncertainty → higher weight
        # - If class only in 1 view → HIGH uncertainty → lower weight
        # This increases pseudo-label quality without requiring spatial alignment
        enable_consensus=True),              # Enable cross-view uncertainty weighting
    semi_test_cfg=dict(predict_on='teacher'),
    views_per_sample=views_per_sample,  
    aggregate_views='mean'   # How to aggregate multi-view losses
)   

dataset_type = 'CocoDataset'
data_root = '/home/coder/data/trong/KLTN/Soft_Teacher/data_drill/'

# TARGETED AUGMENTATION for hard classes (Severe_Rust: 0.006 mAP, Chipped: 0.032 mAP)
# Analysis: These have 27.6% and 19.2% of data but terrible performance
# → Visual features are VERY HARD to distinguish, need aggressive augmentation
# Severe_Rust: Rust patterns vary HEAVILY with lighting/contrast
# Chipped: Small cracks/chips need sharpness variation to be visible
color_space = [
    [dict(type='AutoContrast', prob=0.5)],  # Giảm từ 0.6
    [dict(type='Equalize',    prob=0.4)],   # Giảm từ 0.5
    # MUTUAL EXCLUSIVE: Brightness OR Contrast (not both!)
    [dict(type='Brightness',  prob=0.6, level=7, min_mag=0.8, max_mag=1.25)],  # Giảm prob 0.8→0.6, range moderate
    [dict(type='Contrast',    prob=0.6, level=7, min_mag=0.8, max_mag=1.3)],   # Giảm prob 0.8→0.6, range moderate
    [dict(type='Sharpness',   prob=0.4, level=6, min_mag=0.8, max_mag=1.5)],   # Giảm range extreme (2.0→1.5)
    [dict(type='Posterize',   prob=0.2, level=5, min_mag=5.0, max_mag=7.0)],   # Giảm prob, tăng bits (less extreme)
]
geometric = [
    # REMOVED: Rotate, ShearX, ShearY
    # Reason: Conflict with radial crop pattern + MVViT cross-view attention
    
    # KEEP: Translation (safe for multi-view)
    [dict(type='TranslateX', prob=0.3, level=7, min_mag=0.0, max_mag=0.05)],
    [dict(type='TranslateY', prob=0.3, level=7, min_mag=0.0, max_mag=0.05)],
]

# Single scale for resize operations (width, height)
# Note: Use single scale for simplicity. For multi-scale training, use RandomResize instead
img_scale = (256, 720)

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=False,          # giữ uint8 → RAM giảm ~4x
        imdecode_backend='cv2',
        backend_args=backend_args
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(4, 4)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
# STRONG AUGMENTATION: aug_num=2 for color (combine multiple color transforms)
# This forces student to learn robust features invariant to appearance changes
strong_pipeline = [
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=2),  # INCREASED 1→2 for stronger augmentation
            dict(type='RandAugment', aug_space=geometric, aug_num=1),    # Keep 1 (only 2 ops: TranslateX/Y)
        ]),
    dict(type='RandomErasing', n_patches=(1, 3), ratio=(0, 0.15)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(4, 4)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=False,          # giữ uint8 → RAM giảm ~4x
        imdecode_backend='cv2',
        backend_args=backend_args
    ),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=False,          # giữ uint8 → RAM giảm ~4x
        imdecode_backend='cv2',
        backend_args=backend_args
    ),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Configure semi-supervised dataloader

# Use MultiViewFromFolder for multi-view samples (8 crops per sample)
# Using 60% labeled split (298 groups × 8 = 2384 images, 2423 annotations)
labeled_dataset = dict(
    type='MultiViewFromFolder',
    data_root=data_root,
    views_per_sample=views_per_sample,
    ann_file=data_root + 'semi_anno_multiview/_annotations.coco.labeled.grouped@40.bright.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=4),
    pipeline=sup_pipeline,
    metainfo=metainfo,
    backend_args=backend_args
)

# Using 40% unlabeled split (199 groups × 8 = 1592 images, 1702 annotations)
unlabeled_dataset = dict(
    type='MultiViewFromFolder',
    data_root=data_root,
    views_per_sample=views_per_sample,
    ann_file=data_root + 'semi_anno_multiview/_annotations.coco.unlabeled.grouped@40.bright.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=4),
    pipeline=unsup_pipeline,
    metainfo=metainfo,
    backend_args=backend_args
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,  # Keep workers alive, reduce overhead
    pin_memory=True,
    prefetch_factor=2,  # Reduced from 4 to save memory
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        # source_ratio=[1, 1] means 1 labeled + 1 unlabeled per batch
        # Note: with batch_size=2, ratio [1, 3] would compute as:
        #   labeled: int(2*1/4)=0 → adjusted to 1, unlabeled: int(2*3/4)=1
        #   → actual ratio is [1, 1], not [1, 3]!
        # For true [1, 3] ratio, need batch_size=4 (1 labeled + 3 unlabeled)
        source_ratio=[1, 1]),  # 1:1 ratio for batch_size=2 (labeled:unlabeled)
    collate_fn=dict(type='mmdet.datasets.wrappers.multi_view_from_folder.multi_view_collate_flatten'),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))


# training 
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100000, val_interval=10000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# Explicit validation dataloader/evaluator for multi-view validation
# Note: validation uses multi_view_collate_val to flatten views without branch separation
val_dataloader = dict(
    batch_size=1,
    num_workers=2,  # Reduced from num_workers to save memory during validation
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='mmdet.datasets.wrappers.multi_view_from_folder.multi_view_collate_val'),
    dataset=dict(
        type='MultiViewFromFolder',
        data_root=data_root ,
        views_per_sample=views_per_sample,
        ann_file=data_root + 'anno_valid/_annotations_filtered.bright.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
        backend_args=backend_args
    ))

val_evaluator = dict(
    type='MultiViewCocoMetric',
    ann_file=data_root + 'anno_valid/_annotations_filtered.bright.coco.json',  # Must match dataset ann_file!
    metric='bbox',
    format_only=False,
    classwise=True,  # Show per-class AP
    backend_args=None,
    # Multi-view specific settings
    views_per_sample=8,
    # EVALUATION MODE: Per-crop (no aggregation)
    # Radial crop pattern makes aggregation complex (rotation + translation needed)
    # For now: evaluate each crop independently (640 samples)
    # MVViT still learns cross-view relationships during training!
    aggregation='none',  # 'none' = per-crop evaluation, no aggregation
    enable_aggregation=False,  # Disable aggregation completely
    nms_iou_thr=0.5,    # IoU threshold for clustering overlapping boxes
    extract_base_name=True  # Extract base image name from crop filenames
)


test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Learning rate scheduler: HYBRID strategy for semi-supervised learning
# Strategy: Warmup + High LR plateau + Smooth decay
# This combines benefits of exploration (high LR) and convergence (cosine decay)
# 
# Phase 1 (0-10k):   Warmup - Linear increase from 0.01×lr to lr
#                     → Stable initialization, prevent early divergence
# Phase 2 (10k-80k):  High LR plateau - Constant lr=5e-4
#                     → Explore feature space, pseudo-labels improve over time
#                     → 70k iters at high LR helps hard classes (Severe_Rust, Chipped)
# Phase 3 (80k-120k): Cosine decay - Smooth decrease to 0.1×lr
#                     → Converge to optimal solution, no sudden drops
#                     → Better for teacher-student consistency
# 
# Benefits over pure CosineAnnealing:
# - Longer exploration phase (70k vs 90k, but at constant high LR)
# - Clear convergence phase (40k iters of smooth decay)
# - Better for semi-supervised: pseudo-labels stabilize during plateau
# 
# Benefits over MultiStepLR:
# - Smooth decay (no sudden drops that disrupt teacher-student EMA)
# - Reasonable final LR (5e-5, not too low like 5e-6)
# - Moderate training time (100k vs 180k)
param_scheduler = [
    # Phase 1: Warmup (0-10k)
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=10000),
    # Phase 2: High LR plateau for exploration (10k-70k)
    dict(type='ConstantLR', factor=1.0, by_epoch=False, begin=10000, end=70000),
    # Phase 3: Cosine decay for convergence (70k-100k)
    dict(type='CosineAnnealingLR', by_epoch=False, begin=70000, end=100000, eta_min=5e-5),
]

# Mixed precision disabled due to NMS dtype mismatch (Half vs Float)
# Error: batched_nms requires source and destination dtypes match
# Using FP32 training instead
# REDUCED LR: 1e-3 → 5e-4 to stabilize loss oscillation
# Problem: High variance (CV=5.18%) with batch_size=4
# Solution: Lower LR reduces update magnitude, smoother convergence
optim_wrapper = dict(
    type='OptimWrapper',  # FP32 training
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
)

# ✅ CORRECT: momentum=0.999 (Exponential Moving Average - EMA)
# Formula in MeanTeacherHook: teacher = momentum * teacher + (1-momentum) * student
# momentum=0.999 → teacher = 0.999 * teacher + 0.001 * student
# → Teacher updates SLOWLY, retains 99.9% old weights, 0.1% new (CORRECT!)
# → This is standard EMA used in Mean Teacher, Soft Teacher, BYOL, MoCo
# 
# momentum=0.001 would be WRONG → teacher = 0.001 * teacher + 0.999 * student
# → Teacher ≈ student immediately, loses stability, pseudo-labels become noisy!
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.999)]