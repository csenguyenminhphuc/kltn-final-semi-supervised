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
    # - spatial_attention='moderate': Keep K=512 tokens (good balance)
    # - use_gradient_checkpointing=True: ~50% memory saving in backward pass
    mvvit=dict(
        type='MVViT',
        embed_dim=256,       # Match FPN channels
        num_heads=8,         # INCREASED from 4 → more diverse attention for hard classes
        num_layers=2,        # INCREASED from 1 → deeper cross-view reasoning (CRITICAL!)
        mlp_ratio=3.0,       # INCREASED from 2.0 → more capacity for subtle features
        dropout=0.1,         # Dropout rate (unchanged)
        use_layer_norm=True, # Use LayerNorm instead of BatchNorm
        spatial_attention='moderate',  # Pool to 512 tokens BEFORE transformer (saves FLOPs!)
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
        scales=[4, 8, 16],  # Multiple scales for objects of different sizes
        # TARGETED RATIOS for Severe_Rust (med=0.45, P25-P75: 0.34-0.61) and Chipped (med=0.51, P25-P75: 0.39-0.65)
        # Analysis shows 60% of Severe_Rust and 48% of Chipped have ratio < 0.5 (narrow boxes)
        ratios=[0.2, 0.35, 0.45, 0.55, 0.65, 1.0, 2.0],  # Added 0.45, 0.55, 0.65 for narrow defects
        strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(
        type='FocalLoss',  # CHANGED from CrossEntropyLoss → focus on hard examples
        use_sigmoid=True,
        gamma=2.5,         # INCREASED to 2.5 (was 2.0) → stronger hard example focus
        alpha=0.75,        # INCREASED from 0.25 → weight FG more heavily (sparse data!)
        loss_weight=2.5),  # INCREASED from 2.0 → emphasize classification
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
    gamma=3.0,            # INCREASED to 3.0 (was 2.5) → EXTREME focus on hard examples!
    alpha=0.75,           # High alpha → weight FG more (helps rare classes)
    loss_weight=2.5)      # INCREASED from 2.0 → emphasize classification even more

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
            pos_fraction=0.8,     # INCREASED to 0.8 → 80% must be positive (only 51 negatives!)
            neg_pos_ub=1,         # CRITICAL: max 1 negative per positive (1:1 ratio, not 1:3!)
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
            pos_fraction=0.75,    # 75% positive (sparse data!)
            neg_pos_ub=1,         # Max 1 negative per positive
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))

# Test config for RPN: increase threshold to reduce false positives
detector.test_cfg.rpn = dict(
    nms_pre=4000,           # Increased from 2000 (was too strict)
    max_per_img=2000,       # Increased from 1000 (was too strict)
    nms=dict(type='nms', iou_threshold=0.7),
    min_bbox_size=8)        # Reduced from 16 to 8 (less aggressive filtering)

# Test config for RCNN: stricter threshold
detector.test_cfg.rcnn = dict(
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)


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
        unsup_weight=0.5,   # Balance supervised vs unsupervised learning
        # AGGRESSIVE THRESHOLDS for hard class learning (Severe_Rust: 0.006 mAP, Chipped: 0.032 mAP)
        # Analysis: Despite having 27.6% and 19.2% of data, these classes have VERY low performance
        # → Problem is NOT data quantity but QUALITY/DIFFICULTY of examples
        # Solution: LOWER thresholds aggressively to allow teacher to generate more pseudo-labels
        #           even for uncertain predictions → help student learn from harder examples
        # 
        # Threshold hierarchy (strictest to loosest):
        # 1. pseudo_label_initial_score_thr: Initial filtering
        # 2. cls_pseudo_thr: Classification loss filtering  
        # 3. rpn_pseudo_thr: RPN loss filtering
        pseudo_label_initial_score_thr=0.4,  # LOWERED from 0.5 → allow more uncertain pseudo-labels
        cls_pseudo_thr=0.35,                 # LOWERED from 0.4 → help hard examples learn
        rpn_pseudo_thr=0.35,                 # LOWERED from 0.4 → allow more proposals
        reg_pseudo_thr=0.02,                 # Bbox uncertainty threshold (unchanged)
        jitter_times=5,                      # Augmentation for uncertainty estimation
        jitter_scale=0.03,                   # Scale of jitter augmentation
        min_pseudo_bbox_wh=(8, 8)),         # Filter small noisy boxes
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
    [dict(type='AutoContrast', prob=0.6)],  # Increased from 0.5 → help rust patterns
    [dict(type='Equalize',    prob=0.5)],   # Increased from 0.4 → normalize lighting
    # AGGRESSIVE brightness variation for Severe_Rust (rust visibility changes with light)
    [dict(type='Brightness',  prob=0.8, level=7, min_mag=0.7, max_mag=1.4)],  # Wider range: 0.7-1.4 (was 0.8-1.25)
    # AGGRESSIVE contrast for both classes (rust patterns, crack visibility)
    [dict(type='Contrast',    prob=0.8, level=7, min_mag=0.7, max_mag=1.5)],  # Wider range: 0.7-1.5 (was 0.8-1.3)
    # INCREASED sharpness for Chipped (small cracks need sharp edges to detect)
    [dict(type='Sharpness',   prob=0.5, level=6, min_mag=0.7, max_mag=2.0)],  # Increased prob 0.3→0.5, range 0.7-2.0
    # ADD: Posterize to simulate different lighting conditions for rust
    [dict(type='Posterize',   prob=0.3, level=5, min_mag=4, max_mag=8)],
]
geometric = [
    # Rotate trong khoảng ±5°
    [dict(type='Rotate',     prob=0.3, level=7, min_mag=0.0,  max_mag=5.0)],
    # Shear nhẹ ~ ±3°
    [dict(type='ShearX',     prob=0.3, level=7, min_mag=0.0,  max_mag=3.0)],
    [dict(type='ShearY',     prob=0.3, level=7, min_mag=0.0,  max_mag=3.0)],
    # Translate tối đa 5% kích thước
    [dict(type='TranslateX', prob=0.3, level=7, min_mag=0.0,  max_mag=0.05)],
    [dict(type='TranslateY', prob=0.3, level=7, min_mag=0.0,  max_mag=0.05)],
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
strong_pipeline = [
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
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
        source_ratio=[1, 3]),
    collate_fn=dict(type='mmdet.datasets.wrappers.multi_view_from_folder.multi_view_collate_flatten'),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))


# training 
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=5000)
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
        ann_file=data_root + 'anno_valid/_annotations_filtered.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
        backend_args=backend_args
    ))

val_evaluator = dict(
    type='MultiViewCocoMetric',
    ann_file=data_root + 'anno_valid/_annotations_filtered.coco.json',  # Must match dataset ann_file!
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

# Learning rate scheduler:
# 1. Warmup (0-7000 iters): Linear warmup from 0.01*lr to lr
# 2. Main training (7000-40000 iters): Cosine annealing from lr to 0.1*lr
# Strategy: Slower LR decay helps stabilize pseudo-label learning
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=7000),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,      # Iter-based (not epoch-based)
        begin=7000,          # Start after warmup
        end=40000,           # End of training
        eta_min=5e-5         # Final LR = 0.1 * base_lr (5e-5 / 5e-4)
        # NOTE: convert_to_iter_based removed (only for by_epoch=True)
    )
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