_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', '../_base_/default_runtime.py',
    '../_base_/datasets/semi_coco_detection.py'
]
# Ensure mmengine/mmcv imports our custom MultiView module so the registry can find
# `MultiViewBackbone`. This avoids KeyError when building model from cfg.
# Import the collate callable directly so MMEngine receives a callable for
# `collate_fn` (it does not accept string import paths for callables).

# Only import modules (not individual callables) via custom_imports so the
# registry decorators run when MMEngine loads the config. Keep module paths
# here; the collate callable is imported above and used directly.
custom_imports = dict(imports=['mmdet.models.utils.multi_view', 'mmdet.datasets.wrappers.multi_view_from_folder'], allow_failed_imports=False)
metainfo = {
    'classes': ('Gay','Me','Mon_dau','Ri_set','Xuoc'),
    'palette': [
        (134, 34, 255),
        (0, 255, 206),
        (255, 128, 0),
        (254, 0, 86),
        (199, 252, 0)
    ]
}

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
    fusion='none',
    concat_reduce=True,
    # if using weighted fusion, optional weights can be set via config of the
    # backbone (not as an extra keyword). Remove unsupported keys to avoid
    # unexpected kwargs on construction.
)


model = dict(
    _delete_=True,
    type='SoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True, 
        sup_weight=1.0, 
        unsup_weight=1.0,   
        pseudo_label_initial_score_thr=0.3,  
        rpn_pseudo_thr=0.4, #3
        cls_pseudo_thr=0.4, #4
        reg_pseudo_thr=0.02, #5 các parameter trong việc tạo nhãn giả
        jitter_times=5,
        jitter_scale=0.03, 
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

dataset_type = 'CocoDataset'
data_root = '/home/coder/trong/KLTN_SEMI/code/Soft_Teacher/data_drill/'


# color_space = [
#     [dict(type='ColorTransform')],
#     [dict(type='AutoContrast')],
#     [dict(type='Equalize')],
#     [dict(type='Sharpness')],
#     [dict(type='Posterize')],
#     [dict(type='Solarize')],
#     [dict(type='Color')],
#     [dict(type='Contrast')],
#     [dict(type='Brightness')],
# ]

# geometric = [
#     [dict(type='Rotate')],
#     [dict(type='ShearX')],
#     [dict(type='ShearY')],
#     [dict(type='TranslateX')],
#     [dict(type='TranslateY')],
# ]

# Tập phép phù hợp grayscale; tập trung vào cường độ và độ tương phản
color_space = [
    [dict(type='AutoContrast', prob=0.5)],
    [dict(type='Equalize',    prob=0.4)],
    # Brightness: giữ sáng gốc quanh 1.0; map level → [0.8, 1.25]
    [dict(type='Brightness',  prob=0.7, level=7, min_mag=0.8, max_mag=1.25)],
    # Contrast: tương tự
    [dict(type='Contrast',    prob=0.7, level=7, min_mag=0.8, max_mag=1.3)],
    # Sharpness: nhẹ
    [dict(type='Sharpness',   prob=0.3, level=6, min_mag=0.8, max_mag=1.5)],
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


scale = [(128, 360), (256, 720)]
# Configure semi-supervised dataloader

batch_size = 1
# For smoke-run reduce workers to avoid host RAM pressure
num_workers = 2

# Use MultiViewFromFolder for multi-view samples (9 crops per sample)
labeled_dataset = dict(
    type='MultiViewFromFolder',
    root=data_root + 'train/',
    views_per_sample=1,
    ann_file=data_root + 'semi_anns/_annotations.coco.labeled.grouped@20.json',
    pipeline=_base_.sup_pipeline,
    metainfo=_base_.metainfo,
)

unlabeled_dataset = dict(
    type='MultiViewFromFolder',
    root=data_root + 'train/',
    views_per_sample=1,
    ann_file=data_root + 'semi_anns/_annotations.coco.unlabeled.grouped@20.json',
    pipeline=_base_.unsup_pipeline,
    metainfo=_base_.metainfo,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 4]),
    persistent_workers=False,
    pin_memory=False,          # ✅ tránh double buffer
    prefetch_factor=1,         # ✅ giảm queue size
    collate_fn=dict(type='mmdet.datasets.wrappers.multi_view_from_folder.multi_view_collate_flatten'),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))


# training 
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100, val_interval=5)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# Explicit validation dataloader/evaluator for multi-view validation
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='mmdet.datasets.wrappers.multi_view_from_folder.multi_view_collate_flatten'),
    dataset=dict(
        type='MultiViewFromFolder',
        root=data_root + 'valid/',
        views_per_sample=9,
        ann_file=data_root + 'anno_valid/_annotations.coco.json',
        pipeline=_base_.test_pipeline,
        metainfo=_base_.metainfo,
    ))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'anno_valid/_annotations.coco.json',
    metric='bbox',
    format_only=False,
    backend_args=None)


param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))


custom_hooks = [dict(type='MeanTeacherHook', momentum=0.999)]