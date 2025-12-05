# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/coder/data/trong/KLTN/data_drill_3/'


backend_args = None

metainfo = {
    'classes': ('Broken','Chipped','Scratched','Severe_Rust','Tip_Wear'),
    'palette': [
            (134, 34, 255),   
            (0, 255, 206),
            (199, 252, 0), 
            (254, 0, 86), 
            (255, 128, 0), 
    ]
}

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

# scale = [(1333, 400), (1333, 1200)]
color_space = [
    [dict(type='AutoContrast', prob=0.5)],
    [dict(type='Equalize',    prob=0.4)],
    # Brightness: giữ sáng gốc quanh 1.0; map level → [0.8, 1.25]
    [dict(type='Brightness',  prob=0.7, level=7, min_mag=0.8, max_mag=1.25)],
    # Contrast: tương tự
    [dict(type='Contrast',    prob=0.7, level=5, min_mag=0.8, max_mag=1.3)],
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

scale=[(128,360), (256,720)]

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
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
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    # dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadEmptyAnnotations'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 720), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 4
num_workers = 4
# There are two common semi-supervised learning settings on the coco dataset：
# (1) Divide the train2017 into labeled and unlabeled datasets
# by a fixed percentage, such as 1%, 2%, 5% and 10%.
# The format of labeled_ann_file and unlabeled_ann_file are
# instances_train2017.{fold}@{percent}.json, and
# instances_train2017.{fold}@{percent}-unlabeled.json
# `fold` is used for cross-validation, and `percent` represents
# the proportion of labeled data in the train2017.
# (2) Choose the train2017 as the labeled dataset
# and unlabeled2017 as the unlabeled dataset.
# The labeled_ann_file and unlabeled_ann_file are
# instances_train2017.json and image_info_unlabeled2017.json
# We use this configuration by default.
labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='semi_anns/instances_train.3@40.0.json',
    data_prefix=dict(img='train/'),
    metainfo= metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=8),
    pipeline=sup_pipeline,
    backend_args=backend_args)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='semi_anns/instances_train.3@40.0-unlabeled.json',
    data_prefix=dict(img='train/'),
    metainfo= metainfo,
    filter_cfg=dict(filter_empty_gt=False, min_size=8),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 4]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='anno_valid/_annotations_filtered.coco.json',
        # data_prefix=dict(img='valid/'),
        ann_file='anno_test/_annotations_filtered.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo= metainfo,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'anno_valid/_annotations_filtered.coco.json',
    ann_file=data_root + 'anno_test/_annotations_filtered.coco.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator
