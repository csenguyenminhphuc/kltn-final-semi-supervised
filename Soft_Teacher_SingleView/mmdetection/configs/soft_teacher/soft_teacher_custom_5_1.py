_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', '../_base_/default_runtime.py',
    '../_base_/datasets/semi_coco_detection.py'
]


detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],  # Caffe/BGR
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=16)
detector.backbone = dict(  #backbone resnet 50, style caffe
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
        checkpoint='open-mmlab://detectron2/resnet50_caffe'))


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
        pseudo_label_initial_score_thr=0.5,  
        rpn_pseudo_thr=0.4, #3
        cls_pseudo_thr=0.4, #4
        reg_pseudo_thr=0.02, #5 các parameter trong việc tạo nhãn giả
        jitter_times=5,
        jitter_scale=0.03, 
        min_pseudo_bbox_wh=(4, 4)),
    semi_test_cfg=dict(predict_on='teacher'))

dataset_type = 'CocoDataset'
data_root = '/home/coder/data/trong/KLTN/data_drill_3/'


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


# Configure semi-supervised dataloader

batch_size = 4
num_workers = 4

labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

labeled_dataset.ann_file = 'semi_anns/instances_train.1@5.0.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train.1@5.0-unlabeled.json'
unlabeled_dataset.data_prefix = dict(img='train/')

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 3]),
    persistent_workers=True,
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))


# training 
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=5000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')


param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=5000),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))


custom_hooks = [dict(type='MeanTeacherHook', momentum=0.999)]