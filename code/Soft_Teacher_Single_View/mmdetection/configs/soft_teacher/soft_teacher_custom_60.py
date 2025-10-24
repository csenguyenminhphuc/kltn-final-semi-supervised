_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', '../_base_/default_runtime.py',
    '../_base_/datasets/semi_coco_detection.py'
]


detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[62.232],
    std=[1.0],
    bgr_to_rgb=False,
    pad_size_divisor=1)
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
        rpn_pseudo_thr=0.6, #3
        cls_pseudo_thr=0.6, #4
        reg_pseudo_thr=0.02, #5 các parameter trong việc tạo nhãn giả
        jitter_times=5,
        jitter_scale=0.03, 
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

dataset_type = 'CocoDataset'
data_root = 'data/coco/'


color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

scale = [(1333, 400), (1333, 1200)]


# Configure semi-supervised dataloader

batch_size = 4
num_workers = 4

labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@60.0.json'
unlabeled_dataset.ann_file = 'semi_anns/' \
                             'instances_train2017.1@60.0-unlabeled.json'
unlabeled_dataset.data_prefix = dict(img='train/')

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=4,
        source_ratio=[1, 1]),
    persistent_workers=True,
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))


# training 
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=6020, val_interval=1500)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')


param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=6000,
        by_epoch=False,
        milestones=[1200, 4000],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))


custom_hooks = [dict(type='MeanTeacherHook')]