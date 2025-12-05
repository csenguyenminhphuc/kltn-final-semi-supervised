# base model, and dataset
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', '../_base_/default_runtime.py',
    '../_base_/datasets/semi_coco_detection.py'
]

# config semi supervised model
detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)
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
        freeze_teacher=True, # model teacher sẽ ko đc cập nhật trong lúc train 
        sup_weight=1.0, # trọng số dữ liệu đc gán nhãn
        unsup_weight=4.0,   #1
        pseudo_label_initial_score_thr=0.5,  #2
        rpn_pseudo_thr=0.9, #3
        cls_pseudo_thr=0.9, #4
        reg_pseudo_thr=0.02, #5 các parameter trong việc tạo nhãn giả
        jitter_times=10,    # 
        jitter_scale=0.06, #  tạo nhiễu và tăng cường dữ liệu không được gán nhãn
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

# labeled_dataset = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='annotations/instances_train2017.json',
#     data_prefix=dict(img='train2017/'),
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=sup_pipeline)

labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=4,
        source_ratio=[1, 2]),
    persistent_workers=True,
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))
# unlabeled_dataset = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='annotations/instances_unlabeled2017.json',
#     data_prefix=dict(img='unlabeled2017/'),
#     filter_cfg=dict(filter_empty_gt=False),
#     pipeline=unsup_pipeline)


# training 
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=13520, val_interval=2000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))


custom_hooks = [dict(type='MeanTeacherHook')]
