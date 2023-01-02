# dataset settings
dataset_type = 'TrashDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

# train pipeline에 적용되는 Albu transforms
#aug3
albu_train_transforms_aug3 = [
    dict(type='RandomBrightnessContrast', brightness_limit=0.1, contrast_limit=0.15, p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(type='CLAHE',p=0.5),
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', blur_limit=5, p=1.0),
        dict(type='MotionBlur', p=1.0)
    ], p=0.1
    )
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CustomLoadAnnotations', reduce_zero_label=False,
         coco_json_path='/opt/ml/input/data/train.json'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    ######### 필수 pipline ##########
    ######### augmentation #########
    
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), #현재 crop 안되는 중
    dict(type='RandomFlip', prob=0.5), # horizontal flip
    dict(type='PhotoMetricDistortion'), ## 이미지에 광도 왜곡을 순차적으로 적용
    # alumentation 사용(albu_train_transforms)
    dict(
        type='Albu',
        transforms=albu_train_transforms_aug3,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False,
        ),
    
    ######### augmentation #########
    ######### 필수 pipline ##########
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        is_valid=False,
        img_dir='/opt/ml/input/data',
        coco_json_path='/opt/ml/input/data/train.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        is_valid=True,
        img_dir='/opt/ml/input/data',
        coco_json_path='/opt/ml/input/data/val.json',
        pipeline=valid_pipeline),
    test=dict(
        type=dataset_type,
        is_valid=True,
        img_dir='/opt/ml/input/data',
        coco_json_path='/opt/ml/input/data/test.json',
        pipeline=test_pipeline))
