# dataset settings
dataset_type = 'TrashDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

# train pipeline에 적용되는 Albu transforms
# aug 2
albu_train_transforms =[
    #aug 1 = all
            # dict(
            #     type='ShiftScaleRotate', # 돌리고 밀기
            #     shift_limit=0.0625,
            #     scale_limit=0,
            #     rotate_limit=30,
            #     p=0.5,
            # ),
            dict(
                type='OneOf', # 이 중에 하나만 적용
                transforms=[
                    dict(type='ElasticTransform', p=1.0), # 이미지를 찌그러지게 만든다. # https://www.researchgate.net/figure/Grid-distortion-and-elastic-transform-applied-to-a-medical-image_fig4_327742409
                    dict(type='Perspective', p=1.0), # 4점 투시 변환 # https://minimin2.tistory.com/135
                    dict(type='PiecewiseAffine', p=1.0), # 이미지를 찌그러지게 만든다 # https://scikit-image.org/docs/stable/auto_examples/transform/plot_piecewise_affine.html
                ],
                p=0.3),
            # dict(
            #     type='Affine', # 아핀 변환
            #   p=0.3  
            # ),
            # dict(
            #     type='OneOf',
            #     transforms=[
            #         dict(type='RGBShift', r_shift_limit=20, g_shift_limit=20,b_shift_limit=20,always_apply=False,p=1.0), # rgb 값 변경
            #         dict(type='ChannelShuffle', p=1.0) # rgb 값 무작위 재배열
            #     ],
            #     p=0.5),
            dict(
                type='RandomBrightnessContrast', # 밝기와 대비를 무작위로 변경
                brightness_limit=0.1,
                contrast_limit=0.15,
                p=0.5),
            dict(
                type='HueSaturationValue',  #색조, 채도 및 값을 무작위로 변경
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=10,
                p=0.5),
            # dict(type='GaussNoise', p=0.3), # 가우시안 노이즈를 추가
            dict(type='CLAHE', p=0.5), # https://m.blog.naver.com/samsjang/220543360864
            # dict(
            #     type='OneOf',
            #     transforms=[ # 블러 적용
            #         dict(type='Blur', p=1.0),
            #         dict(type='GaussianBlur', p=1.0),
            #         dict(type='MedianBlur', blur_limit=5, p=1.0)
            #     ],
            #     p=0.3),
        ]

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
