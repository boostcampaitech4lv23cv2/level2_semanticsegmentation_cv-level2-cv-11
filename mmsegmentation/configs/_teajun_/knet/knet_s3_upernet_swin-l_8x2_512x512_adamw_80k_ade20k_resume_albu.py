_base_ = 'knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k.py'

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'  # noqa
# model settings
model = dict(
    pretrained=checkpoint_file,
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        kernel_generate_head=dict(in_channels=[192, 384, 768, 1536])),
    auxiliary_head=dict(in_channels=768))
# In K-Net implementation we use batch size 2 per GPU as default
data = dict(samples_per_gpu=2, workers_per_gpu=2)
resume_from = '/opt/ml/mmsegmentation/work_dirs/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k_resume_30000/iter_20000.pth'
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=2000)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[10000, 25000],
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=35000)
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
