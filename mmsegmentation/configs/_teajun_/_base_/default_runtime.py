# yapf:disable
log_config = dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='MMSegWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'segmentation', 
                name = '실험일_수행인_model_특이사항', 
                tags = ['최대한 많이, backbone, loss, dataset 등등 '],
                notes = '', 
            ),
            # Logging interval (iterations)
            interval = 50,
            # Save the checkpoint at every checkpoint interval as W&B Artifacts. Default False
            # You can reliably store these checkpoints as W&B Artifacts by using the log_checkpoint=True argument in MMDetWandbHook. 
            # This feature depends on the MMCV's CheckpointHook that periodically save the model checkpoints. 
            # The period is determined by checkpoint_config.interval.
            log_checkpoint=False,
            log_checkpoint_metadata=True,
            # The number of validation images to be logged. If zero, the evaluation won't be logged. Defaults to 100.
            num_eval_images=50 
            )     
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False
