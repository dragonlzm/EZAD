_base_ = './mask_rcnn_distillation_lvis_base_12e_range_scale.py'

# optimizer
optimizer = dict(lr=0.005)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=36, metric=['bbox', 'segm'])