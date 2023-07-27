_base_ = './mask_rcnn_distillation_lvis_base_range_scale.py'

# learning policy
lr_config = dict(step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(interval=48, metric=['bbox', 'segm'])