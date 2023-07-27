_base_ = './mask_rcnn_distillation_lvis_base.py'

# learning policy
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=24, metric=['bbox', 'segm'])