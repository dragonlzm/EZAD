_base_ = './mask_rcnn_distillation_lvis_base_seesawloss_36e.py'

# learning policy
lr_config = dict(step=[32, 44])
runner = dict(type='EpochBasedRunner', max_epochs=48)
evaluation = dict(interval=60, metric=['bbox', 'segm'])