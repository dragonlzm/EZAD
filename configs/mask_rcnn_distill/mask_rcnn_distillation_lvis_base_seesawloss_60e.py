_base_ = './mask_rcnn_distillation_lvis_base_seesawloss_36e.py'

# learning policy
lr_config = dict(step=[40, 55])
runner = dict(type='EpochBasedRunner', max_epochs=60)
evaluation = dict(interval=72, metric=['bbox', 'segm'])