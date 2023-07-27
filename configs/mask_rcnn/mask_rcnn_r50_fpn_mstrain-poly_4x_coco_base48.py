_base_ = './mask_rcnn_r50_fpn_mstrain-poly_3x_coco_base48.py'

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 15])
runner = dict(type='EpochBasedRunner', max_epochs=16)
