_base_ = './mask_rcnn_r50_fpn_1x_coco_2gpu_base48.py'

lr_config = dict(step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)