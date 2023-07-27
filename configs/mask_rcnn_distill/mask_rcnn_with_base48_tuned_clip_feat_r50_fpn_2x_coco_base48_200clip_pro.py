_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro.py'

# learning policy
lr_config = dict(step=[16, 22])
runner = dict(max_epochs=24)
