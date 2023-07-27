_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro_with_filp.py'

# with filp become a default setting
# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(reg_with_cls_embedding=True)))
