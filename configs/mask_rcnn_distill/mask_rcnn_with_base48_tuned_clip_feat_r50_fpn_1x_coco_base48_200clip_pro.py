_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    #dict(type='LoadCLIPFeat', file_path_prefix='data/coco/feat/base48_finetuned',
    #     num_of_rand_bbox=100),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/base48_finetuned',
         num_of_rand_bbox=200, select_fixed_subset=200),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline))

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            fg_vec_cfg=dict(load_path='data/embeddings/base_finetuned_48cates.pt'))))
