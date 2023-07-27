_base_ = './mask_rcnn_distillation_r101_base60.py'

# with filp become a default setting
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/fewshot_finetuned',
         num_of_rand_bbox=200, select_fixed_subset=200, load_rand_bbox_weight=True),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_feats',
                               'rand_bboxes', 'rand_feats', 'rand_bbox_weights']),
]

data = dict(
    train=dict(pipeline=train_pipeline))


lr_config = dict(policy='step', step=[14,16])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=18)  # actual epoch = 4 * 3 = 12
