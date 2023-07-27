_base_ = './mask_rcnn_distillation_voc_split1_base_r101.py'

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                   (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='LoadCLIPFeat', file_path_prefix='data/VOCdevkit/clip_proposal_feat/split1_base_finetuned',
         num_of_rand_bbox=200, select_fixed_subset=200, load_rand_bbox_weight=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_feats',
                               'rand_bboxes', 'rand_feats', 'rand_bbox_weights'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

### repeat daatset
data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[5,6])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=7)  # actual epoch = 4 * 3 = 12
