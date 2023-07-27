_base_ = './mask_rcnn_distillation_voc_split1_base_r101_fewshot_dataset.py'

classes = ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
            'chair', 'diningtable', 'dog', 'motorbike', 'person',
            'pottedplant', 'sheep', 'train', 'tvmonitor')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/VOCdevkit/clip_proposal_feat/split2_base_finetuned',
         num_of_rand_bbox=200, select_fixed_subset=200, load_rand_bbox_weight=True),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_feats',
                               'rand_bboxes', 'rand_feats', 'rand_bbox_weights']),
]

data = dict(
    train=dict(pipeline=train_pipeline, classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(fg_vec_cfg=dict(fixed_param=True,
                        load_path='data/embeddings/base_finetuned_voc_split2_15cates.pt'))))
