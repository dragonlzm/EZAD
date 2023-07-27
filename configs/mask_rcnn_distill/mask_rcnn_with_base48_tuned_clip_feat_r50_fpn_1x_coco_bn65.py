_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco.py'

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
 'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
 'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
 'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
 'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')

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
         num_of_rand_bbox=100, select_fixed_subset=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats']),
]

data = dict(
    train=dict(classes=classes, pipeline=train_pipeline),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=65,
                       fg_vec_cfg=dict(fixed_param=True, 
                                       #load_path='/data2/lwll/zhuoming/detection/embeddings/base_finetuned_48cates.pt',
                                       load_path='data/embeddings/base_finetuned_65cates.pt')),
        mask_head=dict(num_classes=65)))

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
