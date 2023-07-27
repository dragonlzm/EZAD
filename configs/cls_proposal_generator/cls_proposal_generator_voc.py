_base_ = [
    '../_base_/models/cls_proposal_generator.py', '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', channel_order='rgb'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile', channel_order='rgb'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data_root = 'data/VOCdevkit/'
data = dict(train=dict(pipeline=train_pipeline),
            val=dict(eval_filter_empty_gt=True, 
                pipeline=test_pipeline, 
                ann_file=data_root + 'annotations/train_100imgs.json',
                img_prefix=data_root + 'train2017/',
                ),
            test=dict(eval_filter_empty_gt=True, 
                pipeline=test_pipeline, 
                ann_file=data_root + 'annotations/train_100imgs.json',
                img_prefix=data_root + 'train2017/',
                ))

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(
        eval_filter_empty_gt=True, 
        #ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        #img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        eval_filter_empty_gt=True, 
        #ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        #img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='proposal_fast')

lr_config = dict(step=[])
runner = dict(type='EpochBasedRunner', max_epochs=6)


# for model setting
model = dict(
    rpn_head=dict(
        num_classes=20,
        cate_names=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']))