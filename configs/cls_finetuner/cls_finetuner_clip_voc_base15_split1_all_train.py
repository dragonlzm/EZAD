_base_ = './cls_finetuner_clip_full_coco.py'

model = dict(
    rpn_head=dict(
        num_classes=15,
        cate_names=['aeroplane', 'bicycle', 'boat', 'bottle', 'car',
                         'cat', 'chair', 'diningtable', 'dog', 'horse',
                         'person', 'pottedplant', 'sheep', 'train',
                         'tvmonitor']))

classes = ('aeroplane', 'bicycle', 'boat', 'bottle', 'car',
                         'cat', 'chair', 'diningtable', 'dog', 'horse',
                         'person', 'pottedplant', 'sheep', 'train',
                         'tvmonitor')

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1000, 600),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

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


data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt'
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=classes,
        eval_filter_empty_gt=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=classes,
        eval_filter_empty_gt=True))
evaluation = dict(interval=1, metric='gt_acc')



# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)  

