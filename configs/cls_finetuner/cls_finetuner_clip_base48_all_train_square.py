_base_ = './cls_finetuner_clip_base48_all_train.py'

test_pipeline = [
    dict(type='LoadImageFromFile', channel_order='rgb'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(val=dict(eval_filter_empty_gt=True, pipeline=test_pipeline),
            test=dict(eval_filter_empty_gt=True, pipeline=test_pipeline))

