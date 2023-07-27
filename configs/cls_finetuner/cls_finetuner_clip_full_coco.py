_base_ = './cls_finetuner_coco.py'

optimizer = dict(
    _delete_=True,    
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(_delete_=True, policy='step', step=[])
runner = dict(max_epochs=12)

# change the classification loss
# add the text encoder module
# fixed the all parametere except the ln of the text encoder
model = dict(
    type='ClsFinetuner',
    neck=None,
    backbone=dict(open_ln=True),
    rpn_head=dict(
        type='ClipEncoderHead',
        num_classes=80,
        in_channels=512,
        vocab_size=49408,
        transformer_width=512,
        transformer_layers=12,
        transformer_heads=8,
        embed_dim=512,
        context_length=77,
        sentence_templates=["a photo of a {}"],
        cate_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
            'hair drier', 'toothbrush'],
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0),
        #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/clip_vitb32_full.pth"),
        init_cfg=dict(type='Pretrained', checkpoint="data/pretrain/clip_vitb32_full.pth"),        
        ))