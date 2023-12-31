# model settings
model = dict(
    type='ClsProposalGenerator',
    neck=None,
    backbone=dict(
        type='myVisionTransformer',
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        #init_cfg=dict(type='Pretrained', checkpoint="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")),
        #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/modified_state_dict.pth"),
        #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/clip_vitb32_full.pth", prefix='visual.'),
        init_cfg=dict(type='Pretrained', checkpoint="data/pretrain/clip_vitb32_full.pth", prefix='visual.'),
        fixed_param=True,
        open_ln=True),
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
        return_test_score=True),
    anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[32])
        )