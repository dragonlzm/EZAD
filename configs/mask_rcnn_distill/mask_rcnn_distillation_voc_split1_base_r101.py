_base_ = './mask_rcnn_distillation_voc_split1_base_reg_with_embed.py'

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='/data/zhuoming/detection/pretrain/resnet101-63fe2227.pth')))
