_base_ = './mask_rcnn_r50_fpn_1x_coco_class_agnostic.py'

# the lr = 0.02 used for 8 gpu training, bs = 2*8
# the lr 0.005 should match the bs = 2*2
optimizer = dict(lr=0.005)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50-0676ba61.pth')))

