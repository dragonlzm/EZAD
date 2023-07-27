_base_ = './mask_rcnn_distillation_lvis_base_range_scale_3x_repeat.py'

# learning policy
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=866,
                loss_weight=1.0))))