_base_ = './mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405.py'

# learning policy
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=405,
                loss_weight=1.0))))