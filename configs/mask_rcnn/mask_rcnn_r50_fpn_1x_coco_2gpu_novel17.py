_base_ = './mask_rcnn_r50_fpn_1x_coco_2gpu.py'

classes = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes, eval_filter_empty_gt=True),
    test=dict(classes=classes, eval_filter_empty_gt=True))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=17),
        mask_head=dict(num_classes=17)))