_base_ = './mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py'

classes = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

data = dict(
    train=dict(dataset=dict(classes=classes)),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=48),
        mask_head=dict(num_classes=48)))


# for 2 gpu training
optimizer = dict(lr=0.005)