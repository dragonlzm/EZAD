_base_ = './cls_finetuner_clip_full_coco_resnet50.py'

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
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]

model = dict(
    rpn_head=dict(
        cate_names=['person', 'bicycle', 'car', 'motorcycle', 'train', 
        'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 
        'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 
        'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
        'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
        'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 
        'vase', 'toothbrush']  
        ))
