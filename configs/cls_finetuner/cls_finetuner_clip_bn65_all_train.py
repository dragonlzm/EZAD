_base_ = './cls_finetuner_clip_full_coco.py'

data_root = 'data/coco/'

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]

model = dict(
    rpn_head=dict(
        cate_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                    'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                    'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
                    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                    'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
                    'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
                    'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush']  
        ))
