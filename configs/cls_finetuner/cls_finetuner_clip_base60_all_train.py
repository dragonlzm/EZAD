_base_ = './cls_finetuner_clip_full_coco.py'

data_root = 'data/coco/'

classes = ('truck', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
            'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 
            'teddy bear', 'hair drier', 'toothbrush')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))


model = dict(
    rpn_head=dict(
        cate_names=['truck', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
            'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 
            'teddy bear', 'hair drier', 'toothbrush']  
        ))
#sorted
#['truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv'