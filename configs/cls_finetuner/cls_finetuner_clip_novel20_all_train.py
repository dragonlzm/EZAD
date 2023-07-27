_base_ = './cls_finetuner_clip_full_coco.py'

data_root = 'data/coco/'

classes = ('person', 'bicycle', 'car', 'motorcycle', 
'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 
'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 
'couch', 'potted plant', 'dining table', 'tv')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]

model = dict(
    rpn_head=dict(
        cate_names=['person', 'bicycle', 'car', 'motorcycle', 
        'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 
        'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 
        'couch', 'potted plant', 'dining table', 'tv']    
        ))