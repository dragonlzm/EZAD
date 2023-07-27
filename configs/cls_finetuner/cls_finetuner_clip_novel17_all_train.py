_base_ = './cls_finetuner_clip_full_coco.py'

data_root = 'data/coco/'

classes = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]

model = dict(
    rpn_head=dict(
        cate_names=['airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors']    
        ))