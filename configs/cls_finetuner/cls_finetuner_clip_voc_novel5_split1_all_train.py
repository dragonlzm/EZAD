_base_ = './cls_finetuner_clip_voc_base15_split1_all_train.py'

data_root = 'data/coco/'

classes = ('bird', 'bus', 'cow', 'motorbike', 'sofa')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

model = dict(
    rpn_head=dict(
        cate_names=['bird', 'bus', 'cow', 'motorbike', 'sofa']    
        ))