_base_ = './cls_finetuner_clip_voc_base15_split1_all_train.py'

data_root = 'data/coco/'

classes = ('boat', 'cat', 'motorbike', 'sheep', 'sofa')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

model = dict(
    rpn_head=dict(
        cate_names=['boat', 'cat', 'motorbike', 'sheep', 'sofa']    
        ))