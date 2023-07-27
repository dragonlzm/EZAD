_base_ = './cls_finetuner_clip_voc_base15_split1_all_train.py'

model = dict(
    rpn_head=dict(
        num_classes=15,
        cate_names=['bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
                         'chair', 'diningtable', 'dog', 'motorbike', 'person',
                         'pottedplant', 'sheep', 'train', 'tvmonitor']))

classes = ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
            'chair', 'diningtable', 'dog', 'motorbike', 'person',
            'pottedplant', 'sheep', 'train', 'tvmonitor')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

