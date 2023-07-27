_base_ = './cls_finetuner_clip_voc_base15_split1_all_train.py'

model = dict(
    rpn_head=dict(
        num_classes=15,
        cate_names=['aeroplane', 'bicycle', 'bird', 'bottle', 'bus',
                    'car', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'person', 'pottedplant', 'train', 'tvmonitor']))

classes = ('aeroplane', 'bicycle', 'bird', 'bottle', 'bus',
            'car', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'person', 'pottedplant', 'train', 'tvmonitor')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

