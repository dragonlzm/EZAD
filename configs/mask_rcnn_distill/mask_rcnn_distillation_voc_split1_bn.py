_base_ = './mask_rcnn_distillation_voc_split1_base.py'

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

data = dict(
    train=dict(dataset=dict(classes=classes)),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20,
                       fg_vec_cfg=dict(fixed_param=True,
                                       load_path='data/embeddings/base_finetuned_voc_split1_20cates.pt'))))
