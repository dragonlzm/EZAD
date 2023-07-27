_base_ = './mask_rcnn_distillation_voc_split3_base_r101_fewshot_dataset.py'

data = dict(
    train=dict(classes='ALL_CLASSES_SPLIT3'),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))

evaluation = dict(
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20,
                       fg_vec_cfg=dict(fixed_param=True,
                                       load_path='data/embeddings/base_finetuned_voc_split1_20cates.pt'))))
