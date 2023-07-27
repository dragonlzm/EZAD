# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .mask_rcnn_with_clip_feat import MaskRCNNWithCLIPFeat


@DETECTORS.register_module()
class FasterRCNNWithCLIPFeat(MaskRCNNWithCLIPFeat):
    """this is the class of hte Faster R-CNN trained with distillation for zero-shot object detection"""
    
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 unknow_rpn_head=None,
                 roi_head=None,
                 rand_bboxes_num=20,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 vit_backbone_cfg=None):
        super(FasterRCNNWithCLIPFeat, self).__init__(
                backbone,
                neck=neck,
                rpn_head=rpn_head,
                unknow_rpn_head=unknow_rpn_head,
                roi_head=roi_head,
                rand_bboxes_num=rand_bboxes_num,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                pretrained=pretrained,
                init_cfg=init_cfg,
                vit_backbone_cfg=vit_backbone_cfg)
