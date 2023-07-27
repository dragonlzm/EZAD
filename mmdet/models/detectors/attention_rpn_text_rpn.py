# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional
import copy
import numpy as np
import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from ..builder import DETECTORS
from torch import Tensor

from .query_support_detector import QuerySupportDetector


@DETECTORS.register_module()
class AttentionRPNTextRPN(QuerySupportDetector):
    """
    Here we implement the our version of the RPN in attentionRPN detector.
    We replace the learnable vision class prototype with text embedding from CLIP.
    Since RPN is regarded as an detector. we place this class in this folder.
    This class is with the backbone and detector head, while the attention_rpn_text_head 
    only has the detector head.
    This class is a RPN, only has one stage.
    """

    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.is_model_init = False
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'res4_roi_feats': [],
            #'res5_roi_feats': []
        }
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {}

    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img: Tensor) -> List[Tensor]:
        """Extract features of support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of support images, each item with shape
                 (N, C, H, W).
        """
        feats = self.support_backbone(img)
        if self.support_neck is not None:
            feats = self.support_neck(feats)
        return feats

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs) -> Dict:
        """Extract and save support features for model initialization.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.

        Returns:
            dict: A dict contains following keys:

                - `gt_labels` (Tensor): class indices corresponding to each
                    feature.
                - `res4_roi_feat` (Tensor): roi features of res4 layer.
                - `res5_roi_feat` (Tensor): roi features of res5 layer.
        """
        self.is_model_init = False
        # extract support template features will reset `is_model_init` flag
        assert gt_bboxes is not None and gt_labels is not None, \
            'forward support template require gt_bboxes and gt_labels.'
        assert len(gt_labels) == img.size(0), \
            'Support instance have more than two labels'

        #feats = self.extract_support_feat(img)
        #rois = bbox2roi([bboxes for bboxes in gt_bboxes])
        #res4_roi_feat = self.rpn_head.extract_roi_feat(feats, rois)
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        #self._forward_saved_support_dict['res4_roi_feats'].append(
        #    res4_roi_feat)

        return {
            'gt_labels': gt_labels,
            #'res4_roi_feats': res4_roi_feat,
            #'res5_roi_feats': res5_roi_feat
        }
        #pass

    def model_init(self) -> None:
        """process the saved support features for model initialization."""
        self.inference_support_dict.clear()
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        # used for attention rpn head
        #res4_roi_feats = torch.cat(
        #    self._forward_saved_support_dict['res4_roi_feats'])
        class_ids = set(gt_labels.data.tolist())
        for class_id in class_ids:
            self.inference_support_dict[class_id] = {}
        # set the init flag
        self.is_model_init = True
        # clear support dict
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()
        #pass

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        #assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if (self.inference_support_dict == {}) or (not self.is_model_init):
            # process the saved support features
            self.model_init()

        results_dict = {}
        query_feats = self.extract_feat(img)
        for class_id in self.inference_support_dict.keys():
            #proposal_list.shape 1 <class 'torch.Tensor'> torch.Size([100, 5])
            proposal_list = self.rpn_head.simple_test(
                    query_feats, class_id, img_metas, rescale=rescale)
            # if rescale:
            #     for proposals, meta in zip(proposal_list, img_metas):
            #         proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])
            results_dict[class_id] = proposal_list
            
        # results torch.Size([6000, 5]), in rpn torch.Size([1000, 5])
        results = torch.cat([
            results_dict[i][0] for i in sorted(results_dict.keys())
            if len(results_dict[i])], dim=0)
        return [results.cpu().numpy()]

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        query_img = query_data['img']
        #support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        #support_feats = self.extract_support_feat(support_img)

        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
        if self.rpn_with_support:
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                query_feats,
                query_gt_bboxes=query_data['gt_bboxes'],
                query_img_metas=query_data['img_metas'],
                query_gt_labels=query_data['gt_labels'],
                query_gt_bboxes_ignore=query_data.get(
                    'gt_bboxes_ignore', None),
                proposal_cfg=proposal_cfg)
        else:
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                query_feats,
                copy.deepcopy(query_data['img_metas']),
                copy.deepcopy(query_data['gt_bboxes']),
                gt_labels=None,
                gt_bboxes_ignore=copy.deepcopy(
                    query_data.get('gt_bboxes_ignore', None)),
                proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        return losses