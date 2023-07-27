# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner import force_fp32
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi, images_to_levels, multi_apply
from mmdet.models import RPNHead
from mmdet.models.builder import HEADS, build_roi_extractor
from torch import Tensor, negative
import torch.nn as nn
import random

from ..utils import build_aggregator


@HEADS.register_module()
class AttentionRPNTextHead(RPNHead):
    """
    Here we implement the our version of the RPN in attentionRPN detector.
    We replace the learnable vision class prototype with text embedding from CLIP.
    This attention_rpn_text_head only has the detector head.
    """

    def __init__(self,
                 num_support_ways: int,
                 num_support_shots: int,
                 aggregation_layer: Dict = dict(
                     type='AggregationLayer',
                     aggregator_cfgs=[
                         dict(
                             type='DepthWiseCorrelationAggregator',
                             in_channels=512,
                             with_fc=False)
                     ]),
                 roi_extractor: Dict = dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=14, sampling_ratio=0),
                     out_channels=1024,
                     featmap_strides=[16]),
                 clip_dim=512,
                 backbone_feat_out_channels=1024,
                 fg_vec_cfg=None,
                 num_classes=80,
                 normalize_img_feat=False,
                 normalize_text_feat=False,
                 linear_mapping=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        assert roi_extractor is not None, \
            'missing config of roi_extractor.'
        assert aggregation_layer is not None, \
            'missing config of aggregation_layer.'
        self.aggregation_layer = \
            build_aggregator(copy.deepcopy(aggregation_layer))
        self.roi_extractor = \
            build_roi_extractor(copy.deepcopy(roi_extractor))
        self.clip_dim = clip_dim
        self.backbone_feat_out_channels = backbone_feat_out_channels
        self.linear_mapping = linear_mapping
        
        if self.linear_mapping == None:
            # add one extra convolution layer to map the feature map to clip dimension
            self.map_to_clip_dim = nn.Conv2d(self.backbone_feat_out_channels, self.clip_dim, 1, bias=False)
        elif self.linear_mapping == 'on_query':
            self.map_to_clip_dim = nn.Linear(self.backbone_feat_out_channels, self.clip_dim, bias=True)
        elif self.linear_mapping == 'on_both':
            self.map_to_clip_dim = nn.Linear(self.backbone_feat_out_channels, self.clip_dim, bias=True)
            self.support_mapping = nn.Linear(self.clip_dim, self.clip_dim, bias=True)
            
        # load the text embeddings
        self.fg_vec_cfg = fg_vec_cfg
        load_value = torch.load(self.fg_vec_cfg.load_path)
        self.load_value = load_value.cuda()
        # fix the text embedding
        self.load_value.require_grad = False
        #print('in init, self.load_value.require_grad', self.load_value.require_grad)
        self.num_classes = num_classes
        self.normalize_img_feat = normalize_img_feat
        self.normalize_text_feat = normalize_text_feat

    def extract_roi_feat(self, feats: List[Tensor], rois: Tensor) -> Tensor:
        """Forward function.

        Args:
            feats (list[Tensor]): Input features with shape (N, C, H, W).
            rois (Tensor): with shape (m, 5).

         Returns:
            Tensor: RoI features with shape (N, C, H, W).
        """
        return self.roi_extractor(feats, rois)

    def forward_train(self,
                      query_feats: List[Tensor],
                      query_gt_bboxes: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      proposal_cfg: Optional[ConfigDict] = None,
                      **kwargs) -> Tuple[Dict, List[Tuple]]:
        """Forward function in training phase.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W)..
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            query_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                query image, each item with shape (num_gts, 4).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            support_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                support image, each item with shape (num_gts, 4).
            query_gt_bboxes_ignore (list[Tensor]): List of ground truth bboxes
                to be ignored of query image with shape (num_ignored_gts, 4).
                Default: None.
            proposal_cfg (:obj:`ConfigDict`): Test / postprocessing
                configuration. if None, test_cfg would be used. Default: None.
        len(query_gt_bboxes) 2, 
        query_gt_bboxes[0].shape torch.Size([5, 4]) 
        len(query_gt_labels) 2, 
        query_gt_labels[0].shape torch.Size([5]) 
        query_gt_labels[0] tensor([8, 8, 8, 8, 8])
        
        len(query_gt_bboxes) 2, 
        query_gt_bboxes[0].shape torch.Size([2, 4]) 
        len(query_gt_labels) 2, 
        query_gt_labels[0].shape torch.Size([2]) 
        query_gt_labels[0] tensor([53, 53])
        Returns:
            tuple: loss components and proposals of each image.

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - proposal_list (list[Tensor]): Proposals of each image.
        """
        batch_size = len(query_img_metas)
        neg_sample_num = self.num_support_ways - 1
        query_feat = query_feats[0]
        query_feat_input, support_feat_input = self.preprocess_feats(query_feat)
        
        # generate the positve feat
        # select the needed text embedding
        # for the image i the cate_idx should be query_gt_labels[i][0]
        pos_pair_feats = [
            self.aggregation_layer(
                query_feat=query_feat_input[i],
                support_feat=support_feat_input[query_gt_labels[i][0]].unsqueeze(dim=0)
                )[0]
            for i in range(batch_size)
        ]
        # generate the negative feat
        # random sameple one embedding which is not needed not the one: query_gt_labels[i][0]
        # generate the idx first
        random_idx = []
        for i in range(batch_size):
            except_value = query_gt_labels[i][0]
            for j in range(neg_sample_num):
                idx_per_image = []
                while True:
                    rand_val = random.randint(0, self.num_classes-1)
                    if rand_val != except_value:
                        idx_per_image.append(rand_val)
                        break
                random_idx.append(idx_per_image)
        
        neg_pair_feats = [
            self.aggregation_layer(
                query_feat=query_feat_input[i],
                support_feat=support_feat_input[random_idx[i][j]].unsqueeze(dim=0)
                )[0]
            for i in range(batch_size)
            for j in range(neg_sample_num)
        ]
        
        # input features for losses: [pos_pair_feats, neg_pair_feats]
        # pair_flags are used to set all the gt_label from negative pairs to
        # bg classes in losses. True means positive pairs and False means
        # negative pairs

        # add positive pairs
        pair_flags = [True for _ in range(batch_size)]
        repeat_query_img_metas = copy.deepcopy(query_img_metas)
        repeat_query_gt_bboxes = copy.deepcopy(query_gt_bboxes)
        # repeat the query_img_metas and query_gt_bboxes to match
        # the order of positive and negative pairs
        for i in range(batch_size):
            repeat_query_img_metas.extend([query_img_metas[i]] * neg_sample_num)
            repeat_query_gt_bboxes.extend([query_gt_bboxes[i]] * neg_sample_num)
            # add negative pairs
            pair_flags.extend([False] * neg_sample_num)
        outs = self([torch.cat(pos_pair_feats + neg_pair_feats)])
        loss_inputs = outs + (repeat_query_gt_bboxes, repeat_query_img_metas)
        losses = self.loss(
            *loss_inputs,
            gt_bboxes_ignore=query_gt_bboxes_ignore,
            pair_flags=pair_flags)
        proposal_list = self.get_bboxes(
            *outs, img_metas=repeat_query_img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores: List[Tensor],
             bbox_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             img_metas: List[Dict],
             gt_labels: Optional[List[Tensor]] = None,
             gt_bboxes_ignore: Optional[List[Tensor]] = None,
             pair_flags: Optional[List[bool]] = None) -> Dict:
        """Compute losses of rpn head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
                Default: None.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss. Default: None
            pair_flags (list[bool]): Indicate predicted result is from positive
                pair or negative pair with shape (N). Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        # get anchors and training targets
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # get the indices of negative pairs
        neg_idxes = [not f for f in pair_flags]
        num_pos_from_neg_pairs = 0
        # all the gt_labels in negative pairs will be set to background
        for lvl in range(len(labels_list)):
            num_pos_from_neg_pairs += (
                labels_list[lvl][neg_idxes] == 0).sum().item()
            labels_list[lvl][neg_idxes] = 1
            bbox_weights_list[lvl][neg_idxes] = 0
        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        else:
            num_total_samples = num_total_pos - num_pos_from_neg_pairs
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single Tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def simple_test(self,
                    query_feats: List[Tensor],
                    class_id: int,
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[Tensor]:
        """Test function without test time augmentation.

        Args:
            query_feats (list[Tensor]): List of query features, each item with
                shape(N, C, H, W).
            support_feat (Tensor): Support features with shape (N, C, H, W).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results.
                Default: False.

        Returns:
            List[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        # fuse support and query features
        query_feat = query_feats[0]
        query_feat_input, support_feat_input = self.preprocess_feats(query_feat)

        # default test batch size is 1
        feats = self.aggregation_layer(
            query_feat=query_feat_input[0], support_feat=support_feat_input[class_id].unsqueeze(dim=0))
        proposal_list = self.simple_test_rpn(feats, query_img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, query_img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])

        return proposal_list


    def preprocess_feats(self, query_feat):
        # map the query feature
        if self.linear_mapping == None:
            query_feat = self.map_to_clip_dim(query_feat)
        else:
            # permute the dim
            query_feat = query_feat.permute([0,2,3,1])
            query_feat = self.map_to_clip_dim(query_feat)
            query_feat = query_feat.permute([0,3,1,2])
            #print("query_feat.shape", query_feat.shape)
        
        ## normalize the query feat
        if self.normalize_img_feat:
            query_feat_input = [(ele / ele.norm(dim=0, keepdim=True)).unsqueeze(0) for ele in query_feat]
        else:
            query_feat_input = [ele.unsqueeze(0) for ele in query_feat]        
        
        # map the support feature
        support_feat_input = self.load_value
        if self.linear_mapping == "on_both":
            support_feat_input = self.support_mapping(support_feat_input)
            #print('support feat shape', support_feat_input.shape)

        # normalize the support feat
        if self.normalize_text_feat:
            support_feat_input = support_feat_input / support_feat_input.norm(dim=-1, keepdim=True)
        # reshape the text embedding
        support_feat_input = support_feat_input.unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        return query_feat_input, support_feat_input