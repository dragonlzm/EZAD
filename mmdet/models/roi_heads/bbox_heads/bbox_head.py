# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import math
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer


@HEADS.register_module()
class BBoxHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 reg_with_cls_embedding=False,
                 combine_reg_and_cls_embedding='cat',
                 per_bbox_reg_weight=False,
                 init_cfg=None):
        super(BBoxHead, self).__init__(init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False
        self.reg_with_cls_embedding = reg_with_cls_embedding
        self.combine_reg_and_cls_embedding = combine_reg_and_cls_embedding

        self.bbox_coder = build_bbox_coder(bbox_coder)
        if loss_cls['type'] == 'my_focal_loss':
            self.loss_cls = self.my_focal_loss
            self.cls_loss_weight = 0.2
        else:
            self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        
        # add for use sigmoid loss
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=in_channels,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=in_channels,
                out_features=out_dim_reg)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]
                
        self.from_idx_to_cate_id = {65: [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 
                                         20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
                                         35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 
                                         53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 
                                         72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 
                                         86, 87, 90],
                                    48: [1, 2, 3, 4, 7, 8, 9, 15, 16, 19, 20, 23, 24, 25, 
                                         27, 31, 33, 34, 35, 38, 42, 44, 48, 50, 51, 52, 
                                         53, 54, 55, 56, 57, 59, 60, 62, 65, 70, 72, 73, 
                                         74, 75, 78, 79, 80, 82, 84, 85, 86, 90]}
        self.per_bbox_reg_weight = per_bbox_reg_weight

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, img_meta, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            # if we use the per bbox weight
            if self.per_bbox_reg_weight is not False:
                img_h, img_w, _ = img_meta['img_shape']
                bbox_w = pos_gt_bboxes[:, 2] - pos_gt_bboxes[:, 0]
                bbox_h = pos_gt_bboxes[:, 3] - pos_gt_bboxes[:, 1]
                if self.per_bbox_reg_weight == 'sqrt':
                    img_factor = math.sqrt(img_h * img_w)
                    bbox_factor = torch.sqrt(bbox_w * bbox_h)
                elif self.per_bbox_reg_weight == 'log':
                    img_factor = math.log(img_h * img_w + 1.1)
                    bbox_factor = torch.log(bbox_w * bbox_h + 1.1)
                else:
                    img_factor = img_h * img_w
                    bbox_factor = bbox_w * bbox_h
                weight = - (bbox_factor / img_factor)
                weight = torch.exp(weight)
                # normalize over all the weight
                normalize_factor = pos_bbox_targets.shape[0] / torch.sum(weight)
                weight *= normalize_factor
                #print('bbox_factor', bbox_factor, 'weight', weight)
                weight = weight.unsqueeze(dim=-1).repeat([1, 4])
                
                bbox_weights[:num_pos, :] = weight
            else:
                bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            neg_weight = cfg.get('neg_weight', 1.0) if cfg is not None else 1.0
            label_weights[-num_neg:] = neg_weight
        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    img_metas=None):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        #print('img_metas', img_metas)
        if img_metas is None:
            img_metas = [None for res in sampling_results]
            
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            img_metas,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def my_focal_loss(self, inputs, targets, label_weights, gamma=0.5, reduction="mean", avg_factor=None, reduction_override=None):
        """Inspired by RetinaNet implementation"""
        if targets.numel() == 0 and reduction == "mean":
            return input.sum() * 0.0  # connect the gradient
        
        # focal scaling
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = F.softmax(inputs, dim=-1)
        p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
        loss = ce_loss * ((1 - p_t) ** gamma)

        # bg loss weight
        if self.cls_loss_weight is not None:
            loss_weight = torch.ones(loss.size(0)).to(p.device)
            loss_weight[targets == self.num_classes] = self.cls_loss_weight
            loss = loss * loss_weight

        if reduction == "mean":
            loss = loss.mean()

        return loss

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            #avg_factor = 1024.0
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic or self.reg_with_cls_embedding:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   proposal_obj_score,
                   dist_cls_score,
                   rescale=False,
                   cfg=None,
                   img_metas=None,
                   bbox_save_path_root=None,
                   clip_infer_bbox=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                Fisrt tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """
        # rois.shape torch.Size([1000, 5]) cls_score.shape torch.Size([1000, 66]) bbox_pred.shape torch.Size([1000, 4])
        if hasattr(self, 'use_pregenerate_proposal_and_score') and self.use_pregenerate_proposal_and_score is not None:
            all_proposals = []
            all_scores = []
            for i, img_meta in enumerate(img_metas): 
                image_name = img_meta['filename'].split('/')[-1]
                proposal_file_name = os.path.join(self.use_pregenerate_proposal_and_score, (image_name + '.json'))
                proposal_res_file_name = os.path.join(self.use_pregenerate_proposal_and_score, (image_name + 'pred_score' + '.json'))
                # load the bbox and the score
                bbox_content = json.load(open(proposal_file_name))
                score_content = json.load(open(proposal_res_file_name))
                # reshape the bbox to torch.Size([1, 4, 100, 152])
                # (1000, 4)
                proposal_per_img = torch.tensor(bbox_content['box']).cuda()
                proposal_per_img *= proposal_per_img.new_tensor(img_meta['scale_factor'])
                roi_idx = torch.full([proposal_per_img.shape[0], 1], i).cuda()
                proposal_per_img = torch.cat([roi_idx, proposal_per_img], dim=-1)

                all_proposals.append(proposal_per_img)
                # reshape the score to ([1, 4, 100, 152])
                # (1000, 65) =>(1000, 66)
                score_per_img = torch.tensor(score_content['score']).cuda()
                bg_score = torch.full([score_per_img.shape[0], 1], -1e5).cuda()
                score_per_img = torch.cat([score_per_img, bg_score], dim=-1)
                all_scores.append(score_per_img)
                
            all_proposals = torch.cat(all_proposals, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_bbox_pred = torch.zeros(all_proposals.shape[0], 4).cuda()
            #print('all_proposals', all_proposals.shape, all_proposals, 'all_scores', all_scores.shape, all_scores, 'all_bbox_pred', all_bbox_pred.shape)
            rois=all_proposals
            cls_score=all_scores
            bbox_pred=all_bbox_pred
        
        # for testing
        if hasattr(self, 'filter_base_cate') and self.filter_base_cate != None:
            if self.custom_cls_channels:
                bg_idx = self.num_classes + 1
            else:
                bg_idx = self.num_classes
            
            ## the filtering procedure
            # BS > BGS and NS 
            max_idx = torch.max(cls_score, dim=1)[1]
            novel_bg_idx = (max_idx <= bg_idx)
            
            # BS > BGS
            # base_max_value = torch.max(scores[:, bg_idx+1:], dim=1)[0]
            # bg_value = scores[:, bg_idx]
            # novel_bg_idx = (base_max_value <= bg_value)
            
            # BS > NS
            # base_max_value = torch.max(scores[:, bg_idx+1:], dim=1)[0]
            # novel_max_value = torch.max(scores[:, :bg_idx], dim=1)[0]
            # novel_bg_idx = (base_max_value <= novel_max_value)      
            
            # # BS > threshold
            # base_max_value = torch.max(scores[:, bg_idx+1:], dim=1)[0]
            # novel_bg_idx = (base_max_value <= 0.7)                     
            
            cls_score = cls_score[novel_bg_idx]
            cls_score = cls_score[:, :bg_idx+1]
            if dist_cls_score != None:
                dist_cls_score = dist_cls_score[novel_bg_idx]
                dist_cls_score = dist_cls_score[:, :bg_idx]
            if proposal_obj_score != None:
                proposal_obj_score = proposal_obj_score[novel_bg_idx]
        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        
        # merge the rpn objectness score with the confidence score
        if proposal_obj_score != None:
            #print(scores.shape, proposal_obj_score.shape)
            proposal_obj_score = proposal_obj_score.unsqueeze(dim=-1).repeat(1, self.num_classes)
            #print('before multiple', scores)
            scores[:, :self.num_classes] *= proposal_obj_score
            #print('before sqrt', scores)
            scores[:, :self.num_classes] = torch.pow(scores[:, :self.num_classes], 0.5)
            #print('after sqrt', scores)
         
        ## if merging the score we completely following the google implemetation
        ## which remove the bg vector before doing the softmax
        ## therefore the above softmax function do not be used 
        if dist_cls_score != None:
            ### the preliminary version of merging the score
            ### in VILD it merge the score after softmax
            # if is not using the bg vector the cls_score and dist_cls_score will have the same size
            if not self.use_bg_vector:
                dist_cls_score = F.softmax(dist_cls_score, dim=-1)
                scores = torch.pow(scores, 0.5) * torch.pow(dist_cls_score, 0.5)
            else:
                # if do not use the additional base categories for filtering
                # the size of the cls_score [num_cls + 1,clip_dim], 
                # dist_cls_score will be [num_cls,clip_dim]
                cls_score_fg = cls_score[:, :self.num_classes]
                cls_score_bg = cls_score[:, self.num_classes:]
                cls_score_fg = F.softmax(cls_score_fg, dim=-1)
                dist_cls_score = F.softmax(dist_cls_score, dim=-1)
                cls_score_fg = torch.pow(cls_score_fg, 0.5) * torch.pow(dist_cls_score, 0.5)
                scores = torch.cat([cls_score_fg, cls_score_bg], dim=-1)

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        # if clip_infer_bbox != None:
        #     print('clip_infer_bbox', clip_infer_bbox.shape, 'bboxes', bboxes.shape,
        #             'compare result', (False in (clip_infer_bbox == bboxes)),
        #             'clip_infer_bbox', clip_infer_bbox,
        #             'bboxes', bboxes)

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # for testing
            if hasattr(self, 'filter_base_cate') and self.filter_base_cate != None:
                bboxes = bboxes[novel_bg_idx]
            # add the empty bg prediction for the sigmoid_cls model
            # if self.use_sigmoid_cls:
            #     # Add a dummy background class to the backend when using sigmoid
            #     # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            #     # BG cat_id: num_class
            #     # the shape of score should be torch.Size([1000, 66])
            #     num_pred, num_classes = scores.shape
            #     padding = scores.new_zeros(num_pred, 1)
            #     scores = torch.cat([scores, padding], dim=-1)  
            
            # save the prediction before the NMS
            if bbox_save_path_root != None:
                if not os.path.exists(bbox_save_path_root):
                    os.makedirs(bbox_save_path_root)
                file_name = os.path.join(bbox_save_path_root, '.'.join(img_metas[0]['ori_filename'].split('.')[:-1]) + '.json')
               
                if not os.path.exists(file_name):
                    file = open(file_name, 'w')
                    fg_classes_num = scores.shape[-1] - 1
                    
                    max_score, max_idx = torch.max(scores[..., :fg_classes_num], dim=1, keepdim=True)
                    result = torch.cat([bboxes, max_score], dim=1)
                    #print('result', result.shape)
                    # the needed format should like this: 
                    # {'image_id': 289343, 'bbox': [202.46621704101562, 234.70960998535156, 64.39511108398438, 199.79380798339844], 'score': 0.9922735095024109, 'category_id': 1}
                    # deal with the categories_id
                    need_cates_list = self.from_idx_to_cate_id[fg_classes_num]
                    #print('need_cates_list',need_cates_list)
                    need_cates_list = torch.tensor(need_cates_list)
                    all_pred_cates = need_cates_list[max_idx]
                    
                    #result_json = {'image_id':int(img_metas[0]['ori_filename'].split('.')[0].strip('0')), 'score':result.tolist()}
                    result_json = {'image_id':int(img_metas[0]['ori_filename'].split('.')[0].strip('0')), 'score':result.tolist(), 'category_id':all_pred_cates.cpu().tolist()}
                    file.write(json.dumps(result_json))
                    file.close()

            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): Rois from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4) or
                (num_proposals, 5).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """

        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        max_shape = img_meta['img_shape']

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    def onnx_export(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    cfg=None,
                    **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed.
                Has shape (B, num_boxes, 5)
            cls_score (Tensor): Box scores. has shape
                (B, num_boxes, num_classes + 1), 1 represent the background.
            bbox_pred (Tensor, optional): Box energies / deltas for,
                has shape (B, num_boxes, num_classes * 4) when.
            img_shape (torch.Tensor): Shape of image.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """

        assert rois.ndim == 3, 'Only support export two stage ' \
                               'model to ONNX ' \
                               'with batch dimension. '
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class',
                                                 cfg.max_per_img)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)

        scores = scores[..., :self.num_classes]
        if self.reg_class_agnostic:
            return add_dummy_nms_for_onnx(
                bboxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img)
        else:
            batch_size = scores.shape[0]
            labels = torch.arange(
                self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)

            max_size = torch.max(img_shape)
            # Offset bboxes of each class so that bboxes of different labels
            #  do not overlap.
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets

            batch_dets, labels = add_dummy_nms_for_onnx(
                bboxes_for_nms,
                scores.unsqueeze(2),
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img,
                labels=labels)
            # Offset the bboxes back after dummy nms.
            offsets = (labels * max_size + 1).unsqueeze(2)
            # Indexing + inplace operation fails with dynamic shape in ONNX
            # original style: batch_dets[..., :4] -= offsets
            bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return batch_dets, labels
