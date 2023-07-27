# Copyright (c) OpenMMLab. All rights reserved.
from cgi import test
from pickle import FALSE
import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init, bias_init_with_prob)

from mmdet.core import bbox_mapping
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from ..builder import DETECTORS, build_backbone, build_head, build_neck, HEADS, build_loss
from .base import BaseDetector
from PIL import Image
import numpy as np
import math
import random
import os
import json
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from torch import nn

sigmoid_fun = nn.Sigmoid()

def get_points_single(featmap_size, dtype, device):
    """Get points of a single scale level."""
    h, w = featmap_size
    # First create Range with the default dtype, than convert to
    # target `dtype` for onnx exporting.
    x_range = torch.arange(w, device=device).to(dtype)
    y_range = torch.arange(h, device=device).to(dtype)
    y, x = torch.meshgrid(y_range, x_range)
    points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    return points


def _get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
    """Compute regression and classification targets for a single image."""
    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        return gt_labels.new_full((num_points,), num_classes), \
                gt_bboxes.new_zeros((num_points, 4))
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
    xs, ys = points[:, 0], points[:, 1]
    xs = xs[:, None].expand(num_points, num_gts)
    ys = ys[:, None].expand(num_points, num_gts)
    left = xs - gt_bboxes[..., 0]
    right = gt_bboxes[..., 2] - xs
    top = ys - gt_bboxes[..., 1]
    bottom = gt_bboxes[..., 3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
    return inside_gt_bbox_mask


def get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
    # this function return the per categories mask
    num_points = points.size(0)
    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
    gt_bboxes[:, 3] - gt_bboxes[:, 1])
    # TODO: figure out why these two are different
    # areas = areas[None].expand(num_points, num_gts)
    areas = areas[None].repeat(num_points, 1)
    
    total_iteration = 10
    gap_per_iter = gt_bboxes.shape[0] / total_iteration
    all_result = []
    for i in range(total_iteration):
        start = int(i * gap_per_iter)
        end = int((i+1) * gap_per_iter)
        #print('start:end', start, end)
        _inside_gt_bbox_mask = _get_target_single(gt_bboxes[start:end], gt_labels[start:end], points)
        #print(temp_assigned_label.shape)
        all_result.append(_inside_gt_bbox_mask)
    # inside_gt_bbox_mask will be a tensor([num_of_pixel, num_of_proposal]), a true/ false mask
    inside_gt_bbox_mask = torch.cat(all_result, dim=-1)
    inside_gt_bbox_mask = inside_gt_bbox_mask.permute([1,0])
    
    return inside_gt_bbox_mask    


@DETECTORS.register_module()
class ProposalSelectorV2(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 loss,
                 ranking_loss=None,
                 ranking_loss_only=None,
                 input_dim=5,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 subset_num=10,
                 num_class=48,
                 infer_subset_num=20):
        super(ProposalSelectorV2, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.input_dim = input_dim
        self.loss = build_loss(loss)
        if ranking_loss is not None:
            if ranking_loss == 'TripletMarginLoss':
                self.ranking_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            else:
                self.ranking_loss = build_loss(ranking_loss)
        else:
            self.ranking_loss = None
        self.ranking_loss_only = ranking_loss_only
        self.iou_calculator = BboxOverlaps2D()
        
        # define the modules
        self.global_info_extractor = nn.Conv2d(1, 10, 3, padding=1)
        self.global_info_dim_reducer_1 = nn.Conv2d(11, 5, 3, padding=1)
        self.global_info_dim_reducer = nn.Conv2d(5, 1, 3, padding=1)
        #self.global_info_extractor = nn.Conv2d(1, 10, 1)
        #self.global_info_dim_reducer = nn.Conv2d(11, 1, 1)
        self.select_idx = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 17, 18, 19, 20, 22, 24, 25, 26, 28, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62, 64]).cuda()
        # select a subset to train in each iteration
        self.subset_num = subset_num
        self.infer_subset_num = infer_subset_num
        self.num_class = num_class
        #for m in self.global_info_extractor.modules():
        #if hasattr(m, 'weight') and m.weight.dim() > 1:
        if loss.type == 'L1Loss':
            xavier_init(self.global_info_extractor.weight, distribution='uniform')
            xavier_init(self.global_info_dim_reducer.weight, distribution='uniform')    
            #bias_init = bias_init_with_prob(0.01)
            #nn.init.constant_(self.global_info_extractor.bias, bias_init)
            #nn.init.constant_(self.global_info_dim_reducer.bias, bias_init)
            nn.init.constant_(self.global_info_extractor.bias, 0.0)
            nn.init.constant_(self.global_info_dim_reducer.bias, 0.0)
        elif loss.type == 'MSELoss':
            print('MSE')
            #xavier_init(self.global_info_extractor.weight, distribution='uniform')
            #xavier_init(self.global_info_dim_reducer.weight, distribution='uniform')
            nn.init.constant_(self.global_info_extractor.weight, 0.0)
            nn.init.constant_(self.global_info_dim_reducer.weight, 0.0)
            
            #bias_init = bias_init_with_prob(0.01)
            #nn.init.constant_(self.global_info_extractor.bias, bias_init)
            #nn.init.constant_(self.global_info_dim_reducer.bias, bias_init)
            nn.init.constant_(self.global_info_extractor.bias, 0.0)
            nn.init.constant_(self.global_info_dim_reducer.bias, 0.0)


    # def init_weights(self):
    #     xavier_init(self.global_info_extractor.weight, distribution='uniform')
    #     xavier_init(self.global_info_dim_reducer.weight, distribution='uniform')    
    #     #bias_init = bias_init_with_prob(0.01)
    #     #nn.init.constant_(self.global_info_extractor.bias, bias_init)
    #     #nn.init.constant_(self.global_info_extractor.bias, bias_init)
    
    def extract_feat(self):
        pass
    
    def forward_dummy(self, img):
        pass

    def find_the_gt_for_proposal(self, proposal_bboxes, gt_bboxes, gt_labels):
        
        '''
        base on the prediction categories
        the proposal_bboxes (list[Tensor])
        the gt_bboxes (list[Tensor])
        '''
        if self.training:
            num_class = 48
        else:
            num_class = 65
        
        all_pred_target = []
        for proposal, gt_bbox, gt_label_per_img in zip(proposal_bboxes, gt_bboxes, gt_labels):
            # calculate the iou between all the proposal and all the gt bboxes, the shape should be [num_proposal, num_gt]
            real_iou = self.iou_calculator(proposal, gt_bbox)
            num_proposal = proposal.shape[0]
            assigned_res = []
            for i in range(num_class):
                if i not in gt_label_per_img:
                    temp = torch.zeros(num_proposal, 1).cuda()
                    assigned_res.append(temp)
                else:
                    idx = (gt_label_per_img == i)
                    selected_iou = real_iou[..., idx]
                    selected_iou, _ = torch.max(selected_iou, dim=-1, keepdim=True)
                    assigned_res.append(selected_iou)
            assigned_res = torch.cat(assigned_res, dim=-1)
            #print('assigned_res', assigned_res.shape)
            all_pred_target.append(assigned_res)
    
        return all_pred_target
    
    def forward_mask(self, per_cate_masks, img_metas):
        #print('per_cate_masks', [ele.shape for ele in per_cate_masks])
        #print('img_metas', [ele['img_shape'] for ele in img_metas])
        result = []
        for mask, meta in zip(per_cate_masks, img_metas):
            h,w,c = meta['img_shape']
            #print('mask', mask.shape)
            cls_num, _ = mask.shape
            mask = mask.reshape(cls_num, 1, h, w)
            # use conv 
            gobal_info = self.global_info_extractor(mask)
            # gobal average pooling
            gobal_info = torch.mean(gobal_info, dim=[-2,-1], keepdim=True)
            #print('gobal_info', gobal_info.shape, gobal_info)
            # extend the global info
            #gobal_info = gobal_info.unsqueeze(dim=-1)
            gobal_info = gobal_info.repeat([1, 1, h, w])
            mask = torch.cat([mask, gobal_info], dim=1)
            #mask = gobal_info + mask
            #print('before mask', mask.shape, mask, 'gobal_info', gobal_info)
            mask = self.global_info_dim_reducer_1(mask)
            mask = self.global_info_dim_reducer(mask)
            #print('after mask', mask.shape)
            result.append(mask)
        #print('result', [ele.shape for ele in result])
        return result
    
    def generate_per_cate_mask(self, inside_gt_bbox_mask, gt_labels, num_classes):
        mask_of_all_cates = []
        for i in range(num_classes):
            cate_matched_idx = (gt_labels == i)
            # selected_masks tensor(num_of_mask, num_of_pixel)
            selected_masks = inside_gt_bbox_mask[cate_matched_idx]
            mask_per_cate = torch.sum(selected_masks, dim=0)
            mask_per_cate = mask_per_cate.float()
            # if selected_masks.shape[0] == 0:
            #     print('in special mask_per_cate', mask_per_cate.shape)
            # print('in outside', mask_per_cate.shape, mask_per_cate)
            # normalize the mask with the max value
            max_val = torch.max(mask_per_cate)
            max_val = max_val if max_val != 0 else 1.0
            mask_per_cate /= max_val
            mask_of_all_cates.append(mask_per_cate.unsqueeze(dim=0))

        mask_of_all_cates = torch.cat(mask_of_all_cates, dim=0)
        return mask_of_all_cates

    def cal_final_pred(self, now_bbox_mask, now_clip_pred, per_cate_masks_per_img):
        #print('now_bbox_mask', now_bbox_mask.shape, 'now_clip_pred', now_clip_pred.shape, 'per_cate_masks_per_img', per_cate_masks_per_img.shape)
        now_clip_pred = now_clip_pred.unsqueeze(dim=1)
        now_clip_pred = now_clip_pred.repeat([1, now_bbox_mask.shape[-1], 1])
        now_clip_pred[now_bbox_mask==False] = 0
        #final_res = now_clip_pred[now_bbox_mask]
        # torch.Size([100, 257920, 65]) check none empty
        #print('final_res', now_clip_pred.shape, now_clip_pred)
        #break
        # do the multipication(non matrix mul, is row-based multiplication)
        now_clip_pred = now_clip_pred.permute([0,2,1])
        # per_cate_masks_per_img torch.Size([48, 1, 640, 403])
        per_cate_masks_per_img = per_cate_masks_per_img.reshape([per_cate_masks_per_img.shape[0], -1])
        
        #print('now_clip_pred', now_clip_pred.shape, 'per_cate_masks_per_img', per_cate_masks_per_img.shape)
        # try normalization
        #clip_pred_non_empty = (torch.sum(now_clip_pred, dim=-1) != 0)
        #per_cate_masks_non_empty = (torch.sum(per_cate_masks_per_img, dim=-1) != 0)
        
        #now_clip_pred[clip_pred_non_empty] = now_clip_pred[clip_pred_non_empty] / now_clip_pred[clip_pred_non_empty].norm(dim=-1, keepdim=True)
        #per_cate_masks_per_img[per_cate_masks_non_empty] = per_cate_masks_per_img[per_cate_masks_non_empty] / per_cate_masks_per_img[per_cate_masks_non_empty].norm(dim=-1, keepdim=True)
        
        batch_predict = now_clip_pred * per_cate_masks_per_img
        batch_predict = batch_predict.sum(dim=-1)
        non_zero_idx = (batch_predict != 0)
        batch_predict[non_zero_idx] = sigmoid_fun(batch_predict[non_zero_idx])
        # normalize the score to the 0-1
        #non_zero_idx = (batch_predict != 0)
        #batch_predict[non_zero_idx] += 1
        #batch_predict[non_zero_idx] /= 2
        #print('batch_predict', batch_predict.shape, batch_predict)
        return batch_predict

    def calculate_the_ranking_samples(self, proposal_bboxes, clip_pred_labels, gt_bboxes, gt_labels):
        # this function aims to calculate the postive and negative sample for each gt bboxes
        # the return format should list[list[tuple(per_cate_mask_idx, pos_proposal_idx, neg_proposal_idx)]]
        # each tuple the infomation per gt bboxes, each inner list contain the info per image
        result_of_all_imgs = []
        for proposal_per_img, pred_label_per_img, gt_bbox_per_img, gt_label_per_img in zip(proposal_bboxes, clip_pred_labels, gt_bboxes, gt_labels):
            # calculate the iou between all the proposal and all the gt bboxes, the shape should be 
            # #[num_proposal, num_gt]
            real_iou = self.iou_calculator(proposal_per_img, gt_bbox_per_img)
            result_per_img = []
            # calculate the for each gt bboxes
            for i, (gt_bbox, gt_label) in enumerate(zip(gt_bbox_per_img, gt_label_per_img)):
                # [num_proposal, 1] => [num_proposal]
                current_gt_iou = real_iou[:, i]
                # if the max value is zero skip
                max_iou_val = torch.max(current_gt_iou)
                if max_iou_val == 0:
                    continue
                # select the proposals with same predicted categories
                need_proposal_idx = (pred_label_per_img == gt_label)
                number_need_proposal_idx = need_proposal_idx.nonzero().reshape(-1)
                # len(number_need_proposal_idx) == 100 then shape(selected_iou) [100, ]
                selected_iou = current_gt_iou[need_proposal_idx]
                # if there is no more remining gt bbox
                if selected_iou.shape[0] == 0:
                    continue
                # check whether the max number can select
                topk_num = selected_iou.shape[0] if selected_iou.shape[0] < 5 else 5
                # if top K number is 1
                if topk_num == 1:
                    #print('number_need_proposal_idx', number_need_proposal_idx)
                    res_idx = (i, number_need_proposal_idx[0], -1) 
                else:
                    #print('number_need_proposal_idx', number_need_proposal_idx)
                    _, topk_iou_per_gt_idx = torch.topk(selected_iou, topk_num, dim=0)
                    #print('topk_iou_per_gt_idx', topk_iou_per_gt_idx)
                    pos_proposal_idx = number_need_proposal_idx[topk_iou_per_gt_idx[0]]
                    # randomly select the negative idx from the topk result
                    rand_idx = torch.randint(1, topk_num, (1,))
                    #print('rand_idx', rand_idx)
                    neg_proposal_idx = number_need_proposal_idx[topk_iou_per_gt_idx[rand_idx[0]]]
                    res_idx = (gt_label, pos_proposal_idx, neg_proposal_idx)
                #print('res_idx', res_idx)
                result_per_img.append(res_idx)
            result_of_all_imgs.append(result_per_img)
                
        return result_of_all_imgs

    def collect_ranking_loss_sample(self, ranking_samples, per_cate_masks, per_proposal_area_masks, proposal_clip_score):
        anchor = []
        pos = []
        neg = []
        for result_per_img, per_cate_masks_per_img, per_proposal_area_masks_per_img, proposal_clip_score_per_img in zip(ranking_samples, per_cate_masks, per_proposal_area_masks, proposal_clip_score):
            anchor_per_img = []
            pos_per_img = []
            neg_per_img = []
            for gt_label, pos_proposal_idx, neg_proposal_idx in result_per_img:
                # handle the anchor mask
                anchor_mask = per_cate_masks_per_img[gt_label]
                anchor_per_img.append(anchor_mask.reshape(1, -1))
                # handle the pos mask
                pos_area_mask = per_proposal_area_masks_per_img[pos_proposal_idx]
                pos_clip_pred = proposal_clip_score_per_img[pos_proposal_idx]
                pos_vec = torch.zeros(pos_area_mask.shape).cuda()
                pos_value = torch.max(pos_clip_pred)
                pos_vec[pos_area_mask] = pos_value
                #if len(pos_vec.shape) == 2:
                #    print('per_proposal_area_masks_per_img', per_proposal_area_masks_per_img.shape, 'pos_proposal_idx', pos_proposal_idx, 'pos_area_mask', pos_area_mask.shape, 'pos_clip_pred', pos_clip_pred.shape, 'pos_vec', pos_vec.shape, 'pos_value', pos_value.shape)
                pos_per_img.append(pos_vec.unsqueeze(dim=0))
                # handle the neg mask
                neg_area_mask = per_proposal_area_masks_per_img[neg_proposal_idx]
                if neg_proposal_idx == -1:
                    neg_vec = torch.zeros(neg_area_mask.shape).cuda()
                else:
                    neg_clip_pred = proposal_clip_score_per_img[neg_proposal_idx]
                    neg_vec = torch.zeros(neg_area_mask.shape).cuda()
                    neg_value = torch.max(neg_clip_pred)
                    neg_vec[neg_area_mask] = neg_value
                neg_per_img.append(neg_vec.unsqueeze(dim=0))
                #print('per_cate_masks_per_img', per_cate_masks_per_img.shape, 'anchor_mask', anchor_mask.shape, 'pos_vec', pos_vec.shape, 'neg_vec', neg_vec.shape)
            if len(anchor_per_img) != 0:
                anchor_per_img = torch.cat(anchor_per_img, dim=0)
                pos_per_img = torch.cat(pos_per_img, dim=0)
                neg_per_img = torch.cat(neg_per_img, dim=0)
                anchor.append(anchor_per_img)
                pos.append(pos_per_img)
                neg.append(neg_per_img)
        return anchor, pos, neg
    
    def forward_train(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_labels=None, 
                    proposal_bboxes=None,
                    proposal_clip_score=None):
        # temp handle for the using 65 prediction
        #selected_proposal_clip_score = [ele[..., self.select_idx] for ele in proposal_clip_score]
        # prepare the clip predicted label
        #clip_pred_labels = [torch.max(ele, dim=-1)[1] for ele in selected_proposal_clip_score]
        clip_pred_labels = [torch.max(ele, dim=-1)[1] for ele in proposal_clip_score]

        # obtain the per proposal area mask
        # first we get the area mask for each proposal
        per_proposal_area_masks = []
        for meta_per_img, proposal_per_img, clip_score_per_img, clip_pred_labels_per_img in zip(img_metas, proposal_bboxes, proposal_clip_score, clip_pred_labels):
            h,w,c = meta_per_img['img_shape']
            all_points = get_points_single((h,w), torch.float16, torch.device('cuda'))
            per_img_per_proposal_area_mask = get_target_single(proposal_per_img, clip_pred_labels_per_img, all_points, num_classes=clip_score_per_img.shape[-1])
            per_proposal_area_masks.append(per_img_per_proposal_area_mask)
        #print('per_proposal_area_masks', [ele.shape for ele in per_proposal_area_masks])        
        
        # aggregate the area mask with the predicted categories to generate the per category mask
        per_cate_masks = []
        for per_proposal_area_masks_per_img, clip_pred_labels_per_img, clip_score_per_img in zip(per_proposal_area_masks, clip_pred_labels, proposal_clip_score):
            per_cate_masks_per_img = self.generate_per_cate_mask(per_proposal_area_masks_per_img, clip_pred_labels_per_img, clip_score_per_img.shape[-1])
            per_cate_masks.append(per_cate_masks_per_img)
        #print('per_cate_masks', [ele.shape for ele in per_cate_masks])
        
        # forward the per categories mask to obtain the final masks(generate mask)
        per_cate_masks = self.forward_mask(per_cate_masks, img_metas)
        
        # calculate the top1 target and the following target
        if self.ranking_loss is not None:
            ranking_samples = self.calculate_the_ranking_samples(proposal_bboxes, clip_pred_labels, gt_bboxes, gt_labels)
            # generate the training sample from the idxs
            anchor, pos, neg = self.collect_ranking_loss_sample(ranking_samples, per_cate_masks, per_proposal_area_masks, proposal_clip_score)
        
        loss_dict = dict()
        
        if not self.ranking_loss_only:
            # obtain a subset
            random_idx = torch.randperm(proposal_bboxes[0].shape[0])[:self.subset_num].cuda()
            per_proposal_area_masks = [ele[random_idx] for ele in per_proposal_area_masks]
            proposal_clip_score = [ele[random_idx] for ele in proposal_clip_score]
            proposal_bboxes = [ele[random_idx] for ele in proposal_bboxes]
            
            # obtain the per proposal per category mask, obtain the prediction
            all_preds = []
            for per_proposal_area_masks_per_img, proposal_clip_score_per_img, per_cate_masks_per_img in zip(per_proposal_area_masks, proposal_clip_score, per_cate_masks):
                #print('per_proposal_area_masks_per_img', per_proposal_area_masks_per_img.shape, 'proposal_clip_score_per_img', proposal_clip_score_per_img.shape)
                #print('proposal_bboxes', [ele.shape for ele in proposal_bboxes], 'proposal_clip_score', [ele.shape for ele in proposal_clip_score])
                prediction_per_img = self.cal_final_pred(per_proposal_area_masks_per_img, proposal_clip_score_per_img, per_cate_masks_per_img)
                all_preds.append(prediction_per_img)
            
            # find the target(for the random selected proposal), for each categories, calculate the target iou per categories
            pred_score_target = self.find_the_gt_for_proposal(proposal_bboxes, gt_bboxes, gt_labels)
            
            # concat the result 
            all_preds = torch.cat(all_preds, dim=0)
            pred_score_target = torch.cat(pred_score_target, dim=0)
            #pred_score_target = torch.zeros(all_preds.shape).cuda()

            # calculate the loss
            loss_value = self.loss(all_preds, pred_score_target)
            loss_dict['main_loss'] = loss_value
        
        if self.ranking_loss is not None:
            all_rank_loss = 0
            for anchor_per_img, pos_per_img, neg_per_img in zip(anchor,pos,neg):
                all_rank_loss += self.ranking_loss(anchor_per_img, pos_per_img, neg_per_img)
            all_rank_loss /= len(anchor)
            loss_dict['ranking_loss'] = all_rank_loss
            # we need to calculate the ranking loss
        return loss_dict

    def simple_test(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_labels=None, 
                    proposal_bboxes=None,
                    proposal_clip_score=None,
                    **kwargs):

        clip_pred_labels = [torch.max(ele, dim=-1)[1] for ele in proposal_clip_score]
        #print('type(img_metas)', type(img_metas), 'gt_labels', type(gt_labels), )
        img_metas = [img_metas]
        # obtain the per proposal area mask
        # first we get the area mask for each proposal
        per_proposal_area_masks = []
        for meta_per_img, proposal_per_img, clip_score_per_img, clip_pred_labels_per_img in zip(img_metas, proposal_bboxes, proposal_clip_score, clip_pred_labels):
            h,w,c = meta_per_img['img_shape']
            all_points = get_points_single((h,w), torch.float16, torch.device('cuda'))
            per_img_per_proposal_area_mask = get_target_single(proposal_per_img, clip_pred_labels_per_img, all_points, num_classes=clip_score_per_img.shape[-1])
            per_proposal_area_masks.append(per_img_per_proposal_area_mask)
        #print('per_proposal_area_masks', [ele.shape for ele in per_proposal_area_masks])        
        
        # aggregate the area mask with the predicted categories to generate the per category mask
        per_cate_masks = []
        for per_proposal_area_masks_per_img, clip_pred_labels_per_img, clip_score_per_img in zip(per_proposal_area_masks, clip_pred_labels, proposal_clip_score):
            per_cate_masks_per_img = self.generate_per_cate_mask(per_proposal_area_masks_per_img, clip_pred_labels_per_img, clip_score_per_img.shape[-1])
            per_cate_masks.append(per_cate_masks_per_img)
        #print('per_cate_masks', [ele.shape for ele in per_cate_masks])
        
        # forward the per categories mask to obtain the final masks(generate mask)
        per_cate_masks = self.forward_mask(per_cate_masks, img_metas)
        
        # obtain a subset
        # random_idx = torch.randperm(proposal_bboxes[0].shape[0])[:self.subset_num].cuda()
        # per_proposal_area_masks = [ele[random_idx] for ele in per_proposal_area_masks]
        # proposal_clip_score = [ele[random_idx] for ele in proposal_clip_score]
        # proposal_bboxes = [ele[random_idx] for ele in proposal_bboxes]
        
        # obtain the per proposal per category mask, obtain the prediction
        iter_num =  proposal_clip_score[0].shape[0] // self.infer_subset_num
        
        all_preds = []
        for per_proposal_area_masks_per_img, proposal_clip_score_per_img, per_cate_masks_per_img in zip(per_proposal_area_masks, proposal_clip_score, per_cate_masks):
            prediction_per_img = []
            for i in range(iter_num):
                start_from = int(i * self.infer_subset_num)
                end_at = int((i+1) * self.infer_subset_num)
                temp_res = self.cal_final_pred(per_proposal_area_masks_per_img[start_from: end_at], proposal_clip_score_per_img[start_from: end_at], per_cate_masks_per_img)
                prediction_per_img.append(temp_res)
            prediction_per_img = torch.cat(prediction_per_img, dim=0)
            all_preds.append(prediction_per_img)
        
        # find the target(for the random selected proposal), for each categories, calculate the target iou per categories
        pred_score_target = self.find_the_gt_for_proposal(proposal_bboxes, gt_bboxes, gt_labels)
        
        # concat the result 
        all_preds = torch.cat(all_preds, dim=0).unsqueeze(dim=0)
        pred_score_target = torch.cat(pred_score_target, dim=0).unsqueeze(dim=0)
        #pred_score_target = torch.zeros(all_preds.shape).cuda()
        result = torch.cat([all_preds, pred_score_target], dim=0)
        
        return  [result.cpu().numpy()]

    def aug_test(self, imgs, img_metas, rescale=False):
        pass
    
    def show_result(self, data, result, top_k=20, **kwargs):
        """Show RPN proposals on the image.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            top_k (int): Plot the first k bboxes only
               if set positive. Default: 20

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        mmcv.imshow_bboxes(data, result, top_k=top_k, **kwargs)
