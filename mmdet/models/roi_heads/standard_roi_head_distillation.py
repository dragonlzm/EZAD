# Copyright (c) OpenMMLab. All rights reserved.
from dis import dis
from hashlib import new
import torch
import torch.nn as nn
import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import os
import json


@HEADS.register_module()
class StandardRoIHeadDistill(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """This is the ROIhead with distillation, designed for Zero-shot detection"""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        self.avg_pool = nn.AvgPool2d(self.bbox_head.roi_feat_size)
        
        self.distillation_loss_config = dict(type='L1Loss', loss_weight=1.0)
        self.distillation_loss = build_loss(self.distillation_loss_config)
        self.distill_loss_factor = self.train_cfg.get('distill_loss_factor', 1) if self.train_cfg is not None else 1
        self.use_contrast_distill = self.train_cfg.get('use_contrast_distill', False) if self.train_cfg is not None else False
        self.rand_distill_bbox_factor = self.train_cfg.get('rand_distill_bbox_factor', 1.0) if self.train_cfg is not None else 1.0
        
        self.contrastive_weight = self.train_cfg.get('contrastive_weight', 0.5) if self.train_cfg is not None else 0.5
        self.gt_bboxes_distill_weight = self.train_cfg.get('gt_bboxes_distill_weight', None) if self.train_cfg is not None else None
        # config for transformer head
        self.use_proposal_for_distill = self.train_cfg.get('use_proposal_for_distill', False) if self.train_cfg is not None else False
        # if using double branch, generate another bbox head which do not have regression branch
        if self.use_double_bbox_head:
            dist_bbox_head_config = bbox_head
            dist_bbox_head_config['with_reg'] = False
            dist_bbox_head_config['reg_with_cls_embedding'] = False
            dist_bbox_head_config['use_bg_vector'] = False
            print('testing dist_bbox_head_config:', dist_bbox_head_config)
            self.dist_bbox_head = build_head(dist_bbox_head_config)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      distilled_feat=None, 
                      rand_bboxes=None,
                      rand_bbox_weights=None,
                      bg_bboxes=None,
                      bg_feats=None,
                      cp_mark=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            distilled_feat (list[Tensor]): only contain the feat for the gt bboxes
                and the random bboxes

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, distilled_feat,
                                                    rand_bboxes,
                                                    rand_bbox_weights=rand_bbox_weights,
                                                    bg_bboxes=bg_bboxes,
                                                    bg_feats=bg_feats,
                                                    cp_mark=cp_mark)
            losses.update(bbox_results['loss_bbox'])
            losses.update(bbox_results['distill_loss_value'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, distilled_feat=None, gt_rand_rois=None, proposal_assigned_gt_labels=None, img_metas=None, distill_ele_weight=None, bboxes_num=None):
        """Box head forward function used in both training and testing.
        bboxes_num: list[tuple(gt_bbox_num, rand_bbox_num, proposal_number)]
        """  
        # is the number of feat map layer
        if distilled_feat != None and gt_rand_rois != None:
            # gt and random bbox feat from backbone_to
            # gt_and_rand_bbox_feat: torch.Size([1024, 256, 7, 7])
            gt_and_rand_bbox_feat = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], gt_rand_rois)
            # conduct the global averger pooling on the gt_and_rand_bbox_feat
            #gt_and_rand_bbox_feat = self.avg_pool(gt_and_rand_bbox_feat)
            # convert to shape from [221, 512, 1, 1] to [221, 512]
            #gt_and_rand_bbox_feat = gt_and_rand_bbox_feat.view(-1, self.bbox_roi_extractor.out_channels)
            # concatenate the distilled_feat
            
            # calculate the distill loss
            #distill_loss_value = self.distillation_loss(gt_and_rand_bbox_feat, distilled_feat)        
        
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        # we use the fpn do not need to consider the share head
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            if distilled_feat != None and gt_rand_rois != None:
                gt_and_rand_bbox_feat = self.shared_head(gt_and_rand_bbox_feat)
        
        # if we use bg proposal, the cls_score will has the the length of samples + bg number
        cls_score, bbox_pred, gt_and_bg_feats = self.bbox_head(bbox_feats, proposal_assigned_gt_labels)
        #print('cls_score', cls_score.shape, 'bbox_pred', bbox_pred.shape, 'gt_and_bg_feats', gt_and_bg_feats.shape, 'bboxes_num', [ele for ele in bboxes_num])
        
        # for save the classification feat
        if self.save_the_feat is not None:
            if not os.path.exists(self.save_the_feat):
                os.makedirs(self.save_the_feat)
            file_name = img_metas[0]['ori_filename'].split('.')[0] + '.json'
            random_file_path = os.path.join(self.save_the_feat, file_name)  
            file = open(random_file_path, 'w')
            result_json = {'feat':gt_and_bg_feats.cpu().tolist()}
            file.write(json.dumps(result_json))
            file.close()
        
        # obtain the feat for the distillation
        if self.training and distilled_feat != None and gt_rand_rois != None:
            ### just for testing
            # before combine
            # print('before the combine:', gt_and_rand_bbox_feat.shape, len(distilled_feat), distilled_feat[0].shape, len(distill_ele_weight), distill_ele_weight[0].shape,
            #       torch.cat(distilled_feat, dim=0).shape, torch.cat(distill_ele_weight, dim=0).shape)
            # test_gt_and_rand_bbox_feat = torch.cat([torch.zeros([1] + list(gt_and_rand_bbox_feat.shape[1:])).cuda(), gt_and_rand_bbox_feat], dim=0)
            # test_distilled_feat = [torch.zeros([1] + list(distilled_feat[0].shape[1:])).cuda()] + distilled_feat
            # test_distill_ele_weight = [torch.zeros([1] + list(distill_ele_weight[0].shape[1:])).cuda()] + distill_ele_weight
            # test_distilled_feat = torch.cat(test_distilled_feat, dim=0)
            # test_distill_ele_weight = torch.cat(test_distill_ele_weight, dim=0)
            # print('after the combine:', test_gt_and_rand_bbox_feat.shape, test_distilled_feat.shape, test_distill_ele_weight.shape)
            
            if gt_and_rand_bbox_feat.shape[0] == 0:
                print('before the combine:', gt_and_rand_bbox_feat.shape, len(distilled_feat), distilled_feat[0].shape, len(distill_ele_weight), distill_ele_weight[0].shape,
                  torch.cat(distilled_feat, dim=0).shape, torch.cat(distill_ele_weight, dim=0).shape)
                gt_and_rand_bbox_feat = torch.cat([torch.zeros([1] + list(gt_and_rand_bbox_feat.shape[1:])).cuda(), gt_and_rand_bbox_feat], dim=0)
                distilled_feat =  [torch.zeros([1] + list(distilled_feat[0].shape[1:])).cuda()] + distilled_feat
                distill_ele_weight =  [torch.zeros([1] + list(distill_ele_weight[0].shape[1:])).cuda()] + distill_ele_weight
            
            if self.use_double_bbox_head:
                dist_cls_score, _, pred_feats = self.dist_bbox_head(gt_and_rand_bbox_feat)
            else:
                _, _, pred_feats = self.bbox_head(gt_and_rand_bbox_feat)
            # normalize the distilled feat
            cat_distilled_feat = torch.cat(distilled_feat, dim=0)
            if distill_ele_weight:
                distill_ele_weight = torch.cat(distill_ele_weight, dim=0)
                
            if hasattr(self.bbox_head, 'use_svd_conversion') and self.bbox_head.use_svd_conversion != None:
                cat_distilled_feat *= self.bbox_head.svd_conversion_mat(cat_distilled_feat)
            cat_distilled_feat = cat_distilled_feat / cat_distilled_feat.norm(dim=-1, keepdim=True)
            # cos_value = torch.sum(pred_feats * cat_distilled_feat)
            # self.all_cosine_value += cos_value.item()
            # self.count += pred_feats.shape[0]
            # self.iter += 1
            # if self.iter % 100 == 0:
            #     print('average cosine:', self.all_cosine_value / self.count)
            
            distill_loss_value = self.distillation_loss(pred_feats, cat_distilled_feat, distill_ele_weight)
            if self.use_contrast_distill:
                # aggregate the gt bboxes rn feature
                # aggregate the clip proposal rn feature                
                start_at = 0
                all_gt_bboxes_rn_feat = []
                all_clip_proposal_rn_feat = []
                for gt_bbox_num, clip_proposal_num, _ in bboxes_num:
                    gt_bbox_rn_feat = pred_feats[start_at:start_at+gt_bbox_num, :]
                    all_gt_bboxes_rn_feat.append(gt_bbox_rn_feat)
                    start_at += gt_bbox_num
                    clip_proposal_rn_feat = pred_feats[start_at:start_at+clip_proposal_num, :]
                    all_clip_proposal_rn_feat.append(clip_proposal_rn_feat)
                    start_at += clip_proposal_num

                # aggregate the gt bboxes clip feature                
                # aggregate the clip proposal clip feature
                start_at = 0
                all_gt_bboxes_clip_feat = []
                all_clip_proposal_clip_feat = []
                for gt_bbox_num, clip_proposal_num, _ in bboxes_num:
                    gt_bbox_rn_feat = cat_distilled_feat[start_at:start_at+gt_bbox_num, :]
                    all_gt_bboxes_clip_feat.append(gt_bbox_rn_feat)
                    start_at += gt_bbox_num
                    clip_proposal_rn_feat = cat_distilled_feat[start_at:start_at+clip_proposal_num, :]
                    all_clip_proposal_clip_feat.append(clip_proposal_rn_feat)
                    start_at += clip_proposal_num                
                
                # concat all the feature
                all_gt_bboxes_rn_feat = torch.cat(all_gt_bboxes_rn_feat, dim=0)
                all_clip_proposal_rn_feat = torch.cat(all_clip_proposal_rn_feat, dim=0)
                all_gt_bboxes_clip_feat = torch.cat(all_gt_bboxes_clip_feat, dim=0)
                all_clip_proposal_clip_feat = torch.cat(all_clip_proposal_clip_feat, dim=0)
                
                num_gt_bbox = all_gt_bboxes_rn_feat.shape[0]
                num_clip_proposal = all_clip_proposal_rn_feat.shape[0]
                # sample negative target for all_gt_bboxes_rn_feat
                random_idx = torch.randint(low=0, high=num_clip_proposal, size=(num_gt_bbox, ))
                negative_target_for_gt_bbox = all_clip_proposal_clip_feat[random_idx]
                contrastive_loss_for_gt_bbox = -self.distillation_loss(all_gt_bboxes_rn_feat, negative_target_for_gt_bbox)
                
                # sample negative target for all_clip_proposal_rn_feat
                random_idx = torch.randint(low=0, high=num_gt_bbox, size=(num_clip_proposal, ))
                negative_target_for_clip_proposal = all_gt_bboxes_clip_feat[random_idx]
                contrastive_loss_for_clip_proposal = -self.distillation_loss(all_clip_proposal_rn_feat, negative_target_for_clip_proposal)
                
                distill_loss_value = distill_loss_value + (contrastive_loss_for_gt_bbox + contrastive_loss_for_clip_proposal) * self.contrastive_weight
                
            #distill_loss_value *= (self.bbox_head.clip_dim * 0.5)
            if self.gt_only_damp_factor:
                temp_distill_loss_factor = (self.distill_loss_factor * cat_distilled_feat.shape[0]) / (400 + cat_distilled_feat.shape[0])
                #print('cat_distilled_feat.shape', cat_distilled_feat.shape, 'temp_distill_loss_factor', temp_distill_loss_factor)
                distill_loss_value *= (self.bbox_head.clip_dim * temp_distill_loss_factor)
            else:
                distill_loss_value *= (self.bbox_head.clip_dim * self.distill_loss_factor)
            
            '''
            # test the feat is matched or not
            gt_feat = [all_feats[:len(gt_lab)] for all_feats, gt_lab in zip(distilled_feat, gt_labels)]
            #print([ele.shape for ele in gt_feat])
            gt_feat = torch.cat(gt_feat, dim=0)
            gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
            # calculate the cos simiarity
            fg_score = self.bbox_head.fc_cls_fg(gt_feat)
            print('self.bbox_head.load_value.t()', self.bbox_head.load_value.t(), 'self.bbox_head.fc_cls_fg', self.bbox_head.fc_cls_fg)
            #fg_score = gt_feat @ self.bbox_head.load_value.t()
            #bg_score = self.fc_cls_bg(x_cls)
            # find the max cos value class
            max_id = torch.max(fg_score, dim=-1)[1]
            
            # calculate the acc
            cat_gt_label = torch.cat(gt_labels, dim=0)
            
            print('max_id', max_id, 'cat_gt_label', cat_gt_label)
            self.match_count += torch.sum((max_id == cat_gt_label)).item()
            self.total += max_id.shape[0]
            print('accumulated acc:', self.match_count / self.total)'''

        if self.use_double_bbox_head and not self.training:
            dist_cls_score, _, _ = self.dist_bbox_head(bbox_feats)
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, dist_cls_score=dist_cls_score)
        elif distilled_feat != None and gt_rand_rois != None:
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, distill_loss_value=dict(distill_loss_value=distill_loss_value))
        else:
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, 
                            gt_bboxes, gt_labels,
                            img_metas, distilled_feat, 
                            rand_bboxes,
                            rand_bbox_weights=None,
                            bg_bboxes=None,
                            bg_feats=None,
                            cp_mark=None):
        """Run forward function and calculate loss for box head in training."""
        """the gt_bboxes and rand_bboxes are in the format of xyxy """
        
        # prepare the roi for the proposal
        if self.use_bg_pro_as_ns:
            rois = bbox2roi([torch.cat([res.bboxes, bg_bbox]).cuda() for res, bg_bbox in zip(sampling_results, bg_bboxes)])
        else:     
            rois = bbox2roi([res.bboxes for res in sampling_results])

        if self.use_bg_pro_as_ns:
            bbox_targets_ori = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, concat=False)
            
            labels, label_weights, bbox_targets, bbox_weights = bbox_targets_ori
            # concat the labels, label_weights, bbox_targets, bbox_weights
            # the labels should be bg label, label_weights should be the same as
            # other label. bbox_weights should be zero
            bg_labels = [torch.full((bg_bboxes[i].shape[0], ),
                                     self.bbox_head.num_classes,
                                     dtype=torch.long).cuda() for i in range(len(bg_bboxes))]
            bg_label_weights = [torch.full((bg_bboxes[i].shape[0], ),
                                     self.bg_pro_as_ns_weight,
                                     dtype=torch.long).cuda() for i in range(len(bg_bboxes))]
            bg_bbox_targets = [torch.zeros(bg_bboxes[i].shape[0], 4).cuda() for i in range(len(bg_bboxes))]
            bg_bbox_weights = [torch.zeros(bg_bboxes[i].shape[0], 4).cuda() for i in range(len(bg_bboxes))]
            # concat inside first
            labels = [torch.cat([label, bg_label], dim=0).cuda() for label, bg_label in zip(labels, bg_labels)]
            label_weights = [torch.cat([label_weight, bg_label_weight], dim=0).cuda() for label_weight, bg_label_weight in zip(label_weights, bg_label_weights)]
            bbox_targets = [torch.cat([bbox_target, bg_bbox_target], dim=0).cuda() for bbox_target, bg_bbox_target in zip(bbox_targets, bg_bbox_targets)]
            bbox_weights = [torch.cat([bbox_weight, bg_bbox_weight], dim=0).cuda() for bbox_weight, bg_bbox_weight in zip(bbox_weights, bg_bbox_weights)]
            # concat outside
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_targets = (labels, label_weights, bbox_targets, bbox_weights)
        else:
            #print('in roi head:', img_metas)
            bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, img_metas=img_metas)

        # purturb the gt bbox only
        if self.add_distill_pertrub:
            prepared_gt_bboxes = []
            for gt_bbox, img_meta in zip(gt_bboxes, img_metas):
                #print('before perturbation:', gt_rand_roi[:10])
                H, W, channel = img_meta['img_shape']
                tl_x, tl_y, br_x, br_y = gt_bbox[:, 0], gt_bbox[:, 1], gt_bbox[:, 2], gt_bbox[:, 3]
                w = br_x - tl_x
                h = br_y - tl_y
                # change the bbox location by changing the top left position
                # bbox change direction
                x_direction_sign = torch.randint(low=-1,high=1,size=w.shape).cuda()
                y_direction_sign = torch.randint(low=-1,high=1,size=w.shape).cuda()
                # bbox direction change ratio(the ration should be 1/2, 1/3, 1/4, 1/5)
                # commonly we will mantain the size of the bbox unchange while changing
                # the localization of the bbox, the change ratio would be even distribution [0, self.crop_loca_modi_ratio]
                x_change_pixel = w * x_direction_sign * torch.rand(w.shape).cuda() * self.crop_loca_modi_ratio
                y_change_pixel = h * y_direction_sign * torch.rand(w.shape).cuda() * self.crop_loca_modi_ratio
                # change the bbox size ratio, would be the even distribution [1, self.crop_size_modi_ratio]
                x_change_for_size = ((self.crop_size_modi_ratio - 1) * torch.rand(w.shape).cuda() / 2) * w
                y_change_for_size = ((self.crop_size_modi_ratio - 1) * torch.rand(w.shape).cuda() / 2) * h
                # the final format for the
                x_start_pos = torch.clamp(tl_x-x_change_for_size+x_change_pixel , min=0.1).unsqueeze(dim=-1)
                y_start_pos = torch.clamp(tl_y-y_change_for_size+y_change_pixel, min=0.1).unsqueeze(dim=-1)
                x_end_pos = torch.clamp(tl_x+x_change_for_size+w, max=W-1).unsqueeze(dim=-1)
                y_end_pos = torch.clamp(tl_y+y_change_for_size+h, max=H-1).unsqueeze(dim=-1)
                # concat the result
                perturbed_bbox_per_img = torch.cat([x_start_pos, y_start_pos, x_end_pos, y_end_pos], dim=-1)
                
                # pad the purturb result with the gt result, remain part of the gt bbox unchange
                remain_ratio = 1 - self.pertrub_ratio 
                random_choice = np.random.choice(gt_bbox.shape[0], int(gt_bbox.shape[0] * remain_ratio), replace=False)
                random_choice = torch.from_numpy(random_choice).cuda()
                perturbed_bbox_per_img[random_choice] = gt_bbox[random_choice]
                #print('perturbed_bbox_per_img', perturbed_bbox_per_img, 'gt_bbox', gt_bbox)
                                
                prepared_gt_bboxes.append(perturbed_bbox_per_img)
        else:
            prepared_gt_bboxes = gt_bboxes   

        # prepare the roi for the gt and the random bboxes
        if self.use_bg_pro_for_distill:
            gt_rand_rois = [torch.cat([gt_bbox, random_bbox, bg_bbox], dim=0) for gt_bbox, random_bbox, bg_bbox in zip(prepared_gt_bboxes, rand_bboxes, bg_bboxes)]
            distill_ele_weight = None
        elif self.use_only_gt_pro_for_distill:
            gt_rand_rois = prepared_gt_bboxes
            distill_ele_weight = None
        elif self.use_only_clip_prop_for_distill:
            rand_bboxes = [rand_bbox[torch.abs(rand_bbox).sum(dim=1) > 0].float() 
                           for rand_bbox in rand_bboxes]
            gt_rand_rois = rand_bboxes
            if rand_bbox_weights is not None:
                distill_ele_weight = []
                feat_dim = distilled_feat[0].shape[-1]
                for rand_bbox_weight in rand_bbox_weights:
                    # whether we have the per clip proposal bbox distillation weigth
                    weight_per_img = rand_bbox_weight.unsqueeze(dim=-1).repeat([1,feat_dim])
                    # normalize the weight
                    # the factor should be: (num of gt bbox + num of random bbox) / (weight of all gt bbox + weight of the all random bboxes)
                    # p.s. here the normalization is for the weight, since the l1loss will always reduce the loss to average value
                    # therefore the size of the loss is irrelavant with the number of bbox
                    # but the weight we add here will affect the loss size, l1 loss is a: weighted sum / number of ele
                    total_weight = torch.sum(weight_per_img[:, 0]).item()
                    if total_weight == 0:
                        #print('rand_bboxes', [ele.shape for ele in rand_bboxes], 'weight_per_img', weight_per_img)
                        total_weight = 1
                    
                    normalize_factor = weight_per_img.shape[0] / total_weight
                    weight_per_img *= normalize_factor
                    distill_ele_weight.append(weight_per_img)   
            else:
                distill_ele_weight = None
        else:
            # filter the padded random bboxes
            rand_bboxes = [rand_bbox[torch.abs(rand_bbox).sum(dim=1) > 0]
                for rand_bbox in rand_bboxes]
            original_gt_nums = [dist_feat.shape[0] - rand_bbox.shape[0] for dist_feat, rand_bbox in zip(distilled_feat, rand_bboxes)]
            gt_rand_rois = [torch.cat([gt_bbox[:original_gt_num, :], random_bbox], dim=0).float() for gt_bbox, random_bbox, original_gt_num in zip(prepared_gt_bboxes, rand_bboxes, original_gt_nums)]
            
            # prepare the distillation weight
            ### there is three situations which need to specify the per clip proposal distillation weight
            ### 1. when using the copy and paste augmentation: at this situation, the number of gt bbox is large than the gt_feat, 
            ###    therefore we use original_gt_nums to get the real gt bboxes number
            ###    at the same time, we also need to set the weight for all the clip proposal to 0, 
            ###    if the image use the copy and paste augmentation, in which the mark of cp_mark = True
            ### 2. when given the gt_bboxes_distill_weight: usually, when we filter the base categories clip porposal, we will give a
            ###    larger weight to the gt bbox when it participate in distillation
            ### 3. when using the per clip proposal distillation weight, at this time we usually give a per bbox weight for clip proposal
            
            if cp_mark is not None or self.gt_bboxes_distill_weight is not None or rand_bbox_weights is not None:
                # prepare for not in the first situtation
                if cp_mark == None:
                    cp_mark = [False for ele in rand_bboxes]
                # prepare for not in the second situation
                gt_bbox_distill_weight = self.gt_bboxes_distill_weight if self.gt_bboxes_distill_weight is not None else 1.0
                # prepare for not in the third situation 
                if rand_bbox_weights == None:
                    rand_bbox_weights = [None for ele in rand_bboxes]

                feat_dim = distilled_feat[0].shape[-1]
                distill_ele_weight = []
                for original_gt_num, random_bbox, mark, rand_bbox_weight in zip(original_gt_nums, rand_bboxes, cp_mark, rand_bbox_weights):
                    if mark == True:
                        # if we using the copy and paste we just simply ignore all the clip proposal bboxes(random_bbox), so the weight of random_bbox is 0
                        weight_per_img = torch.cat([torch.full((original_gt_num, feat_dim), gt_bbox_distill_weight), torch.zeros(random_bbox.shape[0], feat_dim)], dim=0).cuda()
                        # if we using the copy and paste we do not normalize the weight, otherwise the weight for only gt bboxes will be too large
                    else:
                        # whether we have the per clip proposal bbox distillation weigth
                        if rand_bbox_weight is not None:
                            rand_bbox_weight = rand_bbox_weight.unsqueeze(dim=-1).repeat([1,feat_dim])
                        else:
                            rand_bbox_weight = torch.ones(random_bbox.shape[0], feat_dim).cuda()
                        
                        rand_bbox_weight *= self.rand_distill_bbox_factor
                        
                        # otherwise the weight of the random_bbox will be 1
                        weight_per_img = torch.cat([torch.full((original_gt_num, feat_dim), gt_bbox_distill_weight).cuda(), rand_bbox_weight], dim=0)
                        # normalize the weight
                        # the factor should be: (num of gt bbox + num of random bbox) / (weight of all gt bbox + weight of the all random bboxes)
                        # p.s. here the normalization is for the weight, since the l1loss will always reduce the loss to average value
                        # therefore the size of the loss is irrelavant with the number of bbox
                        # but the weight we add here will affect the loss size, l1 loss is a: weighted sum / number of ele
                        normalize_factor = weight_per_img.shape[0] / torch.sum(weight_per_img[:, 0]).item()
                        weight_per_img *= normalize_factor
                    distill_ele_weight.append(weight_per_img)   
            else:
                distill_ele_weight = None
        
        gt_rand_rois = bbox2roi(gt_rand_rois)
        # save the bboxes number for each image:
        # list[tuple(gt_bbox_num, rand_bbox_num, proposal_number)]
        bboxes_num = [(gt_bbox.shape[0], random_bbox.shape[0], res.bboxes.shape[0]) for gt_bbox, random_bbox, res in zip(prepared_gt_bboxes, rand_bboxes, sampling_results)]
        bbox_results = self._bbox_forward(x, rois, distilled_feat, gt_rand_rois, bbox_targets[0], distill_ele_weight=distill_ele_weight, bboxes_num=bboxes_num, img_metas=img_metas)
            
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
