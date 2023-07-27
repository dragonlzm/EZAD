# Copyright (c) OpenMMLab. All rights reserved.
from cgi import test
import warnings
import json
import os
import mmcv
import torch
from mmcv.image import tensor2imgs
from mmcv.ops import batched_nms
from mmcv.utils import Registry, build_from_cfg
from mmdet.core import (bbox_mapping, anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.core.bbox.iou_calculators import build_iou_calculator

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from PIL import Image
import numpy as np
import math
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn as nn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

@DETECTORS.register_module()
class ClsProposalGenerator(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 anchor_generator,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None):
        super(ClsProposalGenerator, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        #rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_train_cfg = train_cfg.get('rpn_head', None) if train_cfg is not None else None
        rpn_test_cfg = test_cfg.get('rpn_head', None) if test_cfg is not None else None
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=rpn_test_cfg)
        self.rpn_head = build_head(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.preprocess = _transform(self.backbone.input_resolution)

        # anchor generator
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.anchor_per_grid = len(anchor_generator['scales']) * len(anchor_generator['ratios'])
        self.down_sample_rate = anchor_generator['strides'][0]
        # deal with the crop size and location
        self.test_crop_size_modi_ratio = self.test_cfg.get('crop_size_modi', 1.0) if self.test_cfg is not None else 1.0
        self.test_crop_loca_modi_ratio = self.test_cfg.get('crop_loca_modi', 0) if self.test_cfg is not None else 0

        self.train_crop_size_modi_ratio = self.train_cfg.get('crop_size_modi', 1.0) if self.train_cfg is not None else 1.0
        self.train_crop_loca_modi_ratio = self.train_cfg.get('crop_loca_modi', 0) if self.train_cfg is not None else 0

        # anchor gt calculator
        self.calc_gt_anchor_iou = self.test_cfg.get('calc_gt_anchor_iou', False) if self.test_cfg is not None else False
        
        if self.calc_gt_anchor_iou:
            iou_calculator=dict(type='BboxOverlaps2D')
            self.iou_calculator = build_iou_calculator(iou_calculator)

        # pad the proposal result to a fixed length
        self.paded_proposal_num = self.test_cfg.get('paded_proposal_num', 1000) if self.test_cfg is not None else 1000
        # use min-entropy instead of max-confidence
        self.min_entropy = self.test_cfg.get('min_entropy', False) if self.test_cfg is not None else False
        # nms on all anchor
        self.nms_on_all_anchors = self.test_cfg.get('nms_on_all_anchors', False) if self.test_cfg is not None else False
        self.nms_threshold = self.test_cfg.get('nms_threshold', 0.5) if self.test_cfg is not None else 0.5
        self.nms_on_diff_scale = self.test_cfg.get('nms_on_diff_scale', False) if self.test_cfg is not None else False

        self.bbox_save_path_root = self.test_cfg.get('bbox_save_path_root', None) if self.test_cfg is not None else None
        self.save_pred_category = self.test_cfg.get('save_pred_category', None) if self.test_cfg is not None else None
        
        self.least_conf_bbox = self.test_cfg.get('least_conf_bbox', False) if self.test_cfg is not None else False
        self.use_sigmoid_for_cos = self.test_cfg.get('use_sigmoid_for_cos', False) if self.test_cfg is not None else False

        print('parameters:', 'anchor_generator["scales"]', anchor_generator['scales'], "anchor_generator['ratios']", anchor_generator['ratios'],
            "anchor_generator['strides']", anchor_generator['strides'], "self.paded_proposal_num", self.paded_proposal_num, "self.min_entropy", self.min_entropy,
            "self.nms_on_all_anchors", self.nms_on_all_anchors, "self.nms_threshold", self.nms_threshold,
            'self.bbox_save_path_root', self.bbox_save_path_root, 'self.least_conf_bbox', self.least_conf_bbox, 'self.use_sigmoid_for_cos',
            self.use_sigmoid_for_cos, 'self.save_pred_category', self.save_pred_category)

    def crop_img_to_patches(self, imgs, gt_bboxes, img_metas):
        # handle the test config
        if self.training: 
            crop_size_modi_ratio = self.train_crop_size_modi_ratio
            crop_loca_modi_ratio = self.train_crop_loca_modi_ratio
        else:
            crop_size_modi_ratio = self.test_crop_size_modi_ratio
            crop_loca_modi_ratio = self.test_crop_loca_modi_ratio            
        
        bs, c, _, _ = imgs.shape
        #'img_shape':  torch.Size([2, 3, 800, 1184])
        # what we need is [800, 1184, 3]
        imgs = imgs.permute(0, 2, 3, 1).numpy()

        all_results = []
        for img_idx in range(bs):
            H, W, channel = img_metas[img_idx]['img_shape']
            all_gt_bboxes = gt_bboxes[img_idx]
            if len(all_gt_bboxes) == 0:
                continue
            img = imgs[img_idx]
            result_per_img = []
            count_i = 0
            batch_new_patch = []
            for box_i, bbox in enumerate(all_gt_bboxes):
                # the original bbox location
                tl_x, tl_y, br_x, br_y = bbox[0], bbox[1], bbox[2], bbox[3]
                x = tl_x
                y = tl_y
                w = br_x - tl_x
                h = br_y - tl_y
                # change the bbox location by changing the top left position
                # bbox change direction
                x_direction_sign = random.randint(-1,1)
                y_direction_sign = random.randint(-1,1)
                # bbox direction change ratio(the ration should be 1/2, 1/3, 1/4, 1/5)
                # commonly we will mantain the size of the bbox unchange while changing
                # the localization of the bbox
                x_change_pixel = w * crop_loca_modi_ratio * x_direction_sign
                y_change_pixel = h * crop_loca_modi_ratio * y_direction_sign

                # change the bbox size ratio
                x_change_for_size = ((crop_size_modi_ratio - 1) / 2) * w
                y_change_for_size = ((crop_size_modi_ratio - 1) / 2) * h

                # the final format for the
                x_start_pos = math.floor(max(x-x_change_for_size+x_change_pixel , 0))
                y_start_pos = math.floor(max(y-y_change_for_size+y_change_pixel, 0))
                x_end_pos = math.ceil(min(x+x_change_for_size+w, W-1))
                y_end_pos = math.ceil(min(y+y_change_for_size+h, H-1))

                #x_start_pos = math.floor(max(x-0.1*w, 0))
                #y_start_pos = math.floor(max(y-0.1*h, 0))
                #x_end_pos = math.ceil(min(x+1.1*w, W-1))
                #y_end_pos = math.ceil(min(y+1.1*h, H-1))

                now_patch = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]           
                # crop the GT bbox and place it in the center of the zero square
                gt_h, gt_w, c = now_patch.shape
                if gt_h != gt_w:
                    long_edge = max((gt_h, gt_w))
                    empty_patch = np.zeros((long_edge, long_edge, 3))
                    if gt_h > gt_w:
                        x_start = (long_edge - gt_w) // 2
                        x_end = x_start + gt_w
                        empty_patch[:, x_start: x_end] = now_patch
                    else:
                        y_start = (long_edge - gt_h) // 2
                        y_end = y_start + gt_h
                        empty_patch[y_start: y_end] = now_patch
                    now_patch = empty_patch
                
                #data = Image.fromarray(np.uint8(now_patch))
                #data.save('/data2/lwll/zhuoming/detection/test/cls_finetuner_clip_base_100shots_train/patch_visualize/' + img_metas[img_idx]['ori_filename'] + '_' + str(box_i) + '.png')
                #new_patch, w_scale, h_scale = mmcv.imresize(now_patch, (224, 224), return_scale=True)
                # convert the numpy to PIL image
                PIL_image = Image.fromarray(np.uint8(now_patch))
                # do the preprocessing
                new_patch = self.preprocess(PIL_image)
                #image_result.append(np.expand_dims(new_patch, axis=0))
                new_patch = new_patch.unsqueeze(dim=0)
                batch_new_patch.append(new_patch)
                count_i += 1
                # extract feat for a patch
                if count_i % 100 == 0 and count_i != 0:
                    batch_new_patch = torch.cat(batch_new_patch, dim=0)
                    x = self.backbone(batch_new_patch.cuda())
                    result_per_img.append(x)
                    
                    count_i = 0
                    batch_new_patch = []
            if count_i != 0:
                batch_new_patch = torch.cat(batch_new_patch, dim=0)
                x = self.backbone(batch_new_patch.cuda())
                result_per_img.append(x)
                    
            result_per_img = torch.cat(result_per_img, dim=0)
            all_results.append(result_per_img)

        #cropped_patches = np.concatenate(result, axis=0)
        # the shape of the cropped_patches: torch.Size([gt_num_in_batch, 3, 224, 224])
        #cropped_patches = torch.cat(result, dim=0).cuda()
        return all_results

    def extract_feat(self, img, gt_bboxes, img_metas=None):
        """Extract features.

        Args:
            img (torch.Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            list[torch.Tensor]: Multi-level features that may have
                different resolutions.
        """
        # the image shape
        # it pad the images in the same batch into the same shape
        #torch.Size([2, 3, 800, 1088])
        #torch.Size([2, 3, 800, 1216])
        #torch.Size([2, 3, 800, 1216])
        #bs = img.shape[0]
        
        # crop the img into the patches with normalization and reshape
        # (a function to convert the img)
        # cropped_patches_list:len = batch_size, list[tensor] each tensor shape [gt_num_of_image, 3, 224, 224]
        result_list = self.crop_img_to_patches(img.cpu(), gt_bboxes, img_metas)

        # convert dimension from [bs, 64, 3, 224, 224] to [bs*64, 3, 224, 224]
        #converted_img_patches = converted_img_patches.view(bs, -1, self.backbone.input_resolution, self.backbone.input_resolution)

        # the input of the vision transformer should be torch.Size([64, 3, 224, 224])
        #result_list = []
        #for patches_per_img in cropped_patches_list:
        #    result_per_img = []
            # need to divide the anchor into different sections
            # reducing the overload of the gpu
        #    patches_per_img = patches_per_img.view(-1, self.anchor_per_grid, 3, 224, 224)
        #    for patches_per_grid_point in patches_per_img:
        #        x = self.backbone(patches_per_grid_point.cuda())
                #result_per_img.append(x.unsqueeze(dim=0))
        #        result_per_img.append(x)
        #    result_per_img = torch.cat(result_per_img, dim=0)
            # the shape of result_per_img should be [anchors_num_of_img, 512]
            #print(result_per_img.shape)
        #    result_list.append(result_per_img)
        # len(result_list) == batch_size, len(anchors_for_each_img) == batch_size
        # the shape of each tensor in result_list is [anchors_num_of_img, 512]
        return result_list

    def forward_dummy(self, img):
        """Dummy forward function."""
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        return rpn_outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_labels=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #if (isinstance(self.train_cfg.rpn, dict)
        #        and self.train_cfg.rpn.get('debug', False)):
        #    self.rpn_head.debug_imgs = tensor2imgs(img)
        bs = img.shape[0]
        # get the size of the paded image size in the batch (max_w, max_h)
        padded_img_size = (max([ele['pad_shape'][0] for ele in img_metas]), max([ele['pad_shape'][1] for ele in img_metas]))
        # in the original C4 setting the feature map is downsampled by 16
        # calculate the respective feat map size for the images in each batch
        # (the multiple images in one batch will share the same size)
        featmap_sizes = (padded_img_size[0] / self.down_sample_rate, padded_img_size[1] / self.down_sample_rate)
        # assuming that the len(multi_level_anchors) = 1
        anchors = self.anchor_generator.grid_anchors([featmap_sizes], device='cpu')
        anchors_for_each_img = anchors * bs

        x = self.extract_feat(img, anchors_for_each_img, img_metas)
        # x: list[tensor] each tensor shape [gt_num_of_image, 512]
        losses = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels,
                                             gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, gt_bboxes, gt_labels, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        img = img.unsqueeze(dim=0)
        img_metas = [img_metas]
        
        if self.bbox_save_path_root != None:
            if not os.path.exists(self.bbox_save_path_root):
                os.makedirs(self.bbox_save_path_root)
            clear_file_name = img_metas[0]['ori_filename']
            # handle Lvis situation
            if '/' in clear_file_name:
                clear_file_name = clear_file_name.split('/')[-1]
                
            file_name = os.path.join(self.bbox_save_path_root, '.'.join(clear_file_name.split('.')[:-1]) + '.json')
            if os.path.exists(file_name):
                return [np.zeros((1,5))]

        bs = img.shape[0]
        # get the size of the paded image size in the batch (max_w, max_h)
        padded_img_size = (max([ele['pad_shape'][0] for ele in img_metas]), max([ele['pad_shape'][1] for ele in img_metas]))
        # in the original C4 setting the feature map is downsampled by 16
        # calculate the respective feat map size for the images in each batch
        # (the multiple images in one batch will share the same size)
        featmap_sizes = (padded_img_size[0] / self.down_sample_rate, padded_img_size[1] / self.down_sample_rate)
        # assuming that the len(multi_level_anchors) = 1
        anchors = self.anchor_generator.grid_anchors([featmap_sizes], device='cpu')
        # len(anchors_for_each_img) == batch_size
        # the shape of each tensor in anchors_for_each_img [anchors_num_of_img, 4]
        anchors_for_each_img = anchors * bs

        # calcluate the iou between the gt and all anchors
        if self.calc_gt_anchor_iou:
            proposal_for_all_imgs = []
            for gt_bboxes_per_img, anchor_per_img in zip(gt_bboxes, anchors_for_each_img):
                iou_result = self.iou_calculator(gt_bboxes_per_img.cuda(), anchor_per_img.cuda())
                max_iou_per_gt = torch.max(iou_result, dim=1)[0]
                proposal_for_all_imgs.append(max_iou_per_gt)
        else:
            result_list = self.extract_feat(img, anchors_for_each_img, img_metas)
            # get origin input shape to onnx dynamic input shape
            if torch.onnx.is_in_onnx_export():
                img_shape = torch._shape_as_tensor(img)[2:]
                img_metas[0]['img_shape_for_onnx'] = img_shape
            # the len(pred_logits) == batch_size, with each ele in [num_anchors, num_cates]
            pred_logits = self.rpn_head.simple_test_bboxes(result_list, gt_labels, img_metas, gt_bboxes)
            softmax = nn.Softmax(dim=1)
            sigmoid = nn.Sigmoid()
            # prepare for nms
            nms=dict(type='nms', iou_threshold=self.nms_threshold)

            if self.nms_on_all_anchors:
                proposal_for_all_imgs = []
                predicted_cates_for_all_imgs = []
                predicted_confs_for_all_imgs = []
                for logits_per_img, anchor_per_img, img_info in zip(pred_logits, anchors_for_each_img, img_metas):
                    h, w, _ = img_info['img_shape']
                    if self.use_sigmoid_for_cos:
                        #print('before sigmoid:', logits_per_img)
                        pred_prob = sigmoid(logits_per_img)
                        #print('after sigmoid:', pred_prob)
                    else:
                        pred_prob = softmax(logits_per_img)
                    if not self.min_entropy:
                        max_pred_prob = torch.max(pred_prob, dim=1)
                        max_score_per_anchor = max_pred_prob[0]
                        max_idx_per_anchor = max_pred_prob[1]
                    else:
                        cate_num = pred_prob.shape[-1]
                        #print(cate_num)
                        factor = torch.log(torch.tensor(cate_num))
                        # calculate the entropy for each anchor
                        prepared_gt_pred = pred_prob
                        prepared_gt_pred[prepared_gt_pred == 0] = 1e-5
                        log_result = - torch.log(prepared_gt_pred)
                        entro = (log_result * pred_prob).sum(dim=-1)
                        max_score_per_anchor = - entro + factor.item()
                        #entro = entro.view(-1, self.anchor_per_grid)
                        # select the anchor with the max negative entropy in each grid
                        #max_score_per_grid = torch.max(entro, dim=1)
                        # the shape of the anchors_for_imgs (num_of_grid, )
                        #max_score_per_grid_val = max_score_per_grid[0]
                        #max_score_per_grid_idx = max_score_per_grid[1] 

                    # find the min confidence proposal
                    if self.least_conf_bbox:
                        max_score_per_anchor = -max_score_per_anchor
                    
                    result_proposal_per_img = []
                    for anchor in anchor_per_img:
                        anchor[0] = 0 if anchor[0] < 0 else anchor[0]
                        anchor[1] = 0 if anchor[1] < 0 else anchor[1]
                        anchor[2] = w if anchor[2] > w else anchor[2]
                        anchor[3] = h if anchor[3] > h else anchor[3]
                        # rescale the proposal
                        result_proposal_per_img.append(anchor.unsqueeze(dim=0))
                    result_proposal_per_img = torch.cat(result_proposal_per_img, dim=0)

                    # regard all anchor as one category
                    ids = torch.zeros(max_score_per_anchor.shape)
                    
                    if self.nms_on_diff_scale:
                        size_of_proposal = (result_proposal_per_img[:, 2] - result_proposal_per_img[:, 0]) * (result_proposal_per_img[:, 3] - result_proposal_per_img[:, 1])
                        small_size_proposal = result_proposal_per_img[size_of_proposal < 36*36]
                        small_size_proposal_score = max_score_per_anchor[size_of_proposal < 36*36]
                        small_size_ids = ids[size_of_proposal < 36*36]
                        samll_dets, _ = batched_nms(small_size_proposal.cuda(), small_size_proposal_score.cuda(), small_size_ids.cuda(), nms)
                        
                        median_size_proposal = result_proposal_per_img[(size_of_proposal > 36*36) & (size_of_proposal < 96*96)]
                        median_size_proposal_score = max_score_per_anchor[(size_of_proposal > 36*36) & (size_of_proposal < 96*96)]
                        median_size_ids = ids[(size_of_proposal > 36*36) & (size_of_proposal < 96*96)]
                        median_dets, _ = batched_nms(median_size_proposal.cuda(), median_size_proposal_score.cuda(), median_size_ids.cuda(), nms)
                        
                        large_size_proposal = result_proposal_per_img[size_of_proposal > 96*96]
                        large_size_proposal_score = max_score_per_anchor[size_of_proposal > 96*96]
                        large_size_ids = ids[size_of_proposal > 96*96]
                        large_dets, _ = batched_nms(large_size_proposal.cuda(), large_size_proposal_score.cuda(), large_size_ids.cuda(), nms)
                     
                        dets = torch.cat([samll_dets, median_dets, large_dets], dim=0)
                        dets_score = dets[:, -1]
                        need_idx = torch.sort(dets_score, descending=True)[1]
                        dets = dets[need_idx]
                        
                    else:
                        dets, keep = batched_nms(result_proposal_per_img.cuda(), max_score_per_anchor.cuda(), ids.cuda(), nms)
                    
                    # resize the proposal
                    dets[:, :4] /= dets.new_tensor(img_info['scale_factor'])
                    # pad the proposal result to the fixed length
                    dets = dets[:self.paded_proposal_num]
                    if len(dets) < self.paded_proposal_num:
                        gap = self.paded_proposal_num - len(dets)
                        padded_empty_proposals = torch.zeros(gap, 5).cuda()
                        dets = torch.cat([dets, padded_empty_proposals], dim=0)
                    
                    #if img_info['ori_filename'] == '000000002985.jpg':
                    #    print(dets[:100], max_idx_per_anchor[keep][:100])
                    
                    #print('dets', dets.shape)
                    proposal_for_all_imgs.append(dets)
                    #print('max_idx_per_anchor', max_idx_per_anchor.shape)
                    #print('max_idx_per_anchor[keep][:self.paded_proposal_num]', max_idx_per_anchor[keep][:self.paded_proposal_num].shape, max_idx_per_anchor[keep][:self.paded_proposal_num])
                    if self.save_pred_category:
                        predicted_cates_for_all_imgs.append(max_idx_per_anchor[keep][:self.paded_proposal_num])
                        predicted_confs_for_all_imgs.append(max_score_per_anchor[keep][:self.paded_proposal_num])
        
            else:
                proposal_for_all_imgs = []
                for logits_per_img, anchor_per_img, img_info in zip(pred_logits, anchors_for_each_img, img_metas):
                    h, w, _ = img_info['img_shape']
                    pred_prob = softmax(logits_per_img)

                    # select the max pred score in the prediction distribution
                    max_pred_prob = torch.max(pred_prob, dim=1)
                    max_score_per_anchor = max_pred_prob[0]
                    max_score_per_anchor = max_score_per_anchor.view(-1, self.anchor_per_grid)
                    # select the anchor with highest max confidence score in each grid
                    max_score_per_grid = torch.max(max_score_per_anchor, dim=1)
                    # the shape of the anchors_for_imgs (num_of_grid, )
                    max_score_per_grid_val = max_score_per_grid[0]
                    max_score_per_grid_idx = max_score_per_grid[1]
                    
                    result_proposal_per_img = []
                    result_score_per_img = []
                    anchor_per_img = anchor_per_img.view(-1, self.anchor_per_grid, 4)
                    for i, (max_grid_val, max_grid_idx) in enumerate(zip(max_score_per_grid_val, max_score_per_grid_idx)):
                        if not self.min_entropy and max_grid_val < 0.9:
                            continue
                        selected_anchor = anchor_per_img[i, max_grid_idx]
                        # adjust the cordinate to make the bbox valid
                        selected_anchor[0] = 0 if selected_anchor[0] < 0 else selected_anchor[0]
                        selected_anchor[1] = 0 if selected_anchor[1] < 0 else selected_anchor[1]
                        selected_anchor[2] = w if selected_anchor[2] > w else selected_anchor[2]
                        selected_anchor[3] = h if selected_anchor[3] > h else selected_anchor[3]
                        # rescale the proposal
                        result_proposal_per_img.append(selected_anchor.unsqueeze(dim=0))
                        result_score_per_img.append(max_grid_val.unsqueeze(dim=0))
                    
                    result_proposal_per_img = torch.cat(result_proposal_per_img, dim=0)
                    result_score_per_img = torch.cat(result_score_per_img, dim=0)

                    # regard all anchor as one category
                    ids = torch.zeros(result_score_per_img.shape)
                    dets, keep = batched_nms(result_proposal_per_img.cuda(), result_score_per_img.cuda(), ids.cuda(), nms)
                    # resize the proposal
                    dets[:, :4] /= dets.new_tensor(img_info['scale_factor'])
                    # pad the proposal result to the fixed length
                    dets = dets[:self.paded_proposal_num]
                    if len(dets) < self.paded_proposal_num:
                        gap = self.paded_proposal_num - len(dets)
                        padded_empty_proposals = torch.zeros(gap, 5).cuda()
                        dets = torch.cat([dets, padded_empty_proposals], dim=0)

                    proposal_for_all_imgs.append(dets)
                    
        result = [proposal.cpu().numpy() for proposal in proposal_for_all_imgs]
        if self.save_pred_category:
            predicted_cates_for_all_imgs = [ele.cpu().numpy() for ele in predicted_cates_for_all_imgs]
            predicted_confs_for_all_imgs = [ele.cpu().numpy() for ele in predicted_confs_for_all_imgs]
        #print('result[0].shape', result[0].shape)
        
        if self.bbox_save_path_root != None:    
            file = open(file_name, 'w')
            if self.save_pred_category:
                result_json = {'image_id':int(clear_file_name.split('.')[0].strip('0')), 'score':result[0].tolist(), 
                               'predicted_cates_for_all_imgs': predicted_cates_for_all_imgs[0].tolist(),
                               'predicted_confs_for_all_imgs': predicted_confs_for_all_imgs[0].tolist()}
            else:
                result_json = {'image_id':int(clear_file_name.split('.')[0].strip('0')), 'score':result[0].tolist()}
            file.write(json.dumps(result_json))
            file.close()
            
        return result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        proposal_list = self.rpn_head.aug_test_rpn(
            self.extract_feats(imgs), img_metas)
        if not rescale:
            for proposals, img_meta in zip(proposal_list, img_metas[0]):
                img_shape = img_meta['img_shape']
                scale_factor = img_meta['scale_factor']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                proposals[:, :4] = bbox_mapping(proposals[:, :4], img_shape,
                                                scale_factor, flip,
                                                flip_direction)
        return [proposal.cpu().numpy() for proposal in proposal_list]

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
