# Copyright (c) OpenMMLab. All rights reserved.
from cgi import test
from email.mime import image
import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import bbox_mapping, bbox2roi
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor
from .base import BaseDetector
from PIL import Image
import numpy as np
import torch.nn as nn
import math
import random
import os
import json
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import functional as F
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

MEANS = (0.48145466, 0.4578275, 0.40821073)
STDS = (0.26862954, 0.26130258, 0.27577711)

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(MEANS, STDS),
    ])


@DETECTORS.register_module()
class ClsFinetuner(BaseDetector):
    """Implementation of the CLIP, we use this class to load the CLIP and finetune the CLIP."""

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None,
                 base_cate_name=None,
                 all_cate_name=None,
                 use_roialign=False):
        super(ClsFinetuner, self).__init__(init_cfg)
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

        # deal with the crop size and location
        self.test_crop_size_modi_ratio = self.test_cfg.get('crop_size_modi', 1.2) if self.test_cfg is not None else 1.2
        self.test_crop_loca_modi_ratio = self.test_cfg.get('crop_loca_modi', 0) if self.test_cfg is not None else 0

        self.train_crop_size_modi_ratio = self.train_cfg.get('crop_size_modi', 1.2) if self.train_cfg is not None else 1.2
        self.train_crop_loca_modi_ratio = self.train_cfg.get('crop_loca_modi', 0) if self.train_cfg is not None else 0   

        # deal with test with random bbox
        self.test_with_rand_bboxes = self.test_cfg.get('test_with_rand_bboxes', False) if self.test_cfg is not None else False
        self.num_of_rand_bboxes = self.test_cfg.get('num_of_rand_bboxes', 100) if self.test_cfg is not None else 100
        self.generate_bbox_feat = self.test_cfg.get('generate_bbox_feat', False) if self.test_cfg is not None else False
        self.generate_mix_gt_feat = self.test_cfg.get('generate_mix_gt_feat', False) if self.test_cfg is not None else False
        self.feat_save_path = self.test_cfg.get('feat_save_path', None) if self.test_cfg is not None else None
        # for loading the clip proposal
        self.use_pregenerated_proposal = self.test_cfg.get('use_pregenerated_proposal', None) if self.test_cfg is not None else None
        self.iou_calculator = BboxOverlaps2D()
        self.filter_low_iou_bboxes = self.test_cfg.get('filter_low_iou_bboxes', True) if self.test_cfg is not None else True
        self.use_base_novel_clip = self.test_cfg.get('use_base_novel_clip', None) if self.test_cfg is not None else None
        self.return_all_feats = self.test_cfg.get('return_all_feats', False) if self.test_cfg is not None else False
        
        # for with extra padding
        self.crop_with_extra_patches = self.test_cfg.get('crop_with_extra_patches', False) if self.test_cfg is not None else False
        self.target_patch_loca = self.test_cfg.get('target_patch_loca', 'center') if self.test_cfg is not None else 'center'
        self.extra_patches_num = self.test_cfg.get('extra_patches_num', 3) if self.test_cfg is not None else 3
        # filter the clip proposal using the categories
        self.filter_clip_proposal_base_on_cates = self.test_cfg.get('filter_clip_proposal_base_on_cates', False) if self.test_cfg is not None else False
        self.experiment_mode = self.test_cfg.get('experiment_mode', False) if self.test_cfg is not None else False

        # for the following self.from_cate_name_to_gt_idx, self.from_gt_idx_to_cate_idx, self.base_cate_name 
        # if base_cate_name == None, then now is for coco dataset, otherwise is for LVIS dataset
        if base_cate_name == None:
            self.from_cate_name_to_gt_idx = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 
                                            'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 
                                            'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 
                                            'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 
                                            'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 
                                            'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 
                                            'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 
                                            'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 
                                            'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 
                                            'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 
                                            'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
            self.from_gt_idx_to_cate_idx = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 
                                            13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 
                                            24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 
                                            35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 
                                            46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 
                                            57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 
                                            68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 
                                            79: 90}
            self.base_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
                                    'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
                                    'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
                                    'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
                                    'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
                                    'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
                                    'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
                                    'mouse', 'remote', 'microwave', 'oven', 'toaster', 
                                    'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
        elif base_cate_name == 'coco_base_only':
            self.from_cate_name_to_gt_idx = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 
                                            'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 
                                            'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 
                                            'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 
                                            'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 
                                            'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 
                                            'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 
                                            'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 
                                            'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 
                                            'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 
                                            'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
            self.from_gt_idx_to_cate_idx = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8, 6: 9, 7: 15, 8: 16, 9: 19, 10: 20, 11: 23, 12: 24, 
                                            13: 25, 14: 27, 15: 31, 16: 33, 17: 34, 18: 35, 19: 38, 20: 42, 21: 44, 22: 48, 23: 50, 
                                            24: 51, 25: 52, 26: 53, 27: 54, 28: 55, 29: 56, 30: 57, 31: 59, 32: 60, 33: 62, 34: 65, 
                                            35: 70, 36: 72, 37: 73, 38: 74, 39: 75, 40: 78, 41: 79, 42: 80, 43: 82, 44: 84, 45: 85, 
                                            46: 86, 47: 90}
            self.base_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
                                    'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
                                    'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
                                    'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
                                    'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
                                    'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
                                    'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
                                    'mouse', 'remote', 'microwave', 'oven', 'toaster', 
                                    'refrigerator', 'book', 'clock', 'vase', 'toothbrush')            
        else:
            # for lvis dataset
            self.base_cate_name = base_cate_name
            self.all_cate_name = all_cate_name
            self.from_gt_idx_to_cate_idx = {i:i+1 for i in range(1203)}
            self.from_cate_name_to_gt_idx = {name:idx for idx, name in enumerate(self.all_cate_name)}
            
        self.base_cate_gt_idx = [self.from_cate_name_to_gt_idx[name] for name in self.base_cate_name]
        # save the predicted cate and confidence
        self.save_cates_and_conf = self.test_cfg.get('save_cates_and_conf', False) if self.test_cfg is not None else False
        # generate the feature
        self.generate_gt_feat = self.test_cfg.get('generate_gt_feat', False) if self.test_cfg is not None else False
        self.rand_select_subset = self.test_cfg.get('rand_select_subset', False) if self.test_cfg is not None else False

        if self.generate_bbox_feat or self.generate_mix_gt_feat or self.rand_select_subset:
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)

        self.use_roialign = use_roialign
        if self.use_roialign:
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=224, sampling_ratio=0),
                out_channels=3,
                featmap_strides=[1])
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

        # for using clip to filter the pregenerated prediction(the predition may come from the vitdet final prediction)
        self.use_pregenerated_prediction = self.test_cfg.get('use_pregenerated_prediction', None) if self.test_cfg is not None else None
        self.num_classes = len(self.rpn_head.cate_names)

    def read_use_base_novel_clip(self, img_metas):
        file_name = img_metas[0]['ori_filename']
        # for lvis dataset
        if file_name.startswith('train2017'):
            file_name = file_name.split('/')[-1]
        # for VOC dataset
        if file_name.startswith('JPEGImages'):
            file_name = file_name.split('/')[-1]
            if 'VOC2007' in img_metas[0]['filename']:
                file_name = os.path.join('VOC2007', file_name)
            elif 'VOC2012' in img_metas[0]['filename']:
                file_name = os.path.join('VOC2012', file_name)
        pregenerate_prop_path = os.path.join(self.use_base_novel_clip, '.'.join(file_name.split('.')[:-1]) + '.json')
        pregenerated_bbox = json.load(open(pregenerate_prop_path))
        
        # preprocessing of the clip proposal
        clip_proposal = pregenerated_bbox['clip']
        if len(clip_proposal) != 0:
            clip_proposal = torch.tensor(clip_proposal).cuda()

            # filter the small bboxes
            w_smaller_than_36 = (clip_proposal[:, 2] - clip_proposal[:, 0]) < 36
            h_smaller_than_36 = (clip_proposal[:, 3] - clip_proposal[:, 1]) < 36
            clip_proposal[w_smaller_than_36, 2] = clip_proposal[w_smaller_than_36, 0] + 36
            clip_proposal[h_smaller_than_36, 3] = clip_proposal[h_smaller_than_36, 1] + 36
            
            # scale the bbox to the size of the image
            clip_proposal[:, :4] *= clip_proposal.new_tensor(img_metas[0]['scale_factor'])
            
            if len(pregenerated_bbox['base']) > 0:
                real_iou = self.iou_calculator(torch.tensor(pregenerated_bbox['base']).cuda(), clip_proposal)
                max_iou_per_proposal = torch.max(real_iou, dim=0)[0]
                all_iou_idx = (max_iou_per_proposal < 0.3)
                remained_bbox = clip_proposal[all_iou_idx]
            else:
                remained_bbox = clip_proposal
            
            # select the top 400 bboxes
            remained_bbox = remained_bbox[:400]
            remained_bbox = remained_bbox[:, :4]
        else:
            remained_bbox = torch.tensor([[1.0,2.0,3.0,4.0]]).cuda()

        # scale the gt bboxes
        all_gt_bboxes = pregenerated_bbox['base'] + pregenerated_bbox['novel']
        if len(all_gt_bboxes) != 0:
            all_gt_bboxes = torch.tensor(all_gt_bboxes).cuda()
            all_gt_bboxes[:, :4] *= all_gt_bboxes.new_tensor(img_metas[0]['scale_factor'])
        
        # concat all bbox
        if len(all_gt_bboxes) != 0:
            all_bboxes = torch.cat([all_gt_bboxes, remained_bbox], dim=0)
        else:
            all_bboxes = remained_bbox
        
        return all_bboxes

    def read_pregenerated_pred(self, img_metas):
        file_name = img_metas[0]['ori_filename']
        # for lvis dataset
        if file_name.startswith('train2017'):
            file_name = file_name.split('/')[-1]
        # for VOC dataset
        if file_name.startswith('JPEGImages'):
            file_name = file_name.split('/')[-1]
            if 'VOC2007' in img_metas[0]['filename']:
                file_name = os.path.join('VOC2007', file_name)
            elif 'VOC2012' in img_metas[0]['filename']:
                file_name = os.path.join('VOC2012', file_name)
        file_name = os.path.join(self.use_pregenerated_prediction, (file_name + '_final_pred' + '.json'))
        #proposal_file_name = os.path.join("/home/zhuoming/detectron_proposal1", (image_name + '_final_pred' + '.json'))
        
        # read the random bbox, the loaded bbox is xyxy format
        pregenerated_bbox = json.load(open(file_name))['box']
        pregenerated_bbox = torch.tensor(pregenerated_bbox).cuda()
        #print('in the loading', pregenerated_bbox.shape)
        # reshape the bboxes to the size of the image
        pregenerated_bbox[:, :4] *= pregenerated_bbox.new_tensor(img_metas[0]['scale_factor'])
        
        return pregenerated_bbox
            
    def read_pregenerated_bbox(self, img_metas, gt_bboxes, num_of_rand_bboxes):
        file_name = img_metas[0]['ori_filename']
        # for lvis dataset
        if file_name.startswith('train2017'):
            file_name = file_name.split('/')[-1]
        # for VOC dataset
        if file_name.startswith('JPEGImages'):
            file_name = file_name.split('/')[-1]
            if 'VOC2007' in img_metas[0]['filename']:
                file_name = os.path.join('VOC2007', file_name)
            elif 'VOC2012' in img_metas[0]['filename']:
                file_name = os.path.join('VOC2012', file_name)
        file_name = os.path.join(self.use_pregenerated_proposal, '.'.join(file_name.split('.')[:-1]) + '.json')
        # read the random bbox, the loaded bbox is xyxy format
        pregenerated_bbox = json.load(open(file_name))['score']
        pregenerated_bbox = torch.tensor(pregenerated_bbox).cuda()
        
        # # filter the small bboxes
        # w_smaller_than_36 = (pregenerated_bbox[:, 2] - pregenerated_bbox[:, 0]) < 36
        # h_smaller_than_36 = (pregenerated_bbox[:, 3] - pregenerated_bbox[:, 1]) < 36
        # pregenerated_bbox[w_smaller_than_36, 2] = pregenerated_bbox[w_smaller_than_36, 0] + 36
        # pregenerated_bbox[h_smaller_than_36, 3] = pregenerated_bbox[h_smaller_than_36, 1] + 36
        
        # # scale the bbox to the size of the image
        pregenerated_bbox[:, :4] *= pregenerated_bbox.new_tensor(img_metas[0]['scale_factor'])
        
        if self.filter_low_iou_bboxes and gt_bboxes[0].shape[0] != 0:
            real_iou = self.iou_calculator(gt_bboxes[0], pregenerated_bbox)
            max_iou_per_proposal = torch.max(real_iou, dim=0)[0]
            all_iou_idx = (max_iou_per_proposal < 0.3)
            remained_bbox = pregenerated_bbox[all_iou_idx]
        else:
            remained_bbox = pregenerated_bbox
        
        # random select
        #print('before random select:', remained_bbox.shape)
        if self.rand_select_subset and remained_bbox.shape[0] > num_of_rand_bboxes:
            random_choice = np.random.choice(remained_bbox.shape[0], num_of_rand_bboxes, replace=False)
            random_choice = torch.from_numpy(random_choice).cuda()
            remained_bbox = remained_bbox[random_choice]
        else:
            remained_bbox = remained_bbox[:num_of_rand_bboxes]
        #print('after random select:', remained_bbox.shape)
        # return the bbox in xyxy in torch tensor 
        return remained_bbox

    def generate_rand_bboxes(self, img_metas, num_of_rand_bbox):
        h, w, _ = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        # generate the top left position base on a evenly distribution
        rand_tl_x = torch.rand(num_of_rand_bbox, 1)
        rand_tl_y = torch.rand(num_of_rand_bbox, 1)
        
        # generate the w and the h base on the average and the std of w and h
        #w_mean: 103.89474514564517 h_mean: 107.41877275724094
        #w_std: 127.61796789111433 h_std: 114.85251970283936
        ratio_list = [0.5, 1, 2]
        rand_w = ((torch.randn(num_of_rand_bbox, 1) * 127.61796789111433) + 103.89474514564517) * np.max(scale_factor)
        rand_h = rand_w * ratio_list[random.randint(0, 2)]
        
        # make the w and h valid
        rand_w[rand_w < 36 * np.max(scale_factor)] = 36 * np.max(scale_factor)
        rand_h[rand_h < 36 * np.max(scale_factor)] = 36 * np.max(scale_factor)
        # handle the random bboxes
        real_tl_x = rand_tl_x * w
        real_tl_y = rand_tl_y * h
        
        now_rand_bbox = torch.cat([real_tl_x, real_tl_y, real_tl_x + rand_w, real_tl_y + rand_h], dim=-1)
        
        return now_rand_bbox

    def default_zero_padding(self, now_patch):
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
        else:
            now_patch = now_patch
        return now_patch

    def pad_on_specific_side(self, now_patch):
        gt_h, gt_w, c = now_patch.shape
        # implement zero padding at the top left side
        if gt_h != gt_w:
            long_edge = max((gt_h, gt_w))
            empty_patch = np.zeros((long_edge, long_edge, 3))
            if gt_h > gt_w:
                x_start = long_edge - gt_w
                empty_patch[:, x_start:] = now_patch
            else:
                y_start = long_edge - gt_h
                empty_patch[y_start:] = now_patch
            now_patch = empty_patch
        else:
            now_patch = now_patch
        return now_patch
        
    def cropping_with_extra_patches(self, img, bbox, H, W):
        # the original bbox location
        tl_x, tl_y, br_x, br_y = bbox[0], bbox[1], bbox[2], bbox[3]
        x = tl_x
        y = tl_y
        w = br_x - tl_x
        h = br_y - tl_y
        # deal with some special case in rpn proposal which width or highth is smaller than 1
        if w < 10.0:
            w = 10.0
        if h < 10.0:
            h = 10.0

        empty_patch = np.zeros((self.extra_patches_num * math.ceil(h) + 1, self.extra_patches_num * math.ceil(w) + 1, 3))
        if self.target_patch_loca == 'center':
            # for the target patch at the center
            x_start_pos = math.floor(max(x - (self.extra_patches_num-1)/2*w, 0))
            y_start_pos = math.floor(max(y - (self.extra_patches_num-1)/2*h, 0))
            x_end_pos = math.ceil(min(x + w + (self.extra_patches_num-1)/2*w, W-1))
            y_end_pos = math.ceil(min(y + h + (self.extra_patches_num-1)/2*h, H-1))
            valid_cropped_region = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]
            
            # map the region back to the empty patches
            x_start_pos = math.floor(max(w-x, 0))
            y_start_pos = math.floor(max(h-y, 0))
            x_end_pos = x_start_pos + valid_cropped_region.shape[1]
            y_end_pos = y_start_pos + valid_cropped_region.shape[0]
            
        elif self.target_patch_loca == 'br':
            # for the target patch at br
            x_start_pos = math.floor(max(x - (self.extra_patches_num-1)*w, 0))
            y_start_pos = math.floor(max(y - (self.extra_patches_num-1)*h, 0))
            x_end_pos = math.ceil(min(x + w, W-1))
            y_end_pos = math.ceil(min(y + h, H-1))
            valid_cropped_region = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]
            
            # map the region back to the empty patches
            x_start_pos = math.floor(max((self.extra_patches_num-1)*w-x, 0))
            y_start_pos = math.floor(max((self.extra_patches_num-1)*h-y, 0))
            x_end_pos = x_start_pos + valid_cropped_region.shape[1]
            y_end_pos = y_start_pos + valid_cropped_region.shape[0]
            
        empty_patch[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :] = valid_cropped_region
        
        return empty_patch        

    def cropping_with_purturb(self, img, bbox, crop_size_modi_ratio, crop_loca_modi_ratio, H, W):
        # the original bbox location
        tl_x, tl_y, br_x, br_y = bbox[0], bbox[1], bbox[2], bbox[3]
        x = tl_x
        y = tl_y
        w = br_x - tl_x
        h = br_y - tl_y
        # deal with some special case in rpn proposal which width or highth is smaller than 1
        if w < 10.0:
            w = 10.0
        if h < 10.0:
            h = 10.0        
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

        now_patch = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]
        
        return now_patch
      
    def crop_img_to_patches(self, imgs, gt_bboxes, img_metas, remain_same_bbox_size=False):
        # # for testing
        # now_patch = imgs.permute(0, 2, 3, 1)[0].numpy()
        # if not os.path.exists('/home/zhuoming/square_image/'):
        #     os.makedirs('/home/zhuoming/square_image/')
        # data = Image.fromarray(np.uint8(now_patch))
        # data.save('/home/zhuoming/square_image/' + img_metas[0]['ori_filename'] + '.png')

        # handle the test config
        if self.training: 
            crop_size_modi_ratio = self.train_crop_size_modi_ratio
            crop_loca_modi_ratio = self.train_crop_loca_modi_ratio
        else:
            crop_size_modi_ratio = self.test_crop_size_modi_ratio if remain_same_bbox_size == False else 1.0
            crop_loca_modi_ratio = self.test_crop_loca_modi_ratio            
        
        ########### for the original implementation ########### 
        if not self.use_roialign:
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
                result = []
                for box_i, bbox in enumerate(all_gt_bboxes):      
                    # crop the image
                    if self.crop_with_extra_patches:
                        now_patch = self.cropping_with_extra_patches(img, bbox, H, W)
                    else:
                        now_patch = self.cropping_with_purturb(img, bbox, crop_size_modi_ratio, crop_loca_modi_ratio, H, W)
                    
                    # pad the patches
                    if self.target_patch_loca == 'br':
                        now_patch = self.pad_on_specific_side(now_patch)
                    else:
                        now_patch = self.default_zero_padding(now_patch)

                    #new_patch, w_scale, h_scale = mmcv.imresize(now_patch, (224, 224), return_scale=True)
                    # convert the numpy to PIL image
                    PIL_image = Image.fromarray(np.uint8(now_patch))
                    # do the preprocessing
                    new_patch = self.preprocess(PIL_image)
                    #print('new patch', new_patch, new_patch.shape, 'img_meta["filename"]', img_metas[img_idx]['filename'])
                    #temp_file_name = img_metas[img_idx]['ori_filename'].split('.')[0] + '_reshape_and_norm_' + '.pt'
                    #torch.save(new_patch, os.path.join('/data/zhuoming/detection/test/test2', temp_file_name))
                    
                    #image_result.append(np.expand_dims(new_patch, axis=0))
                    #if bbox[0] == 126.62 and bbox[1] == 438.82:
                    #    x = self.backbone(new_patch.unsqueeze(dim=0).cuda())
                    #    print(x)
                    
                    result.append(new_patch.unsqueeze(dim=0))
                result = torch.cat(result, dim=0)
                all_results.append(result)
        else:
            #### testing the 
            gt_num_per_img = [len(ele) for ele in gt_bboxes]
            rois = bbox2roi([ele.cpu() for ele in gt_bboxes])
            #print(type(imgs[0,0,0,0]))
            with torch.no_grad():
                bbox_feats = self.bbox_roi_extractor((imgs.float(),), rois, roi_scale_factor=crop_size_modi_ratio)
            #print('bbox_feats', bbox_feats.shape, bbox_feats)
            # normalize
            bbox_feats = F.normalize(bbox_feats, MEANS, STDS)
            
            # split the bbox result
            all_results = bbox_feats.split(gt_num_per_img, 0)
            #print(len(all_results), [ele.shape for ele in all_results])
            ### print the image
            # for all_res_per_img, img_meta in zip(all_results, img_metas):
            #     all_res_per_img = all_res_per_img.permute(0, 2, 3, 1).numpy()
            #     for i, result in enumerate(all_res_per_img):
            #         data = Image.fromarray(np.uint8(result))
            #         data.save('/home/zhuoming/roialigned/' + img_meta['ori_filename'] + "_" + str(i) + '.png')
        #cropped_patches = np.concatenate(result, axis=0)
        # the shape of the cropped_patches: torch.Size([gt_num_in_batch, 3, 224, 224])
        #cropped_patches = torch.cat(result, dim=0).cuda()
        return all_results

    def extract_feat(self, img, gt_bboxes, cropped_patches=None, img_metas=None, remain_same_bbox_size=False):
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
        #cropped_patches_list:len = batch_size, list[tensor] each tensor shape [gt_num_of_image, 3, 224, 224]
        if cropped_patches == None:
            ########### for the original implementation ########### 
            #cropped_patches_list = self.crop_img_to_patches(img.cpu(), gt_bboxes, img_metas, remain_same_bbox_size=remain_same_bbox_size)
            cropped_patches_list = self.crop_img_to_patches(img.cpu(), gt_bboxes, img_metas, remain_same_bbox_size=remain_same_bbox_size)
        else:
            print('testing cropped_patches')
            cropped_patches_list = cropped_patches

        # convert dimension from [bs, 64, 3, 224, 224] to [bs*64, 3, 224, 224]
        #converted_img_patches = converted_img_patches.view(bs, -1, self.backbone.input_resolution, self.backbone.input_resolution)

        # the input of the vision transformer should be torch.Size([64, 3, 224, 224])
        result_list = []
        for patches in cropped_patches_list:
            if self.backbone.__class__.__name__ == 'myVisionTransformer':
                x = self.backbone(patches.cuda(), return_all_feats=self.return_all_feats)
            else:
                x = self.backbone(patches.cuda())
            if self.with_neck:
                x = self.neck(x)
            result_list.append(x)
        # convert the feature [bs*64, 512] to [bs, 64, 512]
        #x = x.view(bs, -1, x.shape[-1])
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

        x = self.extract_feat(img, gt_bboxes, img_metas=img_metas)
        # x: list[tensor] each tensor shape [gt_num_of_image, 512]
        losses = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels,
                                             gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, gt_bboxes, gt_labels, cropped_patches=None, rescale=False):
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
        
        # generate the gt only
        if not self.generate_bbox_feat and self.generate_gt_feat:
            # obtain the gt file path
            gt_save_root = os.path.join(self.feat_save_path, 'gt')
            #file_name = img_metas[0]['ori_filename'].split('.')[0] + '.json'
            file_name = img_metas[0]['ori_filename']
            # for lvis dataset
            if file_name.startswith('train2017'):
                file_name = file_name.split('/')[-1]
            # for VOC dataset
            if file_name.startswith('JPEGImages'):
                file_name = file_name.split('/')[-1]
                if 'VOC2007' in img_metas[0]['filename']:
                    file_name = os.path.join('VOC2007', file_name)
                elif 'VOC2012' in img_metas[0]['filename']:
                    file_name = os.path.join('VOC2012', file_name)
            file_name = file_name.split('.')[0] + '.json'
            
            gt_file_path = os.path.join(gt_save_root, file_name)
            gt_save_root = '/'.join(gt_file_path.split('/')[:-1])
            if not os.path.exists(gt_save_root):
                os.makedirs(gt_save_root)
                
            # if the file has been created, skip this image
            if os.path.exists(gt_file_path):
                return [np.zeros((1,5))]
            
            # handle the image metas
            my_img_meta = img_metas[0]
            my_img_meta['scale_factor'] = my_img_meta['scale_factor'].tolist()
            my_img_meta['img_norm_cfg']['mean'] = my_img_meta['img_norm_cfg']['mean'].tolist()
            my_img_meta['img_norm_cfg']['std'] = my_img_meta['img_norm_cfg']['std'].tolist()
        
            # generate the gt feat
            if len(gt_bboxes) != 0:
                x = self.extract_feat(img, gt_bboxes, cropped_patches, img_metas=img_metas)
                # save the rand_bbox and the feat, img_metas
                file = open(gt_file_path, 'w')
                #print(type(gt_bboxes), type(gt_labels))
                #print('gt', x[0].shape, gt_bboxes[0].shape, gt_labels[0].shape)
                result_json = {'feat':x[0].cpu().tolist() if len(x)!=0 else [], 'bbox':gt_bboxes[0].cpu().tolist() if len(gt_bboxes)!=0 else [], 'gt_labels':gt_labels[0].cpu().tolist() if len(gt_labels)!=0 else [], 'img_metas':my_img_meta}
                #print('testing gt json', result_json)
                file.write(json.dumps(result_json))
                file.close()
                
            return [torch.zeros(10, 4)]        
        
        if self.generate_mix_gt_feat:
            # obtain the gt file path
            gt_save_root = os.path.join(self.feat_save_path, 'mix_gt')
            if not os.path.exists(gt_save_root):
                os.makedirs(gt_save_root)
            #file_name = img_metas[0]['ori_filename'].split('.')[0] + '.json'
            file_name = img_metas[0]['ori_filename']
            # for lvis dataset
            if file_name.startswith('train2017'):
                file_name = file_name.split('/')[-1]
            # for VOC dataset
            if file_name.startswith('JPEGImages'):
                file_name = file_name.split('/')[-1]
                if 'VOC2007' in img_metas[0]['filename']:
                    file_name = os.path.join('VOC2007', file_name)
                elif 'VOC2012' in img_metas[0]['filename']:
                    file_name = os.path.join('VOC2012', file_name)
            file_name = file_name.split('.')[0] + '.json'
            
            gt_file_path = os.path.join(gt_save_root, file_name)
            
            # if the file has been created, skip this image
            if os.path.exists(gt_file_path):
                return [np.zeros((1,5))]
            
            # handle the image metas
            my_img_meta = img_metas[0]
            my_img_meta['scale_factor'] = my_img_meta['scale_factor'].tolist()
            my_img_meta['img_norm_cfg']['mean'] = my_img_meta['img_norm_cfg']['mean'].tolist()
            my_img_meta['img_norm_cfg']['std'] = my_img_meta['img_norm_cfg']['std'].tolist()
        
            # generate the gt feat
            if len(gt_bboxes) != 0:
                self.test_crop_size_modi_ratio = 1.0
                ori_size_x = self.extract_feat(img, gt_bboxes, cropped_patches, img_metas=img_metas)
                self.test_crop_size_modi_ratio = 1.5
                enlarge_size_x = self.extract_feat(img, gt_bboxes, cropped_patches, img_metas=img_metas)
                #print(len(ori_size_x), len(enlarge_size_x), ori_size_x[0], enlarge_size_x[0])
                x = (ori_size_x[0] + enlarge_size_x[0]) / 2
                # save the rand_bbox and the feat, img_metas
                file = open(gt_file_path, 'w')
                #print(type(gt_bboxes), type(gt_labels))
                #print('gt', x[0].shape, gt_bboxes[0].shape, gt_labels[0].shape)
                result_json = {'feat':x.cpu().tolist() if len(x)!=0 else [], 'bbox':gt_bboxes[0].cpu().tolist() if len(gt_bboxes)!=0 else [], 'gt_labels':gt_labels[0].cpu().tolist() if len(gt_labels)!=0 else [], 'img_metas':my_img_meta}
                #print('testing gt json', result_json)
                file.write(json.dumps(result_json))
                file.close()
            
            return [torch.zeros(10, 4)]
                
        if self.generate_bbox_feat:
            # obtain the gt file path
            gt_save_root = os.path.join(self.feat_save_path, 'gt')
            #file_name = img_metas[0]['ori_filename'].split('.')[0] + '.json'
            file_name = img_metas[0]['ori_filename']
            # for lvis dataset
            if file_name.startswith('train2017'):
                file_name = file_name.split('/')[-1]
            # for VOC dataset
            if file_name.startswith('JPEGImages'):
                file_name = file_name.split('/')[-1]
                if 'VOC2007' in img_metas[0]['filename']:
                    file_name = os.path.join('VOC2007', file_name)
                elif 'VOC2012' in img_metas[0]['filename']:
                    file_name = os.path.join('VOC2012', file_name)
            file_name = file_name.split('.')[0] + '.json'
            
            gt_file_path = os.path.join(gt_save_root, file_name)
            gt_save_root = '/'.join(gt_file_path.split('/')[:-1])
            if not os.path.exists(gt_save_root):
                os.makedirs(gt_save_root)
            
            # obtain the random file path
            random_save_root = os.path.join(self.feat_save_path, 'random')
            #file_name = img_metas[0]['ori_filename'].split('.')[0] + '.json'
            random_file_path = os.path.join(random_save_root, file_name)
            random_save_root = '/'.join(random_file_path.split('/')[:-1])
            if not os.path.exists(random_save_root):
                os.makedirs(random_save_root)
            
            # if the file has been created, skip this image
            if (not self.generate_gt_feat and os.path.exists(random_file_path)) or (self.generate_gt_feat and os.path.exists(gt_file_path) and os.path.exists(random_file_path)):
                return [np.zeros((1,5))]
            
            if self.use_pregenerated_proposal != None:
                now_rand_bbox = self.read_pregenerated_bbox(img_metas, gt_bboxes, self.num_of_rand_bboxes)
            elif self.use_base_novel_clip != None:
                now_rand_bbox = self.read_use_base_novel_clip(img_metas)
            else:
                # generate the random feat
                now_rand_bbox = self.generate_rand_bboxes(img_metas, self.num_of_rand_bboxes)
            
            x = self.extract_feat(img, [now_rand_bbox], cropped_patches, img_metas=img_metas)
            
            # for experiments(imagenet ranking)
            if self.experiment_mode:
                ### get the objectness ranking
                #print('now_rand_bbox', now_rand_bbox)
                _, objectness_indices = torch.sort(now_rand_bbox[:, -1], descending=True)
                #print('objectness_indices', objectness_indices)
                #get the objectness rank for each bbox
                all_objectness_rank = []
                for i in range(now_rand_bbox.shape[0]):
                    now_pos = (objectness_indices == i).nonzero(as_tuple=True)
                    #print('now_pos', now_pos)
                    now_pos = now_pos[0][0].item()
                    all_objectness_rank.append(now_pos)
                #print('all_objectness_rank', all_objectness_rank)
                all_objectness_rank = torch.tensor(all_objectness_rank).unsqueeze(dim=-1).cuda()
                
                ### get the base ranking
                sigmoid = nn.Sigmoid()
                # forward of this head requires img_metas
                outs = self.rpn_head.forward(x, img_metas)
                # get the classsification score
                pred_after_sigmoid = sigmoid(outs[0])
                # get the max confidence categories
                max_val, pred_idx = torch.max(pred_after_sigmoid, dim=1)
                _, base_indices = torch.sort(max_val, descending=True)
                #print('base_indices', base_indices)
                #get the base rank for each bbox
                all_base_rank = []
                for i in range(now_rand_bbox.shape[0]):
                    now_pos = (base_indices == i).nonzero(as_tuple=True)
                    now_pos = now_pos[0][0].item()
                    all_base_rank.append(now_pos)
                #print('all_base_rank', all_base_rank)
                all_base_rank = torch.tensor(all_base_rank).unsqueeze(dim=-1).cuda()
                
                ### save the result ranking
                # scale back the bbox 
                origin_bbox = now_rand_bbox[:, :4] / torch.from_numpy(img_metas[0]['scale_factor']).cuda()
                final_res = torch.cat([origin_bbox, all_objectness_rank, all_base_rank], dim=-1)
                
                file = open(random_file_path, 'w')

                #print('random', now_rand_bbox[:, -2])
                result_json = {'res':final_res.cpu().tolist()}
                #print('testing random json', result_json)
                file.write(json.dumps(result_json))
                file.close()
                return [torch.zeros(10, 4)]
            
            
            # filter the clip proposal base on the categories
            if self.filter_clip_proposal_base_on_cates or self.save_cates_and_conf:
                # out list[tensor] tensor with shape [gt_per_img, channel]
                softmax = nn.Softmax(dim=1)

                # select the needed feat
                if len(x[0].shape) > 2 and self.rpn_head.selected_need_feat is not None:
                    x = [feat[:, self.rpn_head.selected_need_feat, :] for feat in x]
                    if x[0].shape[1] == 1:
                        x = [feat.squeeze(dim=1) for feat in x]

                # forward of this head requires img_metas
                outs = self.rpn_head.forward(x, img_metas)
                # get the classsification score
                pred_after_softmax = softmax(outs[0])
                
                # get the max confidence categories
                max_val, pred_idx = torch.max(pred_after_softmax, dim=1)
                
                # add the confidence score (the score after softmax for the max confidence score) replace the original score
                if now_rand_bbox.shape[-1] == 5:
                    objectness_score = now_rand_bbox.clone()[:, -1]
                    objectness_score = objectness_score.unsqueeze(dim=-1)
                    now_rand_bbox[:, -1] = max_val
                elif now_rand_bbox.shape[-1] == 4:
                    objectness_score = None
                    now_rand_bbox = torch.cat([now_rand_bbox, max_val.unsqueeze(dim=-1)], dim=-1)
                else:
                    print('now_rand_bbox.shape[1] is not equal to 4 or 5')
                
                # add the max confidence coco cate id (convert from the gt idx to the coco id)
                all_coco_idx = torch.tensor([self.from_gt_idx_to_cate_idx[ele.item()] for ele in pred_idx]).cuda()
                now_rand_bbox = torch.cat([now_rand_bbox, all_coco_idx.unsqueeze(dim=-1)], dim=-1)
                
                # append the objectness_score
                if objectness_score is not None:
                    now_rand_bbox = torch.cat([now_rand_bbox, objectness_score], dim=-1)

                if self.filter_clip_proposal_base_on_cates:
                    # get the idx which is not predicted as base categories
                    all_novel_idx = None
                    for base_cate_idx in self.base_cate_gt_idx:
                        if all_novel_idx == None:
                            all_novel_idx = (pred_idx != base_cate_idx)
                        else:
                            temp_idx = (pred_idx != base_cate_idx)
                            all_novel_idx = torch.logical_and(all_novel_idx, temp_idx)
                    
                    # filter the bboxes and feature(assuming the batch size is 1)
                    #print('before filter:', now_rand_bbox.shape)
                    x = [x[0][all_novel_idx]]
                    now_rand_bbox = now_rand_bbox[all_novel_idx]
                    
                    
                #sort the feat base on the confidence score
                # _, confi_indices = torch.sort(now_rand_bbox[:, -2], descending=True)
                # x = [x[0][confi_indices]]
                # now_rand_bbox = now_rand_bbox[confi_indices]
                
                if self.filter_clip_proposal_base_on_cates:
                    #print('after filter:', now_rand_bbox.shape)
                    # make the number of remaining bbox become self.num_of_rand_bboxes
                    x = [x[0][:300]]
                    now_rand_bbox = now_rand_bbox[:300]
            
            # save the rand_bbox and the feat, img_metas
            file = open(random_file_path, 'w')
            # handle the image metas
            my_img_meta = img_metas[0]
            my_img_meta['scale_factor'] = my_img_meta['scale_factor'].tolist()
            my_img_meta['img_norm_cfg']['mean'] = my_img_meta['img_norm_cfg']['mean'].tolist()
            my_img_meta['img_norm_cfg']['std'] = my_img_meta['img_norm_cfg']['std'].tolist()
            
            #print('random', now_rand_bbox[:, -2])
            result_json = {'feat':x[0].cpu().tolist(), 'bbox':now_rand_bbox.cpu().tolist(), 'img_metas':my_img_meta}
            #print('testing random json', result_json)
            file.write(json.dumps(result_json))
            file.close()
            
            # generate the gt feat
            if self.generate_gt_feat and len(gt_bboxes) != 0:
                x = self.extract_feat(img, gt_bboxes, cropped_patches, img_metas=img_metas)
                # save the rand_bbox and the feat, img_metas
                file = open(gt_file_path, 'w')
                #print(type(gt_bboxes), type(gt_labels))
                #print('gt', x[0].shape, gt_bboxes[0].shape, gt_labels[0].shape)
                result_json = {'feat':x[0].cpu().tolist() if len(x)!=0 else [], 'bbox':gt_bboxes[0].cpu().tolist() if len(gt_bboxes)!=0 else [], 'gt_labels':gt_labels[0].cpu().tolist() if len(gt_labels)!=0 else [], 'img_metas':my_img_meta}
                #print('testing gt json', result_json)
                file.write(json.dumps(result_json))
                file.close()
            
            return [torch.zeros(10, 4)]
            
        elif self.test_with_rand_bboxes:
            now_rand_bbox = self.generate_rand_bboxes(img_metas, self.num_of_rand_bboxes)
            x = self.extract_feat(img, [now_rand_bbox], cropped_patches, img_metas=img_metas)
        else:
            if self.use_pregenerated_prediction != None:
                #print('proposal_list', [ele.shape for ele in proposal_list])
                file_name = img_metas[0]['ori_filename']
                # for lvis dataset
                if file_name.startswith('train2017'):
                    file_name = file_name.split('/')[-1]
                # for VOC dataset
                if file_name.startswith('JPEGImages'):
                    file_name = file_name.split('/')[-1]
                    if 'VOC2007' in img_metas[0]['filename']:
                        file_name = os.path.join('VOC2007', file_name)
                    elif 'VOC2012' in img_metas[0]['filename']:
                        file_name = os.path.join('VOC2012', file_name)
                file_name = os.path.join(self.use_pregenerated_prediction, (file_name + '_clip_pred_' + str(self.num_classes) + '.json'))
                if os.path.exists(file_name):
                    return [np.zeros((1,5))]
                
                gt_bboxes = [self.read_pregenerated_pred(img_metas)]
            x = self.extract_feat(img, gt_bboxes, cropped_patches, img_metas=img_metas)
        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        if len(x) == 0:
            return [torch.zeros(10, 4)]
        proposal_list = self.rpn_head.simple_test_bboxes(x, gt_labels, img_metas, gt_bboxes)
        if self.use_pregenerated_prediction != None:
            #proposal_file_name = os.path.join("/home/zhuoming/detectron_proposal1", (image_name + '_final_pred' + '.json'))
            result = {'score': proposal_list[0].cpu().tolist()}
            #print(file_name)
            file = open(file_name, 'w')
            file.write(json.dumps(result))
            file.close()                
            
        #if rescale:
        #    for proposals, meta in zip(proposal_list, img_metas):
        #        proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])
        if torch.onnx.is_in_onnx_export():
            return proposal_list

        return [proposal.cpu().numpy() for proposal in proposal_list]

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
