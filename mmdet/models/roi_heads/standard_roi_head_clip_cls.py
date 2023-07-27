# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_loss, build_backbone
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import math
import random
import numpy as np
import mmcv
import os
import json
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
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

@HEADS.register_module()
class StandardRoIHeadCLIPCls(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """This is an unused class for using clip for final classification."""

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
        
    def init_extra_backbone(self, extra_backbone):
        self.clip_backbone = build_backbone(extra_backbone)
        self.preprocess = _transform(self.clip_backbone.input_resolution)


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
                      **kwargs):
        pass

    def crop_img_to_patches(self, imgs, gt_bboxes, img_metas):
        # handle the test config
        #if self.training: 
        #    crop_size_modi_ratio = self.train_crop_size_modi_ratio
        #    crop_loca_modi_ratio = self.train_crop_loca_modi_ratio
        #else:
        crop_size_modi_ratio = 1.5
        crop_loca_modi_ratio = 0
        
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
                
                # data = Image.fromarray(np.uint8(now_patch))
                # data.save('data/mask_rcnn_clip_classifier/' + img_metas[img_idx]['ori_filename'] + '_' + str(box_i) + '.png')
                #new_patch, w_scale, h_scale = mmcv.imresize(now_patch, (224, 224), return_scale=True)
                # convert the numpy to PIL image
                PIL_image = Image.fromarray(np.uint8(now_patch))
                # do the preprocessing
                new_patch = self.preprocess(PIL_image)
                #image_result.append(np.expand_dims(new_patch, axis=0))
                #if bbox[0] == 126.62 and bbox[1] == 438.82:
                #    x = self.backbone(new_patch.unsqueeze(dim=0).cuda())
                #    print(x)
                
                result.append(new_patch.unsqueeze(dim=0))
            result = torch.cat(result, dim=0)
            all_results.append(result)

        #cropped_patches = np.concatenate(result, axis=0)
        # the shape of the cropped_patches: torch.Size([gt_num_in_batch, 3, 224, 224])
        #cropped_patches = torch.cat(result, dim=0).cuda()
        return all_results

    def extract_feat(self, img, gt_bboxes, cropped_patches=None, img_metas=None):
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
        
        # denormalize the image
        #'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32)
        #img_mean = torch.from_numpy(img_metas[0]['img_norm_cfg']['mean']).cuda()
        #img_std = torch.from_numpy(img_metas[0]['img_norm_cfg']['std']).cuda()
        img_mean = img_metas[0]['img_norm_cfg']['mean']
        img_std = img_metas[0]['img_norm_cfg']['std']
        
        img = img.permute([0,2,3,1])[0].cpu().numpy() 
        #img = img * img_std + img_mean
        img = mmcv.imdenormalize(img, img_mean, img_std, to_bgr=False)
        img = torch.from_numpy(img).permute([2,0,1]).unsqueeze(dim=0).cuda()
        
        # # # test for normalization
        # real_image_name = img_metas[0]['filename'].split('/')[-1]
        # file_path = os.path.join("/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/data/mask_rcnn_clip_classifier/", (real_image_name + 'now.pt'))
        # torch.save(img.cpu(), file_path)
        
        
        # #/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/data/mask_rcnn_clip_classifier/data/coco/train2017/000000531244.jpg.pt
        
        # image_before_norm = torch.load(file_path)
        # #img torch.Size([1, 3, 800, 1088]) image_before_norm torch.Size([800, 1067, 3])
        # image_before_norm = image_before_norm.permute(2,0,1).cuda()
        # test_image = img[0][:image_before_norm.shape[0], :image_before_norm.shape[1], :image_before_norm.shape[2]]
        # print('not match:', (False in (test_image == image_before_norm)), torch.sum((test_image == image_before_norm)), test_image.shape[0]*test_image.shape[1]*test_image.shape[2])
        
        
        if cropped_patches == None:
            cropped_patches_list = self.crop_img_to_patches(img.cpu(), gt_bboxes, img_metas)
        else:
            print('testing cropped_patches')
            cropped_patches_list = cropped_patches

        # convert dimension from [bs, 64, 3, 224, 224] to [bs*64, 3, 224, 224]
        #converted_img_patches = converted_img_patches.view(bs, -1, self.backbone.input_resolution, self.backbone.input_resolution)

        # the input of the vision transformer should be torch.Size([64, 3, 224, 224])
        result_list = []
        for patches in cropped_patches_list:
            x = self.clip_backbone(patches.cuda())
            result_list.append(x)
        # convert the feature [bs*64, 512] to [bs, 64, 512]
        #x = x.view(bs, -1, x.shape[-1])
        # for param_name in self.clip_backbone.state_dict():
        #     print(param_name, self.clip_backbone.state_dict()[param_name])
            
        return result_list

    
    def _bbox_forward(self, x, rois, img=None, img_metas=None):
        """this is for test forward only, since bbox_pred.split(num_proposals_per_img, 0) is not fit for hte train forward."""  
        # is the number of feat map layer
        #if distilled_feat != None and gt_rand_rois != None:
            # gt and random bbox feat from backbone_to
            # gt_and_rand_bbox_feat: torch.Size([1024, 256, 7, 7])
        #    gt_and_rand_bbox_feat = self.bbox_roi_extractor(
        #        x[:self.bbox_roi_extractor.num_inputs], gt_rand_rois)
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
            gt_and_rand_bbox_feat = self.shared_head(gt_and_rand_bbox_feat)
        
        cls_score, bbox_pred, return_bbox_feats = self.bbox_head(bbox_feats)
        ## bbox_pred is the prediction result on the scaled image, bbox_pred torch.Size([1000, 4])
        ## the input img in this function is also a scaled, torch.Size([1, 3, 800, 1216])
        
        # send the predict bbox to the CLIP backbone, obtained the feat
        num_proposals_per_img = tuple(bbox_pred.shape[0] for i in range(img.shape[0]))
        splited_rois = rois.split(num_proposals_per_img, 0)
        splited_bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        
        # enumerate the inner
        reshaped_bbox_pred = [self.bbox_head.bbox_coder.decode(
                roi_per_img[..., 1:], bbox_per_img, max_shape=img_meta_per_img['img_shape']) for roi_per_img, bbox_per_img, img_meta_per_img in zip(splited_rois, splited_bbox_pred, img_metas)  ] 
        #reshaped_bbox_pred = reshaped_bbox_pred.unsqueeze(dim=0)
        
        clip_bbox_feat = self.extract_feat(img, reshaped_bbox_pred, img_metas=img_metas)
        #print(type(clip_bbox_feat), len(clip_bbox_feat), clip_bbox_feat[0].shape)
        clip_bbox_feat = torch.cat(clip_bbox_feat, dim=0)
        clip_bbox_feat = clip_bbox_feat / clip_bbox_feat.norm(dim=-1, keepdim=True)
        logit_scale = torch.ones([]) * np.log(1 / 0.07)
        logit_scale = logit_scale.exp()
        clip_bbox_feat = clip_bbox_feat * logit_scale
        
        # use the bbox head classifier to classifier the model
        clip_cls_score = self.bbox_head.fc_cls_fg(clip_bbox_feat)
        
        # add one extra demension for the bg
        additional_bg_score = torch.full([clip_cls_score.shape[0], 1], -float('inf')).cuda()
        clip_cls_score = torch.cat([clip_cls_score, additional_bg_score], dim=-1)
        
        # replace the classification score with the clip score
        # cls_score = torch.Size([1000, 18])
        origin_cls_score = cls_score
        
        cls_score=clip_cls_score 
        
        # calculate the cosine similarity
        # cos_result = {'our_cos_score': origin_cls_score.detach().cpu().tolist(), 
        #               'clip_cos_score':clip_cls_score.detach().cpu().tolist(), 
        #               'our_clip_score':torch.sum(return_bbox_feats * clip_bbox_feat, dim=-1).detach().cpu().tolist()}
        # print('our_cos_score', torch.mean(origin_cls_score.max(dim=-1)[0]),
        #       'clip_cos_score', torch.mean(clip_cls_score.max(dim=-1)[0]))

        # file_name = os.path.join('data/mask_rcnn_clip_classifier/base/', '.'.join(img_metas[0]['ori_filename'].split('.')[:-1]) + '.json')
        # file = open(file_name, 'w')
        # file.write(json.dumps(cos_result))
        # file.close()
        
        bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, clip_infer_bbox=reshaped_bbox_pred)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, distilled_feat, rand_bboxes):
        pass

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        pass

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
                    img=None):
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
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, img=img)

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
