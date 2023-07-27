# Copyright (c) OpenMMLab. All rights reserved.
import sys
import warnings

import numpy as np
import os
import torch
import json
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed

iou_calculator = BboxOverlaps2D()

class BBoxTestMixin:

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    **kwargs):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           **kwargs):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        if self.use_pregenerated_proposal is not None:
            pregenerate_prop_path = os.path.join(self.use_pregenerated_proposal, '.'.join(img_metas[0]['ori_filename'].split('.')[:-1]) + '.json')
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
                    real_iou = iou_calculator(torch.tensor(pregenerated_bbox['base']).cuda(), clip_proposal)
                    max_iou_per_proposal = torch.max(real_iou, dim=0)[0]
                    all_iou_idx = (max_iou_per_proposal < 0.3)
                    remained_bbox = clip_proposal[all_iou_idx]
                else:
                    remained_bbox = clip_proposal
                
                # select the top 400 bboxes
                remained_bbox = remained_bbox[:400]
                remained_bbox = remained_bbox[:, :4]
            else:
                remained_bbox = torch.zeros(1,4).cuda()

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
                
            # pad to 1000
            if len(all_bboxes) < 1000:
                padded_len = 1000 - len(all_bboxes)
                padded_results = torch.zeros([padded_len] + list(all_bboxes.shape[1:])).cuda()
                all_bboxes = torch.cat([all_bboxes, padded_results], dim=0)
                proposals = [all_bboxes]
            rois = bbox2roi(proposals)
        else:
            rois = bbox2roi(proposals)
        
        merge_proposal_obj_score = rcnn_test_cfg.get('merge_proposal_obj_score', False) if rcnn_test_cfg is not None else False
        if merge_proposal_obj_score:
            proposal_obj_score = [ele[:, -1] for ele in proposals]
        else:
            proposal_obj_score = [None for ele in proposals]
            
        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size


        num_proposals_per_img = tuple(len(p) for p in proposals)
        if type(self).__name__ == 'StandardRoIHeadCLIPCls': 
            kwargs['img_metas'] = img_metas
            bbox_results = self._bbox_forward(x, rois, **kwargs)
        elif type(self).__name__.startswith('StandardRoIHeadDistillWithTransformer') or type(self).__name__ == 'StandardRoIHeadWithTransformer':
            kwargs['img_metas'] = img_metas
            kwargs['bboxes_num'] = num_proposals_per_img
            bbox_results = self._bbox_forward(x, rois, **kwargs)
        elif self.save_the_feat:
            bbox_results = self._bbox_forward(x, rois, img_metas=img_metas)
        else:
            bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if self.use_double_bbox_head and not self.training:
            dist_cls_score = bbox_results['dist_cls_score']
            dist_cls_score = dist_cls_score.split(num_proposals_per_img, 0)
        else:
            dist_cls_score = (None, ) * len(proposals)
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    proposal_obj_score[i],
                    dist_cls_score[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,
                    img_metas=img_metas,
                    bbox_save_path_root=self.bbox_save_path_root,
                    clip_infer_bbox = bbox_results['clip_infer_bbox'][0] if 'clip_infer_bbox' in bbox_results else None)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        if merged_bboxes.shape[0] == 0:
            # There is no proposal in the single image
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)
        else:
            det_bboxes, det_labels = multiclass_nms(merged_bboxes,
                                                    merged_scores,
                                                    rcnn_test_cfg.score_thr,
                                                    rcnn_test_cfg.nms,
                                                    rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin:

    if sys.version_info >= (3, 7):

        async def async_test_mask(self,
                                  x,
                                  img_metas,
                                  det_bboxes,
                                  det_labels,
                                  rescale=False,
                                  mask_test_cfg=None):
            """Asynchronized test for mask head without augmentation."""
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]['ori_shape']
            scale_factor = img_metas[0]['scale_factor']
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.num_classes)]
            else:
                if rescale and not isinstance(scale_factor,
                                              (float, torch.Tensor)):
                    scale_factor = det_bboxes.new_tensor(scale_factor)
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get('async_sleep_interval'):
                    sleep_interval = mask_test_cfg['async_sleep_interval']
                else:
                    sleep_interval = 0.035
                async with completed(
                        __name__,
                        'mask_head_forward',
                        sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg, ori_shape,
                    scale_factor, rescale)
            return segm_result

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        return segm_results

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip, flip_direction)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                # convert to numpy array to save memory
                aug_masks.append(
                    mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]['ori_shape']
            scale_factor = det_bboxes.new_ones(4)
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=scale_factor,
                rescale=False)
        return segm_result
