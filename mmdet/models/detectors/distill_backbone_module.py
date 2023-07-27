# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class DistillBackboneModule(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone_to,
                 backbone_from,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DistillBackboneModule, self).__init__(init_cfg)
        #if pretrained:
        #    warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                  'please use "init_cfg" instead')
        #    backbone.pretrained = pretrained
        self.backbone_to = build_backbone(backbone_to)
        self.backbone_from = build_backbone(backbone_from)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def extract_to_feat(self, img):
        """Directly extract features from the backbone+neck."""
        #b_from_feat = self.backbone_from(img)
        b_to_feat = self.backbone_to(img)
        #if self.with_neck:
        #    x = self.neck(x)
        return b_to_feat

    def extract_from_feat(self, img):
        """Directly extract features from the backbone+neck."""
        #b_from_feat = self.backbone_from(img)
        b_from_feat = self.backbone_from(img)
        #if self.with_neck:
        #    x = self.neck(x)
        return b_from_feat

    def crop_img_to_patches(self, img, img_metas, w_patch_num_list, h_patch_num_list):
        bs, c, _, _ = img.shape
        #'img_shape':  torch.Size([2, 3, 800, 1184])
        # what we need is [800, 1184, 3]
        img = img.permute(0, 2, 3, 1)

        result = []
        for img_id in range(bs):
            for h_patch_num, w_patch_num in zip(h_patch_num_list, w_patch_num_list):
                H, W, channel = img_metas[img_id]['img_shape']
                patch_H, patch_W = H / h_patch_num, W / w_patch_num
                h_pos = [int(patch_H) * i for i in range(h_patch_num + 1)]
                w_pos = [int(patch_W) * i for i in range(w_patch_num + 1)]

                for i in range(h_patch_num):
                    h_start_pos = h_pos[i]
                    h_end_pos = h_pos[i+1]
                    for j in range(w_patch_num):
                        w_start_pos = w_pos[j]
                        w_end_pos = w_pos[j+1]
                        # cropping the img into the patches which size is (H/8) * (W/8)
                        # use the numpy to crop the image
                        # img shape: torch.Size([2, 3, 800, 1088])
                        now_patch = img[img_id, h_start_pos: h_end_pos, w_start_pos: w_end_pos, :]
                        PIL_image = Image.fromarray(np.uint8(now_patch))
                        # do the preprocessing
                        new_patch = self.preprocess(PIL_image)

                        #new_patch, w_scale, h_scale = mmcv.imresize(now_patch, (224, 224), return_scale=True)
                        #result.append(np.expand_dims(new_patch, axis=0))
                        result.append(new_patch.unsqueeze(dim=0))

        #cropped_patches = np.concatenate(result, axis=0)
        # the shape of the cropped_patches: torch.Size([bs*sum([ele**2 for ele in w_patch_num_list]), 3, 224, 224])
        cropped_patches = torch.cat(result, dim=0).cuda()
        return cropped_patches

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # random generate some bbox, to fill the gt list len = 100
        bs = len(gt_bboxes)
        for i in range(bs):
            img_w, img_h, c = img_metas['img_shape']
            gt_num = len()
        

        # crop the gt bboxes from the image and sent them it from backbone_from

        # sent the whole image to the backbone_to and obtain the whole feature map
        b_to_feat = self.extract_to_feat(img)

        # sent the features from the backbone_from and backbone_to 

        losses = dict()

        # RPN forward and loss
        #if self.with_rpn:
        #    proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                      self.test_cfg.rpn)
        #    rpn_losses, proposal_list = self.rpn_head.forward_train(
        #        x,
        #        img_metas,
        #        gt_bboxes,
        #        gt_labels=None,
        #        gt_bboxes_ignore=gt_bboxes_ignore,
        #        proposal_cfg=proposal_cfg,
        #        **kwargs)
        #    losses.update(rpn_losses)
        #else:
        #    proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
