# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import bbox_mapping
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from PIL import Image
import numpy as np
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


@DETECTORS.register_module()
class NEWRPN(BaseDetector):
    """Implementation of new Region Proposal Network."""

    def __init__(self,
                 patches_list,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None):
        super(NEWRPN, self).__init__(init_cfg)
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
        self.patches_list = patches_list

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

    def extract_feat(self, img, img_metas=None):
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
        bs = img.shape[0]
        
        # crop the img into the patches with normalization and reshape
        # (a function to convert the img)
        converted_img_patches = self.crop_img_to_patches(img.cpu(), img_metas, self.patches_list, self.patches_list)

        # convert dimension from [bs, 64, 224, 224] to [bs*64, 224, 224]
        #converted_img_patches = converted_img_patches.view(bs, -1, self.backbone.input_resolution, self.backbone.input_resolution)

        # the input of the vision transformer should be torch.Size([64, 3, 224, 224])
        x = self.backbone(converted_img_patches)
        if self.with_neck:
            x = self.neck(x)
        # convert the feature [bs*64, 512] to [bs, 64, 512]
        x = x.view(bs, -1, x.shape[-1])
        return x

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
                      gt_labels=None,
                      patches_gt=None):
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

        x = self.extract_feat(img, img_metas)
        losses = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels,
                                             gt_bboxes_ignore, patches_gt=patches_gt)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        x = self.extract_feat(img, img_metas)
        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        proposal_list = self.rpn_head.simple_test_bboxes(x, img_metas, rescale)
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
