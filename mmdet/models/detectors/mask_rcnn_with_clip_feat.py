# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import numpy as np
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class MaskRCNNWithCLIPFeat(BaseDetector):
    """this is the class of hte Mask R-CNN trained with distillation for zero-shot object detection
    The feature is pregenerated"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 unknow_rpn_head=None,
                 roi_head=None,
                 rand_bboxes_num=20,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 vit_backbone_cfg=None):
        super(MaskRCNNWithCLIPFeat, self).__init__(init_cfg)
        #if pretrained:
        #    warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                  'please use "init_cfg" instead')
        #    backbone.pretrained = pretrained
        #self.backbone_to = build_backbone(backbone_to)
        if vit_backbone_cfg != None:
            self.vit_backbone_cfg = vit_backbone_cfg
            self.vit_backbone = build_backbone(vit_backbone_cfg)
            backbone['vit_backbone_cfg'] = vit_backbone_cfg
        
        self.backbone = build_backbone(backbone)
        
        if vit_backbone_cfg != None:
            self.vit_backbone.ln_post.weight.requires_grad = False
            self.vit_backbone.ln_post.bias.requires_grad = False
            max_merge_lvl = max([int(ele.strip('merge')) for ele in self.backbone.merge_step])
            min_unfreeze_transformer_lvl = (max_merge_lvl - 1) * self.vit_backbone_cfg.layers // 3
            
            # for layer >= min_unfreeze_transformer_lvl: required_grad = False
            for para_name, param in zip(self.vit_backbone.state_dict(), self.vit_backbone.parameters()):
                #print(para_name, self.state_dict()[para_name].shape, param.shape)
                if 'ln_' in para_name and not para_name.startswith('ln_'):
                    #transformer.resblocks.1.ln_1.bias
                    layer_idx = int(para_name.split('.')[2])
                    if layer_idx >= min_unfreeze_transformer_lvl:
                        param.requires_grad = False
                    #print(para_name, param.requires_grad)
                    
                    
            self.backbone.clip_visual_model = self.vit_backbone

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if unknow_rpn_head is not None:
            rpn_train_cfg = train_cfg.unknow_rpn if train_cfg is not None else None
            rpn_head_ = unknow_rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.unknow_rpn)
            self.unknow_rpn_head = build_head(rpn_head_)           
        
        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
        self.rand_bboxes_num = rand_bboxes_num
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.even_sample_uk = self.train_cfg.get('even_sample_uk', False) if self.train_cfg is not None else False
        self.test_head_name = self.test_cfg.get('test_head_name', 'both') if self.test_cfg is not None else 'both'
        self.fixed_param = self.train_cfg.get('fixed_param', False) if self.train_cfg is not None else False
        if self.fixed_param == True:
            self.fix_model_parameter()
        # check the model parameters in backbone
        # for para_name, param in zip(self.backbone.state_dict(), self.backbone.parameters()):
        #     print(para_name, param.requires_grad)
    
    def fix_model_parameter(self):
        # fix backbone 
        for para_name, param in zip(self.backbone.state_dict(), self.backbone.parameters()):
            param.requires_grad = False
            print('backbone', para_name)
        # fix neck
        for para_name, param in zip(self.neck.state_dict(), self.neck.parameters()):
            param.requires_grad = False
            print('neck', para_name)     
        # fix rpn_head
        for para_name, param in zip(self.rpn_head.state_dict(), self.rpn_head.parameters()):
            param.requires_grad = False
            print('rpn_head', para_name)
        print('backbone, neck, rpn_head parameters are fixed')


    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None
    
    @property
    def with_unknow_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'unknow_rpn_head') and self.unknow_rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img, ori_img=None):
        """Directly extract features from the backbone+neck."""
        # prepare the feat from the backbone_to
        if self.backbone.__class__.__name__.startswith("ResNetWithVit"):
            x = self.backbone(img, ori_img)
        else:
            x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        # prepare the feat from the backbone_from
        return x

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

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_feats=None,
                      rand_bboxes=None,
                      rand_feats=None,
                      rand_bbox_weights=None,
                      bg_bboxes=None,
                      bg_feats=None,
                      proposals=None,
                      ori_img=None,
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
        x = self.extract_feat(img, ori_img)
        
        # remove the padded 0 bboxes
        # gt_feats = [gt_feat[torch.abs(gt_feat).sum(dim=1) > 0] 
        #                 for gt_feat in gt_feats]
        
        rand_feats = [rand_feat[torch.abs(rand_feat).sum(dim=1) > 0] 
                for rand_feat in rand_feats]

        if rand_bbox_weights != None:
            rand_bbox_weights = [rand_bbox_weight[rand_bbox_weight > 0] 
                for rand_bbox_weight in rand_bbox_weights]
        
        # concat the feat of gt and random
        if self.roi_head.use_bg_pro_for_distill:
            distilled_feat = [torch.cat([gt_feat_per_img, rand_feat_per_img, bg_feat_per_img], dim=0)
                            for gt_feat_per_img, rand_feat_per_img, bg_feat_per_img in zip(gt_feats, rand_feats, bg_feats)] 
        elif self.roi_head.use_only_gt_pro_for_distill:
            distilled_feat = gt_feats
        elif self.roi_head.use_only_clip_prop_for_distill:
            distilled_feat = rand_feats
        else:
            distilled_feat = [torch.cat([gt_feat_per_img, rand_feat_per_img], dim=0)
                            for gt_feat_per_img, rand_feat_per_img in zip(gt_feats, rand_feats)]

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
            if self.rpn_head.__class__.__name__ == 'TriWayRPNHead':
                if not self.even_sample_uk:
                    trained_bbox = [torch.cat([gt_bbox, rand_bbox], dim=0).cuda() for gt_bbox, rand_bbox in zip(gt_bboxes, rand_bboxes)]
                    rpn_gt_labels = [torch.full((gt_bbox.shape[0],), 0) for gt_bbox in gt_bboxes] 
                    rpn_uk_labels = [torch.full((rand_bbox.shape[0],), 1) for rand_bbox in rand_bboxes]
                    trained_label = [torch.cat([rpn_gt_label, rpn_uk_label], dim=0).cuda() for rpn_gt_label, rpn_uk_label in zip(rpn_gt_labels, rpn_uk_labels)] 
                else:
                    # balance the samples
                    random_choices = [torch.from_numpy(np.random.choice(rand_bbox.shape[0], gt_bbox.shape[0], replace=False)) for gt_bbox, rand_bbox in zip(gt_bboxes, rand_bboxes)]
                    trained_bbox = [torch.cat([gt_bbox, rand_bbox[random_choice]], dim=0).cuda() for gt_bbox, rand_bbox, random_choice in zip(gt_bboxes, rand_bboxes, random_choices)]
                    rpn_gt_labels = [torch.full((gt_bbox.shape[0],), 0) for gt_bbox in gt_bboxes]
                    # sample the same number of random bboxes as the number of gt bboxes
                    rpn_uk_labels = [torch.full((gt_bbox.shape[0],), 1) for gt_bbox in gt_bboxes]
                    trained_label = [torch.cat([rpn_gt_label, rpn_uk_label], dim=0).cuda() for rpn_gt_label, rpn_uk_label in zip(rpn_gt_labels, rpn_uk_labels)] 
                
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    trained_bbox,
                    gt_labels=trained_label,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
            
        if self.with_unknow_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.unknow_rpn)
            use_clip_proposal_as_gt = self.train_cfg.unknow_rpn.get('use_clip_proposal_as_gt', True) if self.train_cfg is not None else True
            if use_clip_proposal_as_gt:
                target_gt = rand_bboxes
            else:
                target_gt = gt_bboxes
            temp_unknow_rpn_losses, unknow_proposal_list = self.unknow_rpn_head.forward_train(
                x,
                img_metas,
                target_gt,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg,
                **kwargs)
            # change the name of the rpn loss
            unknow_rpn_losses = dict(
                loss_uk_rpn_cls=temp_unknow_rpn_losses['loss_rpn_cls'], loss_uk_rpn_bbox=temp_unknow_rpn_losses['loss_rpn_bbox'])
            
            losses.update(unknow_rpn_losses)
            
            # combine the uk_proposal with the original_proposal
            proposal_list = [torch.cat([ori_proposal, uk_proposal], dim=0) for ori_proposal, uk_proposal in zip(proposal_list, unknow_proposal_list)]
                     

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore=gt_bboxes_ignore, 
                                                 gt_masks=gt_masks,
                                                 distilled_feat=distilled_feat, 
                                                 rand_bboxes=rand_bboxes,
                                                 rand_bbox_weights=rand_bbox_weights,
                                                 bg_bboxes=bg_bboxes,
                                                 bg_feats=bg_feats,
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

    def simple_test(self, img, img_metas, proposals=None, rescale=False, ori_img=None):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img, ori_img)
        if proposals is None:
            if self.with_unknow_rpn:
                if self.test_head_name == 'both':
                    proposal_list_1 = self.unknow_rpn_head.simple_test_rpn(x, img_metas)
                    proposal_list_2 = self.rpn_head.simple_test_rpn(x, img_metas)
                    proposal_list = [torch.cat([prop_1, prop_2], dim=0) for prop_1, prop_2 in zip(proposal_list_1, proposal_list_2)]
                elif self.test_head_name == 'extra':
                    proposal_list = self.unknow_rpn_head.simple_test_rpn(x, img_metas)
                else:
                    proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale, img=img)

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
