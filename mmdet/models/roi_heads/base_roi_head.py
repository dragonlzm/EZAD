# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule

from ..builder import build_shared_head


class BaseRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 extra_backbone=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_double_bbox_head=False,
                 init_cfg=None):
        super(BaseRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_double_bbox_head = use_double_bbox_head
        if shared_head is not None:
            if 'pretrained' not in shared_head:
                shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)
        if extra_backbone != None:
            self.init_extra_backbone(extra_backbone)

        self.init_assigner_sampler()
        self.use_bg_pro_for_distill = self.train_cfg.get('use_bg_pro_for_distill', False) if self.train_cfg is not None else False
        # whether to use clip bg proposal as negative sample
        self.use_bg_pro_as_ns = self.train_cfg.get('use_bg_pro_as_ns', False) if self.train_cfg is not None else False
        self.bg_pro_as_ns_weight = self.train_cfg.get('bg_pro_as_ns_weight', 1.0) if self.train_cfg is not None else 1.0
        # save the feat for classification
        self.save_the_feat = self.test_cfg.get('save_the_feat', None) if self.test_cfg is not None else None
        self.use_pregenerated_proposal = self.test_cfg.get('use_pregenerated_proposal', None) if self.test_cfg is not None else None
        # only using the gt bboxes for distillation
        self.use_only_gt_pro_for_distill = self.train_cfg.get('use_only_gt_pro_for_distill', False) if self.train_cfg is not None else False
                # only using the gt bboxes for distillation
        self.use_only_clip_prop_for_distill = self.train_cfg.get('use_only_clip_prop_for_distill', False) if self.train_cfg is not None else False
        # save the prediction
        self.bbox_save_path_root = self.test_cfg.get('bbox_save_path_root', None) if self.test_cfg is not None else None
        # add perturbation to the distillation
        self.add_distill_pertrub = self.train_cfg.get('add_distill_pertrub', False) if self.train_cfg is not None else False
        self.crop_loca_modi_ratio = self.train_cfg.get('crop_loca_modi_ratio', 0.5) if self.train_cfg is not None else 0.5
        self.crop_size_modi_ratio = self.train_cfg.get('crop_size_modi_ratio', 1.5) if self.train_cfg is not None else 1.5
        self.pertrub_ratio = self.train_cfg.get('pertrub_ratio', 0.5) if self.train_cfg is not None else 0.5
        self.gt_only_damp_factor = self.train_cfg.get('gt_only_damp_factor', False) if self.train_cfg is not None else False
        
    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False,
                                **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
