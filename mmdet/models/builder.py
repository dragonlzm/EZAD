# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry, ConfigDict, print_log

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

# update for few-shot detection
def build_detector(cfg, train_cfg=None, test_cfg=None, logger: Optional[object] = None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    # get the prefix of fixed parameters
    frozen_parameters = cfg.pop('frozen_parameters', None)

    model = DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    model.init_weights()
    # freeze parameters by prefix
    if frozen_parameters is not None:
        print_log(f'Frozen parameters: {frozen_parameters}', logger)
        for name, param in model.named_parameters():
            for frozen_prefix in frozen_parameters:
                if frozen_prefix in name:
                    param.requires_grad = False
            if param.requires_grad:
                print_log(f'Training parameters: {name}', logger)
    return model
