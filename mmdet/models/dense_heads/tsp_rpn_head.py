# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
from .rpn_head import RPNHead

class MyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, kernel_size=3, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        if kernel_size == 3:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        elif kernel_size == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            raise NotImplementedError
        self.norm = norm
        if self.norm:
            SyncBN = nn.SyncBatchNorm
            self.bn = SyncBN(num_features=out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = F.relu(x, inplace=True)
        return x


@HEADS.register_module()
class TSPRPNHead(RPNHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 num_convs=2,
                 input_lvl=5,
                 **kwargs):
        self.num_convs = num_convs
        self.input_lvl = input_lvl
        #self.add_extra_bn = add_extra_bn
        #self.input_lvl = input_lvl
        super(TSPRPNHead, self).__init__(
            in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head.""" 
        # 3x3 conv for the hidden representation
        self.obj_conv = nn.ModuleList(
            [MyConvBlock(self.in_channels, self.feat_channels, norm=False, activation=False) for _ in range(self.num_convs)])
        self.anchor_conv = nn.ModuleList(
            [MyConvBlock(self.in_channels, self.feat_channels, norm=False, activation=False) for _ in range(self.num_convs)])

        SyncBN = nn.SyncBatchNorm

        self.obj_bn_list = nn.ModuleList(
            [nn.ModuleList([SyncBN(self.feat_channels) for i in range(self.num_convs)]) for j in range(self.input_lvl)])
        self.anchor_bn_list = nn.ModuleList(
            [nn.ModuleList([SyncBN(self.feat_channels) for i in range(self.num_convs)]) for j in range(self.input_lvl)])

        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(self.feat_channels, self.num_anchors * 4, kernel_size=1, stride=1)

        for l in [self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward_single(self, feats, obj_bn_list_per_lvl, anchor_bn_list_per_lvl):
        """Forward feature map of a single scale level."""
        return None
    
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        # 5 [torch.Size([2, 256, 272, 200]), torch.Size([2, 256, 136, 100]), torch.Size([2, 256, 68, 50]), torch.Size([2, 256, 34, 25]), torch.Size([2, 256, 17, 13])]
        # if hasattr(self, 'add_extra_bn') and self.add_extra_bn:
        #     return multi_apply(self.forward_single, feats, [i for i in range(len(feats))])
        # else:
        #     return multi_apply(self.forward_single, feats)
        pred_objectness_logits = []
        pred_anchor_deltas = []

        for x, obj_bn, anchor_bn in zip(feats, self.obj_bn_list, self.anchor_bn_list):
            t_obj = x
            for bn, conv in zip(obj_bn, self.obj_conv):
                t_obj = conv(t_obj)
                t_obj = bn(t_obj)
                t_obj = F.relu(t_obj, inplace=True)

            t_anchor = x
            for bn, conv in zip(anchor_bn, self.anchor_conv):
                t_anchor = conv(t_anchor)
                t_anchor = bn(t_anchor)
                t_anchor = F.relu(t_anchor, inplace=True)

            pred_objectness_logits.append(self.objectness_logits(t_obj))
            pred_anchor_deltas.append(self.anchor_deltas(t_anchor))
        return pred_objectness_logits, pred_anchor_deltas
