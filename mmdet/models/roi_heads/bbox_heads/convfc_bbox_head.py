# Copyright (c) OpenMMLab. All rights reserved.
from json import load
from typing import final
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
import math


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        nn.init.normal_(self.layers[-1].weight, std=0.001)
        for l in [self.layers[-1]]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class ConvFCEmbeddingBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.
    This is the bbox head for zero-shot detection. It replace the learnable classification
    layer with the text embeddings from the CLIP

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 fg_vec_cfg=None,
                 clip_dim=512,
                 temperature=100,
                 reg_with_mlp=False,
                 use_bg_vector=True,
                 use_zero_bg_vector=False,
                 filter_base_cate=None,
                 use_svd_conversion=None,
                 mapping_after_dist=None,
                 normalize_bg_vec=False,
                 use_pregenerate_proposal_and_score=None,
                 learnable_temperature=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCEmbeddingBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fg_vec_cfg = fg_vec_cfg
        self.clip_dim = clip_dim
        self.filter_base_cate = filter_base_cate
        self.use_bg_vector = use_bg_vector
        self.use_zero_bg_vector = use_zero_bg_vector
        self.reg_with_mlp = reg_with_mlp
        self.use_svd_conversion = use_svd_conversion
        self.mapping_after_dist = mapping_after_dist
        self.normalize_bg_vec = normalize_bg_vec
        self.use_pregenerate_proposal_and_score = use_pregenerate_proposal_and_score
        self.learnable_temperature = learnable_temperature
        
        if self.learnable_temperature:
            self._temperature = nn.Parameter(torch.ones([]) * 4.6052)
        else:
            self._temperature = temperature

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        if self.use_svd_conversion != None:
            conversion_mat = torch.load(self.use_svd_conversion)
            #base_load_value = base_load_value / base_load_value.norm(dim=-1, keepdim=True)
            self.conversion_mat = conversion_mat.cuda()
            self.svd_conversion_mat = build_linear_layer(self.cls_predictor_cfg,
                                in_features=self.clip_dim,
                                out_features=self.clip_dim,
                                bias=False)
        
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.map_to_clip = build_linear_layer(self.cls_predictor_cfg,
                                            in_features=self.fc_out_channels,
                                            out_features=self.clip_dim)
            if self.use_bg_vector:
                if self.custom_cls_channels:
                    cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
                    bg_out_features = cls_channels - self.num_classes
                else:
                    bg_out_features = 1
                
                if self.normalize_bg_vec:
                    bg_predictor_cfg = dict(type='NormedLinear', tempearture=1)
                else:
                    bg_predictor_cfg = self.cls_predictor_cfg    
                    
                self.fc_cls_bg = build_linear_layer(bg_predictor_cfg,
                                                in_features=self.clip_dim,
                                                out_features=bg_out_features,
                                                bias=False)
            self.fc_cls = None
            self.fc_cls_fg = build_linear_layer(self.cls_predictor_cfg,
                                            in_features=self.clip_dim,
                                            out_features=self.num_classes,
                                            bias=False)
            
            load_value = torch.load(self.fg_vec_cfg.load_path)
            load_value = load_value.cuda()
            if self.use_svd_conversion:
                # convert the text feature
                load_value = torch.mm(load_value, self.conversion_mat)
                #print('load_value:', load_value)
            
            load_value = load_value / load_value.norm(dim=-1, keepdim=True)
            #load_value = load_value.t()
            self.load_value = load_value
            
            # for testing
            if self.filter_base_cate != None:
                #self.filter_base_cate = 'data/embeddings/base_finetuned_48cates.pt'
                base_load_value = torch.load(self.filter_base_cate)
                base_load_value = base_load_value.cuda()
                if self.use_svd_conversion:
                    # convert the text feature
                    base_load_value = torch.mm(base_load_value, self.conversion_mat)
                    #print('base_load_value:', base_load_value)
                
                base_load_value = base_load_value / base_load_value.norm(dim=-1, keepdim=True)
                #load_value = load_value.t()
                self.base_load_value = base_load_value
                
                self.fc_cls_base = build_linear_layer(self.cls_predictor_cfg,
                                                in_features=self.clip_dim,
                                                out_features=self.base_load_value.shape[0],
                                                bias=False)
                
        if self.with_reg:
            if self.reg_with_cls_embedding:
                if self.combine_reg_and_cls_embedding == 'add':
                    self.reg_map_to_clip = build_linear_layer(
                        self.reg_predictor_cfg,
                        in_features=self.reg_last_dim,
                        out_features=self.clip_dim)
                    final_reg_in_dim = self.reg_last_dim
                    final_reg_out_dim = 4     
                else:
                    final_reg_in_dim = self.reg_last_dim + self.clip_dim
                    final_reg_out_dim = 4
            else:
                final_reg_in_dim = self.reg_last_dim
                final_reg_out_dim = (4 if self.reg_class_agnostic else 4 *
                            self.num_classes)
            
            if self.reg_with_mlp:
                # self.fc_reg = nn.Sequential(OrderedDict([
                #     ("c_fc", nn.Linear(final_reg_in_dim, 1024)),
                #     ("relu", nn.ReLU(inplace=True)),
                #     #("dropout", nn.Dropout(0.1)),
                #     ("c_proj", nn.Linear(1024, final_reg_out_dim))
                # ]))
                #print('in self.reg_with_mlp')
                self.fc_reg = MLP(final_reg_in_dim, final_reg_in_dim, final_reg_out_dim, 3)
            else:
                self.fc_reg = build_linear_layer(
                    self.reg_predictor_cfg,
                    in_features=final_reg_in_dim,
                    out_features=final_reg_out_dim)
            
        if self.mapping_after_dist == 'linear':
            self.mapping_after_dist = build_linear_layer(self.cls_predictor_cfg,
                                            in_features=self.clip_dim,
                                            out_features=self.clip_dim)
        elif self.mapping_after_dist == 'mlp':
            self.mapping_after_dist = MLP(input_dim=self.clip_dim, hidden_dim=2*self.clip_dim, output_dim=self.clip_dim, num_layers=2)
        else:
            self.mapping_after_dist = None

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs'),
                        dict(name='map_to_clip')
                    ])
            ]

    def init_weights(self):
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super(ConvFCEmbeddingBBoxHead, self).init_weights()
        # load the conversion mat first if it exist
        if self.use_svd_conversion != None:
            with torch.no_grad():
                self.svd_conversion_mat.weight.copy_(self.conversion_mat)
            for param in self.svd_conversion_mat.parameters():
                param.requires_grad = False
        
        # load the module and set the require_grad
        with torch.no_grad():
            self.fc_cls_fg.weight.copy_(self.load_value)
        for param in self.fc_cls_fg.parameters():
            param.requires_grad = False
        self.load_value.require_grad = False
        #self.fc_cls_fg.weight.require_grad = False
        
        if self.filter_base_cate != None:
            with torch.no_grad():
                self.fc_cls_base.weight.copy_(self.base_load_value)
            for param in self.fc_cls_base.parameters():
                param.requires_grad = False 
                
        # handle the zero bg
        if self.use_zero_bg_vector:
            with torch.no_grad():
                nn.init.constant_(self.fc_cls_bg.weight, 0)  # zero embeddings
            for param in self.fc_cls_bg.parameters():
                param.requires_grad = False 

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x, proposal_assigned_gt_labels=None):
        # load the pretrained text embedding again
        if False in (self.fc_cls_fg.weight.data == self.load_value):
            print('loading value again')
            with torch.no_grad():
                self.fc_cls_fg.weight.copy_(self.load_value)
            for param in self.fc_cls_fg.parameters():
                param.requires_grad = False
                
            # for testing
            if self.filter_base_cate != None:
                print('base_load_value is loaded')
                with torch.no_grad():
                    self.fc_cls_base.weight.copy_(self.base_load_value)
                for param in self.fc_cls_base.parameters():
                    param.requires_grad = False
            
            # load the conversion mat again
            if self.use_svd_conversion != None:
                print('svd_conversion_mat is loaded')
                with torch.no_grad():
                    self.svd_conversion_mat.weight.copy_(self.conversion_mat)
                for param in self.svd_conversion_mat.parameters():
                    param.requires_grad = False
        
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        # map the vector from 1024 to clip dim
        x_cls = self.map_to_clip(x_cls)
        
        # add addtional mapping to seperate the vector for distillation 
        if self.mapping_after_dist != None:
            # the x_cls is unnormalized
            final_x_cls = self.mapping_after_dist(x_cls)
            # normalize the image feat for returning and distillation
            x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
            # normalize the image feat for classification
            final_x_cls = final_x_cls / final_x_cls.norm(dim=-1, keepdim=True)
        else:
            # normalize the image feat
            x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
            final_x_cls = x_cls
        
        # x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
        # if self.mapping_after_dist != None:
        #     # the x_cls is unnormalized
        #     final_x_cls = self.mapping_after_dist(x_cls)
        # else:
        #     final_x_cls = x_cls
                
        
        # cosine similarity as logits
        #logit_scale = self.logit_scale.exp()
        fg_score = self.fc_cls_fg(final_x_cls)
        if self.use_bg_vector:
            bg_score = self.fc_cls_bg(final_x_cls)
            cls_score = torch.cat([fg_score, bg_score], dim=-1)
        else:
            cls_score = fg_score
        
        # for testing
        if self.filter_base_cate != None:
            base_score = self.fc_cls_base(final_x_cls)
            cls_score = torch.cat([cls_score, base_score], dim=-1)
        
        # handle the temperature
        if self.learnable_temperature:
            now_temperature = self._temperature.exp()
        else:
            now_temperature = self._temperature
        
        # for see-saw loss
        if self.custom_cls_channels:
            #scores = self.loss_cls.get_activation(cls_score)
            cls_score[..., :self.num_classes] *= now_temperature
            cls_score[..., self.num_classes+2:] *= now_temperature
        else:
            cls_score *= now_temperature
        
        if self.reg_with_cls_embedding:
            #print('original shape', x_reg.shape)
            # x_reg = x_reg.unsqueeze(dim=-2)
            # x_reg = x_reg.repeat(1, self.num_classes, 1)
            # prepared_class_embedding = self.load_value.unsqueeze(dim=0).repeat(x_reg.shape[0],1,1)
            # if self.combine_reg_and_cls_embedding == 'cat':
            #     # concat the word embedding
            #     final_x_reg = torch.cat([x_reg, prepared_class_embedding],dim=-1)
            # else:
            #     x_reg = self.reg_map_to_clip(x_reg)
            #     final_x_reg = x_reg + prepared_class_embedding
            
            # new version of implementation
            # means it's training time
            empty_bg_vec = torch.zeros(1, self.load_value.shape[-1]).cuda()
            prepared_class_embedding = torch.cat([self.load_value, empty_bg_vec], dim=0)
            if proposal_assigned_gt_labels is not None:
                #print('torch.max(proposal_assigned_gt_labels)', torch.max(proposal_assigned_gt_labels), 'prepared_class_embedding.shape', prepared_class_embedding.shape)
                selected_embedding = prepared_class_embedding[proposal_assigned_gt_labels]
                #selected_embedding = torch.rand([x_reg.shape[0], 512]).cuda()
                final_x_reg = torch.cat([x_reg, selected_embedding],dim=-1)
            # means it's testing
            else:
                if self.custom_cls_channels:
                    scores = self.loss_cls.get_activation(cls_score)
                else:
                    scores = F.softmax(cls_score[:, :self.num_classes+1], dim=-1) if cls_score is not None else None
                
                # for the filter_base_cate and see-saw loss
                predicted_label = torch.argmax(scores, dim=-1)
                selected_embedding = prepared_class_embedding[predicted_label]
                final_x_reg = torch.cat([x_reg, selected_embedding],dim=-1)
                
            #print('final_x_reg', final_x_reg.shape)
            bbox_pred = self.fc_reg(final_x_reg)
            # in the original version the bbox_pred [1000, 1203, 4]
            # new version should be [1000, 4]
            bbox_pred = bbox_pred.view(x_reg.shape[0], -1)
            #print('final_prediction', bbox_pred.shape)
        else:
            bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, x_cls