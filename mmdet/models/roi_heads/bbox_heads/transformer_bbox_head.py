# Copyright (c) OpenMMLab. All rights reserved.
from json import load
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
class TransformerBBoxHead(BBoxHead):
    r""" This is the detector head with the transformer module.
    The transformer module will refine the feature.
    The detector head will be use in training a Mask R-CNN with distillation.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared fcs ->  Transformer
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
                 reg_with_mlp=True,
                 encoder=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(TransformerBBoxHead, self).__init__(
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
        self.reg_with_mlp = reg_with_mlp
        self.encoder = encoder

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim
        
        # add tranformer
        self.encoder = build_transformer_layer_sequence(self.encoder)

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
            
            if self.reg_with_mlp:
                self.fc_reg = MLP(self.reg_last_dim, self.reg_last_dim, out_dim_reg, 3)
            else:
                self.fc_reg = build_linear_layer(
                    self.reg_predictor_cfg,
                    in_features=self.reg_last_dim,
                    out_features=out_dim_reg)


        # initial dec_pos_embed_proj
        self.post_input_proj_norm = nn.LayerNorm(self.fc_out_channels)   

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

    def init_weights(self):
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super(TransformerBBoxHead, self).init_weights()
        # initial the transformer
        for m in self.encoder.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

    def _forward(self, x, bboxes, img_meta):
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
        x = x.unsqueeze(dim=0)
        
        # normalize the proposal
        img_h, img_w, _ = img_meta['img_shape']
        # remove the first ele in each bbox(the idx for image)
        bboxes = bboxes[:, 1:]
        bboxes = bboxes / torch.tensor([img_w, img_h, img_w, img_h],
                                           dtype=torch.float32, device=bboxes.device)
        bboxes = torch.cat([box_xyxy_to_cxcywh(bboxes), bboxes], dim=-1)
        bboxes = bboxes.unsqueeze(dim=0)

        # prepare the positional encoding
        num_pos_feats = self.fc_out_channels // 8
        temperature = 10000.0
        
        # dim_t :torch.Size([num_pos_feats // 8])
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        #dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / num_pos_feats)

        # bboxes: [bs, h, w] => [bs, bbox_num, 8]
        # dim_t :torch.Size([num_pos_feats // 8])
        bbox_pos_embed = (bboxes[:, :, :, None] * 2 * math.pi) / dim_t
        # bbox_pos_embed:  [bs, h, w, num_pos_feats] => [bs, bbox_num, num_pos_feats]
        bbox_pos_embed = torch.stack((bbox_pos_embed[:, :, 0::2].sin(),
                                        bbox_pos_embed[:, :, 1::2].cos()), dim=4).flatten(2)
        
        # transformer
        x = self.post_input_proj_norm(x)
        
        #x = self.transformer_encoder(x, src_key_padding_mask=dec_mask, pos=bbox_pos_embed)
        # pos_embed: [h*w, bs, c] => [bbox_num, bs, num_pos_feats]
        # x: [h*w, bs, c] => [bbox_num, bs, num_pos_feats]
        x = x.permute(1,0,2)
        bbox_pos_embed = bbox_pos_embed.permute(1,0,2)
        #print('bbox_pos_embed', bbox_pos_embed.shape, bbox_pos_embed)
        #print('bboxes', bboxes.shape)
        x = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=bbox_pos_embed)
        #print('x', x.shape)
        #x = x.transpose(0, 1).contiguous().view(batch_size * seq_length, hidden_size)
        
        # separate branches
        # x: [h*w, bs, c] => [bbox_num, bs, num_pos_feats]
        x = x.squeeze(dim=1)
        #print('x', x.shape)
        # x: [bbox_num, num_pos_feats]
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
        return cls_score, bbox_pred, x_cls

    def forward(self, bbox_feats, proposals, img_metas, bboxes_num=None):
        """
            The input should like this:
            bbox_feats: shape (n, 256, 7, 7), n is the total number of the proposal in the batch
            proposals: shape (n, 5), [batch_ind, x1, y1, x2, y2], with xyxy format, n is the total number of the proposal in the batch
            img_metas: list[dict], len == batch_size
            gt_rand_rois: shape (k, 5), k is the total number of distillation bboxes in the batch,
                        the order of the feat would like this [gt_bbox_for_img1, rand_bbox_for_img1, gt_bbox_for_img2, rand_bbox_for_img2, ····]
            gt_and_rand_bbox_feat: shape (k, 256, 7, 7), k is the total number of distillation bboxes in the batch,
                        the order of the feat would like this [gt_bbox_for_img1, rand_bbox_for_img1, gt_bbox_for_img2, rand_bbox_for_img2, ····]
            bboxes_num: list[tuple(gt_bbox_num, rand_bbox_num, proposal_number)] or list[tuple(proposal_number, )]
        
            The output of this function should following the format like this:
            cls_score: torch.Size([1024, 49]) 
            bbox_pred: torch.Size([1024, 192]) 
            gt_and_bg_feats: torch.Size([1024, 512])
        """
        all_feats_per_image = []
        all_boxes_per_image = []
        proposal_feat_start_idx = 0

        assert isinstance(bboxes_num[0], int)
        for proposal_number in bboxes_num:
            now_proposal = proposals[proposal_feat_start_idx: proposal_feat_start_idx + proposal_number]
            now_proposal_feat = bbox_feats[proposal_feat_start_idx: proposal_feat_start_idx + proposal_number]
            
            proposal_feat_start_idx = proposal_feat_start_idx + proposal_number
            all_boxes_per_image.append(now_proposal)
            all_feats_per_image.append(now_proposal_feat)

        # forward for each image
        all_cls_score_per_image = []
        all_bbox_pred_per_image = []
        all_x_cls_per_image = []
        for feat_per_image, boxes_per_image, img_meta in zip(all_feats_per_image, all_boxes_per_image, img_metas):
            cls_score_per_image, bbox_pred_per_image, x_cls_per_image = self._forward(feat_per_image, boxes_per_image, img_meta)
            all_cls_score_per_image.append(cls_score_per_image)
            all_bbox_pred_per_image.append(bbox_pred_per_image)
            all_x_cls_per_image.append(x_cls_per_image)
            
        # concat the result
        all_cls_score_per_image = torch.cat(all_cls_score_per_image, dim=0)
        all_bbox_pred_per_image = torch.cat(all_bbox_pred_per_image, dim=0)
        all_x_cls_per_image = torch.cat(all_x_cls_per_image, dim=0)
        
        return all_cls_score_per_image, all_bbox_pred_per_image

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


@HEADS.register_module()
class TransformerEmbeddingBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared fcs ->  Transformer
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
                 reg_with_mlp=True,
                 use_bg_vector=True,
                 filter_base_cate=None,
                 encoder=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(TransformerEmbeddingBBoxHead, self).__init__(
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
        self._temperature = temperature
        self.filter_base_cate = filter_base_cate
        self.use_bg_vector = use_bg_vector
        self.reg_with_mlp = reg_with_mlp
        self.encoder = encoder

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim
        
        # add tranformer
        self.encoder = build_transformer_layer_sequence(self.encoder)

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
            self.map_to_clip = build_linear_layer(self.cls_predictor_cfg,
                                            in_features=self.fc_out_channels,
                                            out_features=self.clip_dim)
            
            if self.use_bg_vector:
                self.fc_cls_bg = build_linear_layer(self.cls_predictor_cfg,
                                                in_features=self.clip_dim,
                                                out_features=1,
                                                bias=False)
            self.fc_cls = None
            
            self.fc_cls_fg = build_linear_layer(self.cls_predictor_cfg,
                                            in_features=self.clip_dim,
                                            out_features=self.num_classes,
                                            bias=False)
            
            load_value = torch.load(self.fg_vec_cfg.load_path)
            load_value = load_value / load_value.norm(dim=-1, keepdim=True)
            #load_value = load_value.t()
            self.load_value = load_value.cuda()
            
            # for testing
            if self.filter_base_cate != None:
                #self.filter_base_cate = 'data/embeddings/base_finetuned_48cates.pt'
                base_load_value = torch.load(self.filter_base_cate)
                base_load_value = base_load_value / base_load_value.norm(dim=-1, keepdim=True)
                #load_value = load_value.t()
                self.base_load_value = base_load_value.cuda()
                
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
                self.fc_reg = MLP(final_reg_in_dim, final_reg_in_dim, final_reg_out_dim, 3)
            else:
                self.fc_reg = build_linear_layer(
                    self.reg_predictor_cfg,
                    in_features=final_reg_in_dim,
                    out_features=final_reg_out_dim)
        
        # initial dec_pos_embed_proj
        self.post_input_proj_norm = nn.LayerNorm(self.fc_out_channels)   
        
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

    def init_weights(self):
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super(TransformerEmbeddingBBoxHead, self).init_weights()
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
        
        # initial the transformer
        for m in self.encoder.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')  

    def _forward(self, x, bboxes, img_meta):
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
        x = x.unsqueeze(dim=0)
        
        # normalize the proposal
        img_h, img_w, _ = img_meta['img_shape']
        # remove the first ele in each bbox(the idx for image)
        bboxes = bboxes[:, 1:]
        bboxes = bboxes / torch.tensor([img_w, img_h, img_w, img_h],
                                           dtype=torch.float32, device=bboxes.device)
        bboxes = torch.cat([box_xyxy_to_cxcywh(bboxes), bboxes], dim=-1)
        bboxes = bboxes.unsqueeze(dim=0)

        # prepare the positional encoding
        num_pos_feats = self.fc_out_channels // 8
        temperature = 10000.0
        
        # dim_t :torch.Size([num_pos_feats // 8])
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        #dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / num_pos_feats)

        # bboxes: [bs, h, w] => [bs, bbox_num, 8]
        # dim_t :torch.Size([num_pos_feats // 8])
        bbox_pos_embed = (bboxes[:, :, :, None] * 2 * math.pi) / dim_t
        # bbox_pos_embed:  [bs, h, w, num_pos_feats] => [bs, bbox_num, num_pos_feats]
        bbox_pos_embed = torch.stack((bbox_pos_embed[:, :, 0::2].sin(),
                                        bbox_pos_embed[:, :, 1::2].cos()), dim=4).flatten(2)
        
        # transformer
        x = self.post_input_proj_norm(x)
        
        #x = self.transformer_encoder(x, src_key_padding_mask=dec_mask, pos=bbox_pos_embed)
        # pos_embed: [h*w, bs, c] => [bbox_num, bs, num_pos_feats]
        # x: [h*w, bs, c] => [bbox_num, bs, num_pos_feats]
        x = x.permute(1,0,2)
        bbox_pos_embed = bbox_pos_embed.permute(1,0,2)
        #print('bbox_pos_embed', bbox_pos_embed.shape, bbox_pos_embed)
        #print('bboxes', bboxes.shape)
        x = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=bbox_pos_embed)
        #print('x', x.shape)
        #x = x.transpose(0, 1).contiguous().view(batch_size * seq_length, hidden_size)
        
        # separate branches
        # x: [h*w, bs, c] => [bbox_num, bs, num_pos_feats]
        x = x.squeeze(dim=1)
        #print('x', x.shape)
        # x: [bbox_num, num_pos_feats]
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
        # normalize the image feat
        x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        #logit_scale = self.logit_scale.exp()
        fg_score = self.fc_cls_fg(x_cls)
        if self.use_bg_vector:
            bg_score = self.fc_cls_bg(x_cls)
            cls_score = torch.cat([fg_score, bg_score], dim=-1)
        else:
            cls_score = fg_score
        
        # for testing
        if self.filter_base_cate != None:
            base_score = self.fc_cls_base(x_cls)
            cls_score = torch.cat([cls_score, base_score], dim=-1)
        
        cls_score *= self._temperature
        
        if self.reg_with_cls_embedding:
            #print('original shape', x_reg.shape)
            x_reg = x_reg.unsqueeze(dim=-2)
            x_reg = x_reg.repeat(1, self.num_classes, 1)
            prepared_class_embedding = self.load_value.unsqueeze(dim=0).repeat(x_reg.shape[0],1,1)
            if self.combine_reg_and_cls_embedding == 'cat':
                # concat the word embedding
                final_x_reg = torch.cat([x_reg, prepared_class_embedding],dim=-1)
            else:
                x_reg = self.reg_map_to_clip(x_reg)
                final_x_reg = x_reg + prepared_class_embedding
            #print('final_x_reg', final_x_reg.shape)
            bbox_pred = self.fc_reg(final_x_reg)
            bbox_pred = bbox_pred.view(x_reg.shape[0], -1)
            #print('final_prediction', bbox_pred.shape)
        else:
            bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, x_cls

    def forward(self, img_metas, bbox_feats=None, proposals=None,
                gt_rand_rois=None, gt_and_rand_bbox_feat=None, bboxes_num=None):
        """
            The input should like this:
            bbox_feats: shape (n, 256, 7, 7), n is the total number of the proposal in the batch
            proposals: shape (n, 5), [batch_ind, x1, y1, x2, y2], with xyxy format, n is the total number of the proposal in the batch
            img_metas: list[dict], len == batch_size
            gt_rand_rois: shape (k, 5), k is the total number of distillation bboxes in the batch,
                        the order of the feat would like this [gt_bbox_for_img1, rand_bbox_for_img1, gt_bbox_for_img2, rand_bbox_for_img2, ····]
            gt_and_rand_bbox_feat: shape (k, 256, 7, 7), k is the total number of distillation bboxes in the batch,
                        the order of the feat would like this [gt_bbox_for_img1, rand_bbox_for_img1, gt_bbox_for_img2, rand_bbox_for_img2, ····]
            bboxes_num: list[tuple(gt_bbox_num, rand_bbox_num, proposal_number)] or list[tuple(proposal_number, )]
        
            The output of this function should following the format like this:
            cls_score: torch.Size([1024, 49]) 
            bbox_pred: torch.Size([1024, 192]) 
            gt_and_bg_feats: torch.Size([1024, 512])
        """
        
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
        
        # if in the training: 
        # 1. forward with both proposal and distillation bboxes: 
        # i.concat the bbox_feats and gt_and_rand_bbox_feat,
        # ii. aggregate the feature base on the image. iii. aggregate the bbox base on the image
        # the order of the feat should be the same as the order of the bbox
        all_feats_per_image = []
        all_boxes_per_image = []
        proposal_feat_start_idx = 0
        distill_bbox_feat_start_idx = 0
        if gt_and_rand_bbox_feat is not None:
            assert len(bboxes_num[0]) == 3
            # split the feat base on the image
            for gt_bbox_num, rand_bbox_num, proposal_number in bboxes_num:
                now_proposal = proposals[proposal_feat_start_idx: proposal_feat_start_idx + proposal_number]
                now_distill_bbox = gt_rand_rois[distill_bbox_feat_start_idx: distill_bbox_feat_start_idx + gt_bbox_num + rand_bbox_num]
                now_proposal_feat = bbox_feats[proposal_feat_start_idx: proposal_feat_start_idx + proposal_number]
                now_distill_bbox_feat = gt_and_rand_bbox_feat[distill_bbox_feat_start_idx: distill_bbox_feat_start_idx + gt_bbox_num + rand_bbox_num]
                
                proposal_feat_start_idx = proposal_feat_start_idx + proposal_number
                distill_bbox_feat_start_idx = distill_bbox_feat_start_idx + gt_bbox_num + rand_bbox_num
                # concat the proposal and the distillation bboxes
                bbox_for_now_image = torch.cat([now_proposal, now_distill_bbox], dim=0)
                feat_for_now_image = torch.cat([now_proposal_feat, now_distill_bbox_feat], dim=0)
                all_boxes_per_image.append(bbox_for_now_image)
                all_feats_per_image.append(feat_for_now_image)
        # if 1.in the testing / in the training (forward with the proposal only)
        # 2. forward with distillation bboxes only:
        else:
            assert isinstance(bboxes_num[0], int)
            for proposal_number in bboxes_num:
                now_proposal = proposals[proposal_feat_start_idx: proposal_feat_start_idx + proposal_number]
                now_proposal_feat = bbox_feats[proposal_feat_start_idx: proposal_feat_start_idx + proposal_number]
                
                proposal_feat_start_idx = proposal_feat_start_idx + proposal_number
                all_boxes_per_image.append(now_proposal)
                all_feats_per_image.append(now_proposal_feat)
            
        # forward for each image
        all_cls_score_per_image = []
        all_bbox_pred_per_image = []
        all_x_cls_per_image = []
        for feat_per_image, boxes_per_image, img_meta in zip(all_feats_per_image, all_boxes_per_image, img_metas):
            cls_score_per_image, bbox_pred_per_image, x_cls_per_image = self._forward(feat_per_image, boxes_per_image, img_meta)
            all_cls_score_per_image.append(cls_score_per_image)
            all_bbox_pred_per_image.append(bbox_pred_per_image)
            all_x_cls_per_image.append(x_cls_per_image)
            
        # concat the result
        all_cls_score_per_image = torch.cat(all_cls_score_per_image, dim=0)
        all_bbox_pred_per_image = torch.cat(all_bbox_pred_per_image, dim=0)
        all_x_cls_per_image = torch.cat(all_x_cls_per_image, dim=0)
        
        return all_cls_score_per_image, all_bbox_pred_per_image, all_x_cls_per_image

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
