# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)

from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, mask, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

"""This is an unused script which load the pregenerate 
the CLIP text embedding for CLIP classiification"""

@HEADS.register_module()
class ClipClsHead(AnchorFreeHead):
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_reg_fcs=2,
                 word_embeddings_path=None,
                 linear_probe=True,
                 mlp_probe=False,
                 loss_cls=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        if loss_cls != None:
            class_weight = loss_cls.get('class_weight', None)
            if class_weight is not None and (self.__class__ is ClipClsHead):
                assert isinstance(class_weight, float), 'Expected ' \
                    'class_weight to have type float. Found ' \
                    f'{type(class_weight)}.'
                # NOTE following the official DETR rep0, bg_cls_weight means
                # relative classification weight of the no-object class.
                bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
                assert isinstance(bg_cls_weight, float), 'Expected ' \
                    'bg_cls_weight to have type float. Found ' \
                    f'{type(bg_cls_weight)}.'
                class_weight = torch.ones(num_classes + 1) * class_weight
                # set background class as the last indice
                class_weight[num_classes] = bg_cls_weight
                loss_cls.update({'class_weight': class_weight})
                if 'bg_cls_weight' in loss_cls:
                    loss_cls.pop('bg_cls_weight')
                self.bg_cls_weight = bg_cls_weight

        #self.num_query = num_query
        self.num_class = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.word_embeddings_path=word_embeddings_path
        self.linear_probe=linear_probe
        self.mlp_probe=mlp_probe

        if loss_cls != None:
            self.loss_cls = build_loss(loss_cls)
            if self.loss_cls.use_sigmoid:
                self.cls_out_channels = num_classes
            else:
                self.cls_out_channels = num_classes + 1
        else:
            self.loss_cls = None
        #self.activate = build_activation_layer(self.act_cfg)
        self._init_layers()

    def get_bboxes(self):
        pass
    def get_targets(self): 
        pass

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.linear_probe:
            self.fc_cls = Linear(self.in_channels, self.cls_out_channels)
        elif self.mlp_probe:
            self.act_cfg = dict(type='ReLU', inplace=True)
            self.activate = build_activation_layer(self.act_cfg)
            self.num_reg_fcs = 2
            self.cls_ffn = FFN(
                self.in_channels,
                self.in_channels*2,
                self.num_reg_fcs,
                self.act_cfg,
                dropout=0.0,
                add_residual=False)
            self.fc_cls = Linear(self.in_channels, self.cls_out_channels)
        else:
            self.fc_cls = None
        if self.word_embeddings_path != None:
            self.word_embeddings = torch.load(self.word_embeddings_path)
        else:
            self.word_embeddings = None

    def init_weights(self):
        # follow the official DETR to init parameters
        if self.linear_probe or self.mlp_probe:
            for m in self.fc_cls.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')
        if self.mlp_probe:
            for m in self.cls_ffn.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')            
        self._is_init = True

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list [Tensor]): len of list is batchsize. [gt_num_in_image, 512]
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores [gt_num_in_batch, cls_out_channels].
        """
        #feats = [feats]
        
        #feats = torch.cat(feats)
        all_cls_scores_list = []
        for feat in feats:
            if self.linear_probe:
                cls_scores = self.fc_cls(feat)
            elif self.mlp_probe:
                cls_scores = self.fc_cls(self.activate(
                            self.cls_ffn(feat)))
            else:
                cls_scores = (feat @ self.word_embeddings.T).softmax(dim=-1)
            all_cls_scores_list.append(cls_scores)
            
        return all_cls_scores_list

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             cls_scores_list,
             gt_labels_list,
             img_metas):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores(Tensor): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [gt_num_in_batch, cls_out_channels].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_pred = torch.cat(cls_scores_list)
        all_label = torch.cat(gt_labels_list)
        
        # classification loss
        loss_cls = self.loss_cls(all_pred, all_label)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = loss_cls
        return loss_dict

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #out: <class 'tuple'>
        #out = (all_cls_scores_list, all_bbox_preds_list)
        #all_cls_scores_list: 1
        #all_bbox_preds_list: 1
        #ele in all_cls_scores_list: torch.Size([64, 2, 1])
        #ele in all_bbox_preds_list: torch.Size([64, 2, 4])

        assert proposal_cfg is None, '"proposal_cfg" must be None'
        # the out here should be two lists, all_cls_scores_list and all_bbox_preds_list
        outs = self(x, img_metas)
        #if patches_gt is None:
        loss_inputs = (outs,) + (gt_labels, img_metas)
        losses = self.loss(*loss_inputs)
        return losses

    def simple_test_bboxes(self, feats, gt_labels, img_metas, gt_bboxes):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        #out list[tensor] tensor with shape [gt_per_img, channel]

        # calculate the acc
        # predict_results list(tensor) each tensor is a true false tensor shape [gt_per_img]
        predict_results = []
        for pred, gt_label, gt_bbox, img_meta in zip(outs, gt_labels, gt_bboxes, img_metas):
            pred_idx = torch.argmax(pred, dim=1)
            #result = (pred_idx == gt)
            #predict_results.append(result)
            # scale the gt bboxes back to the original size 
            scale_factor = img_meta['scale_factor']
            #print('before scale:', gt_bbox)
            #print('scale_factor:', scale_factor)
            gt_bbox /= gt_bbox.new_tensor(scale_factor)
            #print('after scale:', gt_bbox)
            # calculate the area
            area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])
            #print('area:', area.shape)
            size_result = torch.full(area.shape, -1)
            size_result[area > 96 ** 2] = 2
            size_result[(area < 96 ** 2) & (area > 32 **2)] = 1
            size_result[area < 32 **2] = 0
            #size_result.cuda()
            #print('size_result:', size_result.shape, size_result)
            
            # concat the gt and the pred result
            pred_and_gt = torch.cat([pred_idx.unsqueeze(dim=0).cuda(), gt_label.unsqueeze(dim=0).cuda(), size_result.unsqueeze(dim=0).cuda()], dim=0)
            predict_results.append(pred_and_gt)

        return predict_results

    def forward_onnx(self, feats, img_metas):
        """Forward function for exporting to ONNX.

        Over-write `forward` because: `masks` is directly created with
        zero (valid position tag) and has the same spatial size as `x`.
        Thus the construction of `masks` is different from that in `forward`.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        """"Forward function for a single feature level with ONNX exportation.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # Note `img_shape` is not dynamically traceable to ONNX,
        # since the related augmentation was done with numpy under
        # CPU. Thus `masks` is directly created with zeros (valid tag)
        # and the same spatial shape as `x`.
        # The difference between torch and exported ONNX model may be
        # ignored, since the same performance is achieved (e.g.
        # 40.1 vs 40.1 for DETR)
        batch_size = x.size(0)
        h, w = x.size()[-2:]
        masks = x.new_zeros((batch_size, h, w))  # [B,h,w]

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        #outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
        #                               pos_embed)
        # for encoder procedure
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        masks = masks.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        outs_dec = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=masks)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    def onnx_export(self, all_cls_scores_list, all_bbox_preds_list, img_metas):
        """Transform network outputs into bbox predictions, with ONNX
        exportation.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert len(img_metas) == 1, \
            'Only support one input image while in exporting to ONNX'

        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        # Note `img_shape` is not dynamically traceable to ONNX,
        # here `img_shape_for_onnx` (padded shape of image tensor)
        # is used.
        img_shape = img_metas[0]['img_shape_for_onnx']
        #max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        max_per_img = sum([ele * ele for ele in self.patches_list])
        batch_size = cls_scores.size(0)
        # `batch_index_offset` is used for the gather of concatenated tensor
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)

        # supports dynamical batch inference
        if self.loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(batch_size, -1).topk(
                max_per_img, dim=1)
            det_labels = indexes % self.num_class
            bbox_index = indexes // self.num_class
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
        else:
            scores, det_labels = F.softmax(
                cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img, dim=1)
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            det_labels = det_labels.view(-1)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            det_labels = det_labels.view(batch_size, -1)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
        # use `img_shape_tensor` for dynamically exporting to ONNX
        img_shape_tensor = img_shape.flip(0).repeat(2)  # [w,h,w,h]
        img_shape_tensor = img_shape_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, det_bboxes.size(1), 4)
        det_bboxes = det_bboxes * img_shape_tensor
        # dynamically clip bboxes
        x1, y1, x2, y2 = det_bboxes.split((1, 1, 1, 1), dim=-1)
        from mmdet.core.export import dynamic_clip_for_onnx
        x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, img_shape)
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

        return det_bboxes, det_labels
