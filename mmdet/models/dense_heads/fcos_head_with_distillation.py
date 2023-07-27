# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmdet.models.utils import build_linear_layer
from ..utils import build_aggregator
import copy

INF = 1e8


@HEADS.register_module()
class FCOSHeadWithDistillation(AnchorFreeHead):
    """This is the detector head for zero-shot detection.
    It implement the distillation module in the head. 
    The distillation will be conducted in the training.
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     ### for mapping ###
                     override=dict(
                         type='Normal',
                         name='map_to_clip',
                         std=0.01,
                         bias_prob=0.01)),
                 clip_dim=512,
                 fg_vec_cfg=None,
                 temperature=1,
                 use_centerness=False,
                 induced_centerness=True,
                 conv_based_mapping=True,
                 distill_negative=False,
                 use_cross_correlation=False,
                 correlation_linear_layer=False,
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.cls_predictor_cfg=dict(type='Linear')
        self.clip_dim=clip_dim
        self.fg_vec_cfg=fg_vec_cfg
        load_value = torch.load(self.fg_vec_cfg.load_path)
        load_value = load_value / load_value.norm(dim=-1, keepdim=True)
        self.load_value = load_value.cuda()
        self.use_centerness = use_centerness
        self.induced_centerness = induced_centerness
        self.conv_based_mapping = conv_based_mapping
        self.distill_negative = distill_negative
        self.use_cross_correlation = use_cross_correlation
        self.correlation_linear_layer = correlation_linear_layer
        self.aggregation_layer = dict(
                     type='AggregationLayer',
                     aggregator_cfgs=[
                         dict(
                             type='DepthWiseCorrelationAggregator',
                             in_channels=self.clip_dim,
                             with_fc=self.correlation_linear_layer,
                             out_channels=1 if self.correlation_linear_layer else None)
                     ])
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        #self.loss_centerness = build_loss(loss_centerness)
        self.conv_cls = None
        self.distill_loss_factor = self.train_cfg.get('distill_loss_factor', 1.0) if self.train_cfg is not None else 1.0   
        self.distillation_loss_config = dict(type='L1Loss', loss_weight=1.0)
        self.distillation_loss = build_loss(self.distillation_loss_config)
        self._temperature = temperature
        if self.use_centerness and self.induced_centerness:
            self.loss_centerness = build_loss(loss_centerness)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        ### for mapping ###
        if self.conv_based_mapping:
            self.map_to_clip = nn.Conv2d(
                self.feat_channels, self.clip_dim, 3, padding=1)
        else:
            self.map_to_clip = build_linear_layer(self.cls_predictor_cfg,
                                    in_features=self.feat_channels,
                                    out_features=self.clip_dim)
        if self.use_cross_correlation:
            self.fc_cls = build_aggregator(copy.deepcopy(self.aggregation_layer))
        else:
            self.fc_cls = build_linear_layer(self.cls_predictor_cfg,
                                    in_features=self.clip_dim,
                                    out_features=self.num_classes,
                                    bias=False)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        if self.filter_base_cate:
            base_load_value = torch.load(self.filter_base_cate)
            base_load_value = base_load_value / base_load_value.norm(dim=-1, keepdim=True)
            #load_value = load_value.t()
            self.base_load_value = base_load_value.cuda()
            
            self.fc_cls_base = build_linear_layer(self.cls_predictor_cfg,
                                            in_features=self.clip_dim,
                                            out_features=self.base_load_value.shape[0],
                                            bias=False)            

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        #self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.use_cross_correlation:
            self.support_feat = self.load_value.unsqueeze(dim=-1).unsqueeze(dim=-1)
            self.support_feat.require_grad = False
        else:
            with torch.no_grad():
                self.fc_cls.weight.copy_(self.load_value)
            for param in self.fc_cls.parameters():
                param.requires_grad = False
        if self.use_centerness:
            # by default we calculate the centerness on the classifcation banch
            self.conv_centerness = nn.Conv2d(self.clip_dim, 1, 3, padding=1)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_feats=None,
                      rand_bboxes=None,
                      rand_feats=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, gt_feats, rand_bboxes, rand_feats)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        return losses

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        # load the pretrained text embedding again
        if not self.use_cross_correlation and (False in (self.fc_cls.weight.data == self.load_value)):
            print('loading value again')
            with torch.no_grad():
                self.fc_cls.weight.copy_(self.load_value)
            for param in self.fc_cls.parameters():
                param.requires_grad = False        
            # for testing
            if self.filter_base_cate != None:
                print('base_load_value is loaded')
                with torch.no_grad():
                    self.fc_cls_base.weight.copy_(self.base_load_value)
                for param in self.fc_cls_base.parameters():
                    param.requires_grad = False         
        
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def _forward_single(self, x):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        
        ###### for mapping the feature to clip feature space ######
        # for the version of the self.map_to_clip is an linear layer #
        if not self.conv_based_mapping:
            # change the feat dim from ([bs, dim, h, w]) to ([bs, h, w, dim])
            cls_feat = cls_feat.permute([0, 2, 3, 1])
            cls_feat = self.map_to_clip(cls_feat)
        # for the version of the self.map_to_clip is an conv layer #
        else:
            cls_feat = self.map_to_clip(cls_feat)
            # change the feat dim from ([bs, dim, h, w]) to ([bs, h, w, dim])
            cls_feat = cls_feat.permute([0, 2, 3, 1])
        
        ###### for classifier ######
        # for linear based classifier structure
        if not self.use_cross_correlation:  
            ### normalize the cls_feat
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)          
            cls_score = self.fc_cls(cls_feat)
            # for test only, calculate the base score
            if self.filter_base_cate != None:
                base_score = self.fc_cls_base(cls_feat)
                cls_score = torch.cat([cls_score, base_score], dim=-1)        
                
            # change the cls_score and the cls_feat back to original
            # cls_feat would be ([bs, h, w, dim]) => ([bs, dim, h, w])
            # cls_score would be ([bs, h, w, cls_num]) => ([bs, cls_num, h, w])
            cls_feat = cls_feat.permute([0, 3, 1, 2])
            cls_score = cls_score.permute([0, 3, 1, 2])
        # for the cross-correlation layer
        else:
            #([bs, h, w, dim]) => ([bs, dim, h, w])
            cls_feat = cls_feat.permute([0, 3, 1, 2])
            #cls_feat = cls_feat.reshape([-1, self.clip_dim, h, w])
            cls_feat = cls_feat / cls_feat.norm(dim=1, keepdim=True)
            
            #generate the positve feat
            #select the needed text embedding
            #for the image i the cate_idx should be query_gt_labels[i][0]
            #query_feat torch.Size([1, 1024, 43, 48]) support_feat torch.Size([1, 1024, 1, 1])
            #query_feat_input[0] torch.Size([1, 512, 40, 54]) query_gt_labels[0][0] tensor(2, device='cuda:1') support_feat_input[query_gt_labels[i][0]].unsqueeze(dim=0) torch.Size([1, 512, 1, 1]) result: torch.Size([1, 512, 40, 54])
            #print('before aggregation:', cls_feat.shape, self.support_feat.shape)
            
            cls_score = [self.fc_cls(
                    query_feat=cls_feat,
                    support_feat=self.support_feat[i].unsqueeze(dim=0)
                    ) for i in range(self.support_feat.shape[0])]
            #after reshape: 48 1 torch.Size([2, 512, 160, 100])
            if self.correlation_linear_layer:
                cls_score = torch.cat([res_per_cate[0] for res_per_cate in cls_score], dim=1)
            else:
                cls_score = torch.cat([res_per_cate[0].sum(dim=1, keepdim=True) for res_per_cate in cls_score], dim=1)
            #print(cls_score.shape)
        
        ###### for regression ######
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat, reg_feat

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = self._forward_single(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        
        ### multiple the temperature score to the classification score
        cls_score *= self._temperature
        #print('cls_score', cls_score)
        
        if not self.use_centerness:
            return cls_score, bbox_pred, None, cls_feat
        else:
            centerness = self.conv_centerness(cls_feat)
            if not self.induced_centerness:
                cls_score = cls_score * centerness
            return cls_score, bbox_pred, centerness, cls_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_feat'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             cls_feat,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_feats=None, 
             rand_bboxes=None, 
             rand_feats=None,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(cls_feat)
        assert (self.use_centerness and None not in centernesses) or (not self.use_centerness and None in centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        
        # all_level_points dimension 2 means the (y,x) location of the point only for one image
        # since the feature map of two images are the same
        # [torch.Size([16000, 2]), torch.Size([4000, 2]), torch.Size([1000, 2]), torch.Size([260, 2]), torch.Size([70, 2])] 
        # tensor([[   4.,    4.], [  12.,    4.], ...,
        # [ 788., 1276.], [ 796., 1276.]], device='cuda:1')
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, _ = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        # change the feat dim from ([bs, dim, h, w]) => ([bs, h, w, dim]) => ([bs*h*w, dim])
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
                            [points.repeat(num_imgs, 1) for points in all_level_points])

        if self.use_centerness and self.induced_centerness:
            flatten_centerness = [
                centerness.permute(0, 2, 3, 1).reshape(-1)
                for centerness in centernesses
            ]
            flatten_centerness = torch.cat(flatten_centerness) 

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        #print('pos_bbox_targets', pos_bbox_targets.shape)
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        #print('pos_centerness_targets', pos_centerness_targets.shape)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            if self.use_centerness and self.induced_centerness:
                pos_centerness = flatten_centerness[pos_inds]
                loss_centerness = self.loss_centerness(
                    pos_centerness, pos_centerness_targets, avg_factor=num_pos)
            
        else:
            loss_bbox = pos_bbox_preds.sum()

        ########################### for distillation part ##################################
        # only distill the region which is only belong to any forground
        # select the region which is belongs to the foregourd region for distillation
        # also need to consider the weight of the distillation grids
        
        # assign the distillation bbox to the each grid
        all_distill_bboxes = [torch.cat([gt_bbox, rand_bbox], dim=0) 
                              for gt_bbox, rand_bbox in zip(gt_bboxes, rand_bboxes)]
        temp_label = [torch.ones(ele.shape[0]).cuda() for ele in all_distill_bboxes]
        _, matched_distill_bbox, assigned_idx = self.get_targets(all_level_points, all_distill_bboxes,
                                              temp_label)  
        # cls_feat 5 list[tensor] tensor [bs,dim,h,w] [2, 512, 152, 100]
        # assigned_idx list[tuple(tensor)] 2 5  [torch.Size([16000]), torch.Size([4000]), torch.Size([1000]), torch.Size([260]), torch.Size([70])]
        # (tensor([-1, -1, -1,  ..., -1, -1, -1], device='cuda:0'), 
        # tensor([-1, -1, -1,  ..., -1, -1, -1], device='cuda:0'), 
        # tensor([ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
        #  ..., 
        #  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  91,  91,  91,  91,  91,
        #  91,  91,  91,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
        #  ...,
        #  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1], device='cuda:0')
        
        # convert the feat shape, converted cls_feat 5 torch.Size([2, 15200, 512])
        cls_feat = [ele.reshape(list(ele.shape[:-2]) + [-1]).permute([0,2,1]) for ele in cls_feat]
        # aggregate the fg grid prediction base on the idx
        all_predict_feat = []
        for img_id in range(len(assigned_idx)):
            for feat_lvl in range(len(cls_feat)):
                now_idx = assigned_idx[img_id][feat_lvl]
                # select the grid which is not the BG grid
                valid_idx = (now_idx != -1)
                now_feat = cls_feat[feat_lvl][img_id]
                selected_feat = now_feat[valid_idx]
                all_predict_feat.append(selected_feat)
        all_predict_feat = torch.cat(all_predict_feat, dim=0)
        # all_predict_feat torch.Size([4232, 512])
        #print('all_predict_feat', all_predict_feat.shape)
        
        # preprocess the distillation feat
        gt_feats = [patches[:len(gt_bbox)] 
                        for patches, gt_bbox in zip(gt_feats, gt_bboxes)]
        # all_gt_feat 2 torch.Size([111, 512])        
        all_gt_feat = [torch.cat([gt_feat_per_img, rand_feat_per_img], dim=0)
                        for gt_feat_per_img, rand_feat_per_img in zip(gt_feats, rand_feats)]
        # aggregate the fg grid target base on the idx
        all_target_feat = []
        for img_id in range(len(assigned_idx)):
            now_feat = all_gt_feat[img_id]
            for feat_lvl in range(len(cls_feat)):
                now_idx = assigned_idx[img_id][feat_lvl]
                # select the grid which is not the BG grid
                valid_idx = now_idx[now_idx != -1]
                selected_feat = now_feat[valid_idx]
                all_target_feat.append(selected_feat)
        all_target_feat = torch.cat(all_target_feat, dim=0)
        all_target_feat = all_target_feat / all_target_feat.norm(dim=-1, keepdim=True)
        # all_target_feat torch.Size([4232, 512])
        # print('all_target_feat', all_target_feat.shape)
        
        # TODO: following the centerness method, use the centerness target as the weight 
        # use the matched_distill_bbox to calculate the weight following the way we calculate pos_centerness_targets
        distill_loss_value = self.distillation_loss(all_predict_feat, all_target_feat, weight=None)
        
        #### add distillation loss for negative sample ###
        if self.distill_negative:
            # select all negative sample
            all_predict_neg_feat = []
            for img_id in range(len(assigned_idx)):
                for feat_lvl in range(len(cls_feat)):
                    now_idx = assigned_idx[img_id][feat_lvl]
                    # select the grid which is not the BG grid
                    valid_idx = (now_idx == -1)
                    now_feat = cls_feat[feat_lvl][img_id]
                    selected_feat = now_feat[valid_idx]
                    all_predict_neg_feat.append(selected_feat)
            all_predict_neg_feat = torch.cat(all_predict_neg_feat, dim=0)
            
            all_predict_neg_feat_target = torch.zeros(all_predict_neg_feat.shape).cuda()
            distill_loss_neg_value = self.distillation_loss(all_predict_neg_feat, all_predict_neg_feat_target, weight=None)
            distill_loss_value += (distill_loss_neg_value*0.25)
        
        distill_loss_value *= (self.clip_dim * self.distill_loss_factor)

        if self.use_centerness and self.induced_centerness:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_distillation=distill_loss_value,
                loss_centerness=loss_centerness)

        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_distillation=distill_loss_value)


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_feat'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   cls_feat,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert (self.use_centerness and None not in centernesses) or (not self.use_centerness and None in centernesses)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        if self.use_centerness:
            centerness_pred_list = [
                centernesses[i].detach() for i in range(num_levels)
            ]
        else:
            centerness_pred_list = centernesses
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        if not self.use_centerness or (self.use_centerness and not self.induced_centerness):
            result_list = self._get_bboxes_wocenterness(cls_score_list, bbox_pred_list, mlvl_points,
                                        img_shapes, scale_factors, cfg, rescale,
                                        with_nms)
        else:
            result_list = self._get_bboxes(cls_score_list, bbox_pred_list, 
                            centerness_pred_list, mlvl_points,
                            img_shapes, scale_factors, cfg, rescale,
                            with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                        1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                                                batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_bboxes_wocenterness(self,
                    cls_scores,
                    bbox_preds,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).                
                
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(
                cls_scores, bbox_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                1).reshape(batch_size, -1, 4)
            
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, bbox_pred.shape[1], -1).sigmoid()

            points = points.expand(batch_size, -1, 2)
            if self.filter_base_cate != None:
                # add for filter the base categories
                bg_idx = self.num_classes
                ## the filtering procedure
                # BS > BGS and NS 
                max_idx = torch.max(scores, dim=-1)[1]
                novel_bg_idx = (max_idx < bg_idx)            
                # filter the bbox
                scores = scores[novel_bg_idx]
                # assuming that it's using the sigmoid loss and does not have the bg vector
                scores = scores[:, :bg_idx]
                points = points[novel_bg_idx]
                bbox_pred = bbox_pred[novel_bg_idx]
            
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = scores.max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                        1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                                                batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
            concat_lvl_labels 5 [torch.Size([29600]), torch.Size([7400]), torch.Size([1850]), torch.Size([494]), torch.Size([140])] 
            concat_lvl_bbox_targets 5 [torch.Size([29600, 4]), torch.Size([7400, 4]), torch.Size([1850, 4]), torch.Size([494, 4]), torch.Size([140, 4])]
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, target_idx_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        target_idx_list = [
            target_idx.split(num_points, 0)
            for target_idx in target_idx_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets, target_idx_list

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        
        #### the main logic is to see whether a point is in any once of the gt bbox
        #### by looking at whether the left side of the point (xs - gt_bboxes[..., 0])
        #### the right side of the point gt_bboxes[..., 2] - xs
        #### the top side of the point ys - gt_bboxes[..., 1]
        #### the bottom side of the point gt_bboxes[..., 3] - ys
        #### if all the above value are larger than 0 means the point is in one of the gt bboxes
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        
        # need to be modified for distillation
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))
        # keep the area of each gt bbox the shape is torch.Size([204])
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        # repeat areas with the number of the point torch.Size([20267, 204])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        
        # bbox_targets torch.Size([20267, 3, 4]) [point_num, gt_num, 4]
        # this is the relationship between the points and gt bboxes, to see whether a point is in the gt bboxes
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        
        # also need the assigned idx for each grid
        target_idx = min_area_inds
        target_idx[min_area == INF] = -1

        return labels, bbox_targets, target_idx

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
