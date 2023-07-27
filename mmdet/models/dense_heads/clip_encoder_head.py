# Copyright (c) OpenMMLab. All rights reserved.
from enum import unique
from locale import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
from ..builder import HEADS, build_loss, build_backbone, build_head, build_neck
from .anchor_free_head import AnchorFreeHead
import numpy as np
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

_tokenizer = _Tokenizer()



@HEADS.register_module()
class ClipEncoderHead(AnchorFreeHead):
    """This class implement the text encoder of the CLIP
    It take the text as input and generate the text embedding.
    It can also take the feature from the vision encoder to classify the image"""

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 vocab_size,
                 transformer_width,
                 transformer_layers,
                 transformer_heads,
                 sentence_templates,
                 cate_names,
                 embed_dim,
                 context_length,
                 fixed_param=True,
                 open_ln=True,
                 test_with_attribute=False,
                 cate_attribute=None,
                 loss_cls=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 return_test_score=False,
                 use_gt_name=False,
                 use_rand_name=False,
                 use_size_weight=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        '''
        self.bg_cls_weight = 0
        if loss_cls != None:
            class_weight = loss_cls.get('class_weight', None)
            if class_weight is not None and (self.__class__ is ClipEncoderHead):
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
                self.bg_cls_weight = bg_cls_weight'''

        #self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.vision_embed_dim = embed_dim
        self.sentence_templates = sentence_templates
        self.cate_names = cate_names
        self.template_num = len(self.sentence_templates)
        self.fixed_param = fixed_param
        self.open_ln = open_ln
        self.test_with_attribute = test_with_attribute
        self.cate_attribute = cate_attribute
        self.return_test_score = return_test_score
        self.test_with_rand_bboxes = self.test_cfg.get('test_with_rand_bboxes', False) if self.test_cfg is not None else False 
        self.selected_need_feat = self.test_cfg.get('selected_need_feat', None) if self.test_cfg is not None else None 
        self.use_gt_name = use_gt_name
        self.use_rand_name = use_rand_name
        self.from_id_to_cate_name = {i:name for i, name in enumerate(self.cate_names)}
        self.use_size_weight = use_size_weight

        # create the layers
        self._init_layers()
        # fix the model parameter
        if self.fixed_param == True:
            self.fix_model_parameter()        

        # prepare the template with categories name
        self.prepare_the_text_embedding()

        if loss_cls != None:
            self.loss_cls = build_loss(loss_cls)
            if self.loss_cls.use_sigmoid:
                self.cls_out_channels = num_classes
            else:
                self.cls_out_channels = num_classes + 1
        else:
            self.loss_cls = None
        #self.activate = build_activation_layer(self.act_cfg)
        self.text_embeddings = None

    def prepare_the_text_embedding(self):
        # if do not use the attribute
        self.all_cate_tokenize_res = []
        for i, cate_name in enumerate(self.cate_names):
            #sentences_result_for_cate = []
            for template in self.sentence_templates:
                now_sentence = template.replace('{}', cate_name)
                #print(now_sentence)
                tokenized_result = self.tokenize(now_sentence).cuda()
                #sentences_result_for_cate.append(tokenized_result)
                #sentences_result_for_cate = torch.cat(sentences_result_for_cate, dim=0)
                self.all_cate_tokenize_res.append(tokenized_result)
        
        # deal with the attributes
        # need a mapping from the cate_name to attribute
        if self.test_with_attribute:
            self.from_cate_id_to_attr_id = {}
            for cate_id, key in enumerate(self.cate_attribute):
                now_attribute = self.cate_attribute[key]
                # if there is no attribute for this categories
                # just skip this cate
                self.from_cate_id_to_attr_id[cate_id] = []
                if len(now_attribute) == 0:
                    continue 
                for attribute in now_attribute:
                    now_id = int(len(self.all_cate_tokenize_res) / len(self.sentence_templates))
                    self.from_id_to_cate_name[now_id] = attribute
                    for template in self.sentence_templates:
                        now_sentence = template.replace('{}', attribute)
                        #print(now_sentence)
                        tokenized_result = self.tokenize(now_sentence).cuda()
                        #sentences_result_for_cate.append(tokenized_result)
                        #sentences_result_for_cate = torch.cat(sentences_result_for_cate, dim=0)
                        self.all_cate_tokenize_res.append(tokenized_result)
                    self.from_cate_id_to_attr_id[cate_id].append(now_id)

        # concatenate all the result to torch tensor
        self.all_cate_tokenize_res = torch.cat(self.all_cate_tokenize_res, dim=0)

    def prepare_the_text_embedding_subset(self, now_cate_names, img_meta):
        # if do not use the attribute
        cate_tokenize_res = []
        for i, cate_name in enumerate(now_cate_names):
            #sentences_result_for_cate = []
            for template in self.sentence_templates:
                now_sentence = template.replace('{}', cate_name)
                #print(now_sentence)
                tokenized_result = self.tokenize(now_sentence).cuda()
                #sentences_result_for_cate.append(tokenized_result)
                #sentences_result_for_cate = torch.cat(sentences_result_for_cate, dim=0)
                cate_tokenize_res.append(tokenized_result)
        
        cate_tokenize_res = torch.cat(cate_tokenize_res, dim=0)
        return cate_tokenize_res

    def fix_model_parameter(self):
        if self.open_ln == False:
            for param in self.parameters():
                param.requires_grad = False
            print('text head parameters are fixed')
        else:
            #print(self.state_dict())
            for para_name, param in zip(self.state_dict(), self.parameters()):
                #print(para_name, self.state_dict()[para_name].shape, param.shape)
                if 'ln_' not in para_name:
                    param.requires_grad = False
            #for para_name, param in zip(self.state_dict(), self.parameters()):
            #    print(para_name, param.requires_grad, param.shape)            
            print('text head parameters are fixed, with ln open')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.transformer_width))
        self.ln_final = LayerNorm(self.transformer_width)
        transformer_config_dict = dict(type='myTransformer',
                                        width=self.transformer_width,
                                        layers=self.transformer_layers,
                                        heads=self.transformer_heads,
                                        attn_mask=self.build_attention_mask()
                                        )
        self.transformer = build_backbone(transformer_config_dict)
        self.text_projection = nn.Parameter(torch.empty(self.transformer_width, self.vision_embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def tokenize(self, texts, context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def get_bboxes(self):
        pass

    def get_targets(self): 
        pass

    def encode_text(self, text):
        x = self.token_embedding(text) # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def get_text_embedding(self,  gt_labels=None, img_metas=None):
        # for each forward we need to calculate the text embeddings
        # self.all_cate_tokenize_res: tensor tensor.shape = [number of cls * num_of_template, 77]
        # obtain the text embedding [number of cls * num_of_template, 512]
        if (self.training and self.open_ln == True) or self.text_embeddings == None:
            if self.use_gt_name and (self.training and self.open_ln == True):
                # obtain all gt name
                text_embeddings = []
                all_updated_label = []
                if img_metas == None:
                    img_metas = [None for i in range(len(gt_labels))]
                for label_per_img, img_meta in zip(gt_labels, img_metas):
                    unique_gt_label = list(set(label_per_img.detach().cpu().tolist()))
                    from_old_label_to_new_label = {old_label: new_label for new_label, old_label in enumerate(unique_gt_label)}
                    unique_gt_name = [self.from_id_to_cate_name[label] for label in unique_gt_label]
                    
                    # handle the situation which does not have gt
                    if len(unique_gt_name) == 0:
                        print(label_per_img, img_meta)
                        #now_text_embeddings = torch.tensor([0, 512]).cuda()
                        #text_embeddings.append(now_text_embeddings)
                        continue
                    
                    if self.use_rand_name != False:
                        # random sample idx
                        random_choice = np.random.choice(self.num_classes, self.use_rand_name, replace=True)
                        # filter the idx already in the unique_gt_label
                        random_label = [ele for ele in random_choice if ele not in unique_gt_label]
                        # obtain the random name 
                        random_name = [self.from_id_to_cate_name[label] for label in random_label]
                        # concat with unique_gt_name
                        unique_gt_name = unique_gt_name + random_name
                    
                    # get the updated label
                    updated_label = torch.tensor([from_old_label_to_new_label[old_label.item()] for old_label in label_per_img]).cuda()
                    all_updated_label.append(updated_label)
                    
                    # prepare the tokenized_res
                    now_tokenize_res = self.prepare_the_text_embedding_subset(unique_gt_name, img_meta)
                    # encode the text
                    now_text_embeddings = self.encode_text(now_tokenize_res)
                    now_text_embeddings = now_text_embeddings.view(-1, len(self.sentence_templates), now_text_embeddings.shape[-1])
                    # average over all templates: [number_of_cls, 512]
                    now_text_embeddings = torch.mean(now_text_embeddings, dim=1)
                    
                    now_text_embeddings = now_text_embeddings / now_text_embeddings.norm(dim=-1, keepdim=True)
                    text_embeddings.append(now_text_embeddings)
            else:
                text_embeddings = self.encode_text(self.all_cate_tokenize_res)
                # group by the cate_name [number of cls, num_of_template, 512]
                #text_embeddings = text_embeddings.view(len(self.cate_names), -1, text_embeddings.shape[-1])
                text_embeddings = text_embeddings.view(-1, len(self.sentence_templates), text_embeddings.shape[-1])
                # average over all templates: [number_of_cls, 512]
                text_embeddings = torch.mean(text_embeddings, dim=1)
                #path = '/data/zhuoming/code/new_rpn/mmdetection/embedding.pt'
                #print(text_embeddings.shape, 'saving to path', path)
                #torch.save(text_embeddings.cpu(), path)
                # normalized features
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            if not self.training or (self.training and self.open_ln == False):
                self.text_embeddings = text_embeddings
            
            if self.use_gt_name and (self.training and self.open_ln == True):
                return text_embeddings, all_updated_label
            else:
                return text_embeddings
        else:
            #print('testing the update')
            return self.text_embeddings

    def forward(self, feats, img_metas, gt_labels=None):
        """Forward function.

        Args:
            feats (list [Tensor]): len of list is batchsize. [gt_num_in_image, 512]
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores [gt_num_in_batch, cls_out_channels].
        """
        logit_scale = self.logit_scale.exp()
        all_cls_scores_list = []
        if self.use_gt_name and (self.training and self.open_ln == True):
            text_embeddings, updated_label = self.get_text_embedding(gt_labels=gt_labels, img_metas=img_metas)
            # in some situation, one image does not has annotation, at this time the len(feats) == 1
            # while len(text_embeddings) == 2
            # at this situation, just concat the text_embeddings and updated_label, then include it into a list
            #if len(feats) != len(text_embeddings):
            #    text_embeddings = [torch.cat(text_embeddings, dim=0)]
            #    updated_label = [torch.cat(updated_label, dim=0)]
            
            for image_features, text_embedding in zip(feats, text_embeddings):
                if text_embedding.shape[0] == 0:
                    #print('in forward', len(text_embeddings), text_embeddings, len(feats), feats)
                    all_cls_scores_list.append(text_embedding)
                    continue
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # since for the cross entropy loss the input should be logit
                # we do not need the softmax here
                logits_per_image = logit_scale * image_features @ text_embedding.t()
                #cls_scores = (feat @ self.word_embeddings.T).softmax(dim=-1)
                all_cls_scores_list.append(logits_per_image)
            return all_cls_scores_list, updated_label
        else:
            text_embeddings = self.get_text_embedding()
            for image_features in feats:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # since for the cross entropy loss the input should be logit
                # we do not need the softmax here
                logits_per_image = logit_scale * image_features @ text_embeddings.t()
                #cls_scores = (feat @ self.word_embeddings.T).softmax(dim=-1)
                all_cls_scores_list.append(logits_per_image)
            return all_cls_scores_list            

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             cls_scores_list,
             gt_labels_list,
             img_metas,
             gt_bboxes=None):
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
        # prepare the weight
        # using the weight function e ^ -(sqrt(bbox_area) / sqrt(img_area)) and normalize the weight to make the total equal to instance number
        if self.use_size_weight:
            all_weights = []
            for bbox_per_img, image_meta in zip(gt_bboxes, img_metas):
                if bbox_per_img.shape[0] == 0:
                    continue
                img_h, img_w, _ = image_meta['img_shape']
                img_factor = math.sqrt(img_h * img_w)
                bbox_w = bbox_per_img[:, 2] - bbox_per_img[:, 0]
                bbox_h = bbox_per_img[:, 3] - bbox_per_img[:, 1]
                bbox_factor = torch.sqrt(bbox_w * bbox_h)
                weight = - (bbox_factor / img_factor)
                weight = torch.exp(weight)
                # normalize over all the weight
                normalize_factor = bbox_per_img.shape[0] / torch.sum(weight)
                weight *= normalize_factor
                #print(bbox_factor, weight, torch.sum(weight))
                all_weights.append(weight)
        
        # classification loss
        # ori_all_pred = torch.cat(cls_scores_list)
        # ori_all_label = torch.cat(gt_labels_list)
        # ori_all_weights = torch.cat(all_weights)
        # ori_loss_cls = self.loss_cls(ori_all_pred, ori_all_label, weight=ori_all_weights)
        all_cls_loss = []
        all_weight_factor = []
        
        # in some situation since the image without annoation 
        # there is possible that len(cls_scores_list) != len(gt_labels_list)
        # in most of the time is len(gt_labels_list) > len(cls_scores_list)
        if len(cls_scores_list) != len(gt_labels_list):
            cls_scores_list = [ele for ele in cls_scores_list if ele.shape[0] != 0]
            gt_labels_list = [ele for ele in gt_labels_list if ele.shape[0] != 0]
        
        for i, (pred, label) in enumerate(zip(cls_scores_list, gt_labels_list)):
            if self.use_size_weight:
                loss_cls = self.loss_cls(pred, label, weight=all_weights[i])
            else:
                loss_cls = self.loss_cls(pred, label)
            all_cls_loss.append(loss_cls.unsqueeze(dim=0))
            all_weight_factor.append(pred.shape[0])

        all_cls_loss = torch.cat(all_cls_loss, dim=0)
        all_weight_factor = torch.tensor(all_weight_factor).float().cuda()
        all_weight_factor /= torch.sum(all_weight_factor)
        all_cls_loss = all_cls_loss * all_weight_factor
        all_cls_loss = torch.sum(all_cls_loss, dim=0)

        #print('ori:', ori_loss_cls, 'new:', all_cls_loss)
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = all_cls_loss
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
        if hasattr(self, 'use_gt_name') and self.use_gt_name:
            outs, updated_gt_labels = self(x, img_metas, gt_labels)
            loss_inputs = (outs,) + (updated_gt_labels, img_metas, gt_bboxes)
            #print('gt_labels:', [ele.shape for ele in gt_labels], 
            #      'updated_gt_labels:', [ele.shape for ele in updated_gt_labels])
        else:
            outs = self(x, img_metas)
            loss_inputs = (outs,) + (gt_labels, img_metas, gt_bboxes)
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
        # out list[tensor] tensor with shape [gt_per_img, channel]
        softmax = nn.Softmax(dim=1)

        # select the needed feat
        if len(feats[0].shape) > 2 and self.selected_need_feat is not None:
            feats = [feat[:, self.selected_need_feat, :] for feat in feats]
            if feats[0].shape[1] == 1:
                feats = [feat.squeeze(dim=1) for feat in feats]

        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        
        # return the raw predicted result
        if self.return_test_score:
            return outs

        # deal with the score aggregation
        if self.test_with_attribute:
            new_out = []
            for pred_res, img_meta in zip(outs, img_metas):
                pred_res = softmax(pred_res)
                # print the top prediction
                #for i, box_res in enumerate(pred_res):
                #    values, indices = box_res.topk(5)
                #    # Print the result
                #    print("\nTop predictions:\n" + img_meta['filename'] + 'bbox:' , i)
                #    for value, index in zip(values, indices):
                #        #print(imagenet_cate_name[str(index.item())], 100 * value.item())
                #        print(self.from_id_to_cate_name[index.item()], 100 * value.item())

                # pred_res is all prediction of all bboxes in the cate
                for cate_id in self.from_cate_id_to_attr_id:
                    attri_ids = self.from_cate_id_to_attr_id[cate_id]
                    if len(attri_ids) == 0:
                        continue
                    first_id = attri_ids[0]
                    last_id = attri_ids[-1]
                    #sum_for_cate_i = torch.sum(pred_res[:, first_id:last_id+1], dim=1)
                    temp = torch.cat([pred_res[:, cate_id].unsqueeze(dim=-1), pred_res[:, first_id:last_id+1]], dim=1)
                    #print(temp.shape)
                    max_for_cate_i = torch.max(temp, dim=1)[0]
                    #pred_res[:, cate_id] = pred_res[:, cate_id] + sum_for_cate_i
                    pred_res[:, cate_id] = max_for_cate_i
                new_out.append(pred_res[:, :len(self.cate_names)])
            outs = new_out

        # calculate the acc
        # predict_results list(tensor) each tensor is a true false tensor shape [gt_per_img]
        predict_results = []
        for pred, gt_label, gt_bbox, img_meta in zip(outs, gt_labels, gt_bboxes, img_metas):
            # turning the logit to the prob score
            pred_after_softmax = softmax(pred)
            #pred_idx = torch.argmax(pred, dim=1)
            temp = torch.max(pred_after_softmax, dim=1)
            pred_idx = temp[1]
            max_val = temp[0]
            #print('pred_idx', pred_idx, 'test[0]', test[0], 'test[1]', test[1])
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
            # calculate the entropy
            prepared_gt_pred = pred_after_softmax
            prepared_gt_pred[prepared_gt_pred == 0] = 1e-5
            log_result = - torch.log(prepared_gt_pred)
            entro = (log_result * pred_after_softmax).sum(dim=-1)
            #print("entro.shape", entro.shape, "gt_label.shape", gt_label.shape)
            
            # get the GT cos score
            idx = gt_label.reshape(-1 ,1)
            gt_cos_score = pred.gather(1, idx)
            gt_cos_score = gt_cos_score.reshape(1, -1)
            #print('pred', pred, 'idx', idx, 'gt_cos_score', gt_cos_score)
            #print('gt_cos_score', gt_cos_score.shape, 'pred_idx.unsqueeze(dim=0)', pred_idx.unsqueeze(dim=0).shape)

            # concat the gt and the pred result
            if self.test_with_rand_bboxes:
                pred_and_gt = torch.cat([pred_idx.unsqueeze(dim=0).cuda(), torch.full(pred_idx.shape, -1).unsqueeze(dim=0).cuda(), torch.full(pred_idx.shape, -1).unsqueeze(dim=0).cuda(), entro.unsqueeze(dim=0).cuda(), max_val.unsqueeze(dim=0)], dim=0)
            else:
                pred_and_gt = torch.cat([pred_idx.unsqueeze(dim=0).cuda(), gt_label.unsqueeze(dim=0).cuda(), size_result.unsqueeze(dim=0).cuda(), entro.unsqueeze(dim=0).cuda(), max_val.unsqueeze(dim=0), gt_cos_score.cuda()], dim=0)
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
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
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
