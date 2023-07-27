# Copyright (c) OpenMMLab. All rights reserved.
from hashlib import new
import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import BasicBlock, Bottleneck, ResNet
from ..builder import build_backbone
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from collections import OrderedDict
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import numpy as np

# setup device
# if(torch.cuda.is_available()):
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
    
# CLIP_CKPT_DOWNLOAD_ROOT = 'models_ckpt'

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


@BACKBONES.register_module()
class ResNetWithVitV3(ResNet):
    ''' This class is for merging the feature from pretrained ViT 
    (For instance, the ViT backbone in the CLIP) 
    with the one from ResNet backbone, to enable the vision backbone with better perfromance.
    This class ResNetWithVit provide the third version for merging the feature.
    In this version, it concat the features from all levels from ViT and merge it into on or more specific level'''    
    def __init__(self, 
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 #clip_architecture="ViT-L/14"):
                 vit_backbone_cfg=None,
                 merge_step=dict(merge1='from_vit_1', merge2='from_vit_2', merge3='from_vit_3', merge4='from_vit_4'),
                 merge_method='add',
                 merge_with_mlp=False):
        super(ResNetWithVitV3, self).__init__(
                 depth, in_channels, stem_channels, base_channels,
                 num_stages, strides, dilations, out_indices,
                 style, deep_stem, avg_down, frozen_stages, conv_cfg,
                 norm_cfg, norm_eval, dcn, stage_with_dcn, plugins,
                 with_cp, zero_init_residual, pretrained, init_cfg)
        # preprocessing for the merge_step
        if isinstance(merge_step, list):
            new_merge_step = {}
            for ele in merge_step:
                # default setting
                if '=' not in ele:
                    lvl_num = ele[-1]
                    new_merge_step[ele] = 'from_vit_' + lvl_num
                else:
                    res = ele.split('=')
                    new_merge_step[res[0]] = res[1]
            merge_step = new_merge_step
        self.merge_step = merge_step
        self.merge_method = merge_method
        self.merge_with_mlp = merge_with_mlp
        self.vit_backbone_cfg = vit_backbone_cfg
        self.setup_clip_component()
        self.setup_clip_adapter()
        self.girds_num = int(self.vit_backbone_cfg.input_resolution / self.vit_backbone_cfg.patch_size)
            
        
    def setup_clip_component(self):
        # setup number of layer in transformer
        # if (clip_architecture in ["ViT-L/14", "ViT-L/14@336px"]):
        #     self.transformer_layer = 24
        # elif (clip_architecture in ["ViT-B/32","ViT-B/16"]):
        #     self.transformer_layer = 12
        # else:
        #     raise TypeError("wrong architecture choice!")     
        # load model
        self._preprocess = _transform(self.vit_backbone_cfg.input_resolution)
        #self.clip_visual_model = build_backbone(self.vit_backbone_cfg)
        
        # with torch.no_grad():
        #    clip_model, self.preprocess = clip.load(download_root=CLIP_CKPT_DOWNLOAD_ROOT, name=clip_architecture, device=device)
        #    self.clip_visual_model = clip_model.visual      
        
    def setup_clip_adapter(self):
        # inject clip feature 4 times (4th time it is same dimension no need to adapt)
        if 'merge1' in self.merge_step:
            resnet_dim = 64
            in_dim = self.vit_backbone_cfg.width
            if not self.merge_with_mlp:
                self.adapt_layer_1 = nn.Linear(in_dim*4, resnet_dim)
            else:
                self.adapt_layer_1 = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(in_dim*4, 1024)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("dropout", nn.Dropout(0.1)),
                    ("c_proj", nn.Linear(1024, resnet_dim))
                ]))
        
        if 'merge2' in self.merge_step:
            resnet_dim = 256
            in_dim = self.vit_backbone_cfg.width
            if not self.merge_with_mlp:
                self.adapt_layer_2 = nn.Linear(in_dim*4, resnet_dim)
            else:
                self.adapt_layer_2 = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(in_dim*4, 1024)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("dropout", nn.Dropout(0.1)),
                    ("c_proj", nn.Linear(1024, resnet_dim))
                ]))
        
        if 'merge3' in self.merge_step:
            resnet_dim = 512
            in_dim = self.vit_backbone_cfg.width
            if not self.merge_with_mlp:
                self.adapt_layer_3 = nn.Linear(in_dim*4, resnet_dim)
            else:
                self.adapt_layer_3 = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(in_dim*4, 1024)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("dropout", nn.Dropout(0.1)),
                    ("c_proj", nn.Linear(1024, resnet_dim))
                ]))
        
        if 'merge4' in self.merge_step:
            resnet_dim = 1024
            in_dim = self.vit_backbone_cfg.width
            if not self.merge_with_mlp:
                self.adapt_layer_4 = nn.Linear(in_dim*4, resnet_dim)
            else:
                self.adapt_layer_4 = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(in_dim*4, 1024)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("dropout", nn.Dropout(0.1)),
                    ("c_proj", nn.Linear(1024, resnet_dim))
                ]))
        
    def clip_pre_transformer(self, x):
        # x = self.preprocess(x)
        x = self.clip_visual_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_visual_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_visual_model.positional_embedding.to(x.dtype)
        x = self.clip_visual_model.ln_pre(x)
        return x
                 
    def clip_transfomer(self, x, layer):
        x = x.permute(1, 0, 2)
        x = self.clip_visual_model.transformer.resblocks[layer](x)
        x = x.permute(1, 0, 2)
        return x
    
    def clip_post_transformer(self, x):
        x = self.clip_visual_model.ln_post(x[:, 0, :])
        if self.clip_visual_model.proj is not None:
            x = x @ self.clip_visual_model.proj
        return x
        
    def clip_step_1(self, x):
        x = self.clip_pre_transformer(x) # [bs, 257, 1024]
        return x
    
    def clip_step_2(self, x):
        # for layer [0-4] for small model, [0-8] for large model
        for i in range(self.vit_backbone_cfg.layers // 3):
            x = self.clip_transfomer(x, i)
        return x
    
    def clip_step_3(self, x):
        # for layer [4-8] for small model, [8-16] for large model
        for i in range(self.vit_backbone_cfg.layers // 3, self.vit_backbone_cfg.layers // 3 * 2):
            x = self.clip_transfomer(x, i)
        return x
    
    def clip_step_4(self, x):
        # for layer [8-12] for small model, [16-24] for large model
        for i in range(self.vit_backbone_cfg.layers // 3 * 2, self.vit_backbone_cfg.layers):
            x = self.clip_transfomer(x, i)
        return x
    
    def res_step_1(self, x):
        # resnet stream
        if(self.deep_stem):
            x = self.stem(x)
        else:
            x = self.conv1(x) # first downsize
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x) # second downsize
        return x
        
    def res_step(self, x, res_layer_idx):
        res_layer = getattr(self, self.res_layers[res_layer_idx])
        x = res_layer(x)
        return x

    def add_merge(self, res_x, step_idx):
        bs, _, clip_dim = self.clip1.shape
        reshape_x = [self.clip1[:,1:,:], self.clip2[:,1:,:], self.clip3[:,1:,:], self.clip4[:,1:,:]] # [bs, self.girds_num*self.girds_num, 1024] / [2, 49, 768]
        reshape_x = torch.cat(reshape_x, dim=-1)
        
        #print('before convert:', reshape_x.shape)
        if(step_idx == 1):
            # [2, 49, 768] => [2 * 49, 64]
            reshape_x = self.adapt_layer_1(reshape_x) # [bs * self.girds_num*self.girds_num, 64]
        elif(step_idx == 2):
            reshape_x = self.adapt_layer_2(reshape_x) # [bs * self.girds_num*self.girds_num, 256]
        elif(step_idx == 3):
            reshape_x = self.adapt_layer_3(reshape_x) # [bs * self.girds_num*self.girds_num, 512]
        elif(step_idx == 4):
            reshape_x = self.adapt_layer_4(reshape_x) # [bs * self.girds_num*self.girds_num, 1024]
        #print('after convert:', reshape_x.shape)
    
        reshape_x = reshape_x.reshape(bs, self.girds_num, self.girds_num, -1) # [bs, self.girds_num, self.girds_num, -]
        reshape_x = reshape_x.permute(0, 3, 1, 2)
        #print('after reshape:', reshape_x.shape)
        
        reshape_x = F.interpolate(reshape_x, size=(res_x.size(2), res_x.size(3)), mode='bicubic', align_corners=False) # [bs, 200, 304, 64]
        #print('after interpolate', reshape_x.shape)
        #reshape_x = reshape_x.permute(0, 3, 1, 2) # [bs, 64, 200, 304]
        #print('before merge:', reshape_x.shape, res_x.shape)
        merge_x = reshape_x + res_x
        #print('after merge:', merge_x.shape, res_x.shape)
        return merge_x
    
    def cat_merge(self, res_x, step_idx):
        reshape_x = clip_x[:,1:,:] # [bs, self.girds_num*self.girds_num, 1024] / [2, 49, 768]
        reshape_x = reshape_x.reshape(clip_x.size(0), self.girds_num, self.girds_num, -1) # [bs, self.girds_num, self.girds_num, vit_dim]
        reshape_x = reshape_x.permute(0, 3, 1, 2) # [bs, vit_dim, self.girds_num, self.girds_num]
        reshape_x = F.interpolate(reshape_x, size=(res_x.size(2), res_x.size(3)), mode='bicubic', align_corners=False) # [bs, vit_dim, res_x.size(2), res_x.size(3)]
        #print('after interolation:', reshape_x.shape, 'res_x', res_x.shape)
        
        # concat the feature
        reshape_x = torch.cat([reshape_x, res_x], dim=1) # [bs, vit_dim + res_dim, res_x.size(2), res_x.size(3)]
        # permute the dim
        reshape_x = reshape_x.permute([0, 2, 3, 1]) # [bs, res_x.size(2), res_x.size(3), vit_dim + res_dim]
        
        if(step_idx == 1):
            reshape_x = self.adapt_layer_1(reshape_x) # [bs, res_x.size(2), res_x.size(3), 64]
        elif(step_idx == 2):
            reshape_x = self.adapt_layer_2(reshape_x) # [bs, res_x.size(2), res_x.size(3), 256]
        elif(step_idx == 3):
            reshape_x = self.adapt_layer_3(reshape_x) # [bs, res_x.size(2), res_x.size(3), 512]
        elif(step_idx == 4):
            reshape_x = self.adapt_layer_4(reshape_x) # [bs, res_x.size(2), res_x.size(3), 1024]
            
        # permute the dim back
        reshape_x = reshape_x.permute(0, 3, 1, 2).contiguous()
        return reshape_x    
    
    def merge(self, res_x, step_idx):
        if self.merge_method == 'add':
            return self.add_merge(res_x, step_idx)
        elif self.merge_method == 'cat':
            return self.cat_merge(res_x, step_idx)

    def preprocess(self, ori_images):
        ori_images = [ori_image for ori_image in ori_images]
        if len(ori_images[0].shape) == 4:
            ori_images = [ori_image.squeeze(dim=0) for ori_image in ori_images]
        if ori_images[0].shape[0] == 3:
            ori_images = [ori_image.permute(1,2,0) for ori_image in ori_images]
        ori_images = [ori_image.cpu().numpy() for ori_image in ori_images]

        all_images = []
        for img in ori_images:
            PIL_image = Image.fromarray(np.uint8(img))
            # do the preprocessing
            new_image = self._preprocess(PIL_image)
            all_images.append(new_image.unsqueeze(dim=0))
        
        all_images = torch.cat(all_images, dim=0).cuda()
        return all_images
    
    def forward(self, img, ori_image):
        #img torch.Size([2, 3, 1280, 800]) ori_image torch.Size([2, 1024, 1024, 3])
        # in testing the ori_image list[torch.Size([1, 3, 1024, 1024])]
        # for para_name, param in zip(self.clip_visual_model.state_dict(), self.clip_visual_model.parameters()):
        #     if para_name == 'ln_post.bias':
        #         print(para_name, param.requires_grad, param.shape, param)        
        
        ori_image = self.preprocess(ori_image)
            
        # res0: regular resnet backbone input
        # clip0: clip input with corresponding preprocessing
        self.clip1 = self.clip_step_1(ori_image)
        self.clip2 = self.clip_step_2(self.clip1)
        self.clip3 = self.clip_step_3(self.clip2)
        self.clip4 = self.clip_step_4(self.clip3)
        
        res1 = self.res_step_1(img)
        if 'merge1' in self.merge_step:
            merge1 = self.merge(res1, 1)
        else:
            merge1 = res1
        
        outs = []
        res2 = self.res_step(merge1, 0)
        if 'merge2' in self.merge_step:
            merge2 = self.merge(res2, 2)
        else:
            merge2 = res2
        outs.append(merge2)
        
        res3 = self.res_step(merge2, 1)
        if 'merge3' in self.merge_step:
            merge3 = self.merge(res3, 3)
        else:
            merge3 = res3
        outs.append(merge3)
        
        res4 = self.res_step(merge3, 2)
        if 'merge4' in self.merge_step:
            merge4 = self.merge(res4, 4)
        else:
            merge4 = res4
        outs.append(merge4)
        
        res5 = self.res_step(merge4, 3)
        outs.append(res5)
        
        return outs