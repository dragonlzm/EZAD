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
class ResNetWithVitMultiScale(ResNet):
    ''' This class is for merging the feature from pretrained ViT 
    (For instance, the ViT backbone in the CLIP) 
    with the one from ResNet backbone, to enable the vision backbone with better performance.
    This class ResNetWithVit provide the Fourth version for merging the feature.
    In this version, it change the input into multiple resolutions.
    It merge the feature for multiple resolutions to the feature from the same level from ResNet'''
    
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
                 image_grids_size_list=[(8,8), (4,4), (2,2), (1,1)],
                 vit_feat_level='lvl4'):
        super(ResNetWithVitMultiScale, self).__init__(
                 depth, in_channels, stem_channels, base_channels,
                 num_stages, strides, dilations, out_indices,
                 style, deep_stem, avg_down, frozen_stages, conv_cfg,
                 norm_cfg, norm_eval, dcn, stage_with_dcn, plugins,
                 with_cp, zero_init_residual, pretrained, init_cfg)
        self.vit_backbone_cfg = vit_backbone_cfg
        self.vit_feat_level = vit_feat_level
        self.setup_clip_component()
        self.setup_clip_adapter()
        self.vit_girds_num = int(self.vit_backbone_cfg.input_resolution / self.vit_backbone_cfg.patch_size)
        self.image_grids_size_list = image_grids_size_list
        self.grids_per_level = [x*y for x, y in self.image_grids_size_list]
        self.accumu_grids_per_level = torch.cumsum(torch.tensor(self.grids_per_level), dim=0) - self.grids_per_level[0]
        self.grids_per_img = sum(self.grids_per_level)
        
    def setup_clip_component(self):
        self._preprocess = _transform(self.vit_backbone_cfg.input_resolution)
        
    def setup_clip_adapter(self):
        # inject clip feature 4 times (4th time it is same dimension no need to adapt) 
        self.adapt_mlp_1 = nn.Linear(self.vit_backbone_cfg.width, 64)
        self.adapt_mlp_2 = nn.Linear(self.vit_backbone_cfg.width, 256)
        self.adapt_mlp_3 = nn.Linear(self.vit_backbone_cfg.width, 512)
        self.adapt_mlp_4 = nn.Linear(self.vit_backbone_cfg.width, 1024)
        
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
        with torch.no_grad():
            x = self.clip_pre_transformer(x) # [bs, 257, 1024]
        return x
    
    def clip_step_2(self, x):
        with torch.no_grad():
            # for layer [0-4] for small model, [0-8] for large model
            for i in range(self.vit_backbone_cfg.layers // 3):
                x = self.clip_transfomer(x, i)
        return x
    
    def clip_step_3(self, x):
        with torch.no_grad():
            # for layer [4-8] for small model, [8-16] for large model
            for i in range(self.vit_backbone_cfg.layers // 3, self.vit_backbone_cfg.layers // 3 * 2):
                x = self.clip_transfomer(x, i)
        return x
    
    def clip_step_4(self, x):
        with torch.no_grad():
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

    def merge(self, clip_x, res_x, step_idx):
        # clip_x torch.Size([2, 56, 56, 768])
        if(step_idx == 1):
            # [2, 49, 768] => [2 * 49, 64]
            reshape_x = self.adapt_mlp_1(clip_x) # [bs * self.vit_girds_num*self.vit_girds_num, 64]
        elif(step_idx == 2):
            reshape_x = self.adapt_mlp_2(clip_x) # [bs * self.vit_girds_num*self.vit_girds_num, 256]
        elif(step_idx == 3):
            reshape_x = self.adapt_mlp_3(clip_x) # [bs * self.vit_girds_num*self.vit_girds_num, 512]
        elif(step_idx == 4):
            reshape_x = self.adapt_mlp_4(clip_x) # [bs * self.vit_girds_num*self.vit_girds_num, 1024]
        #print('after convert:', reshape_x.shape)
    
        #reshape_x = reshape_x.reshape(clip_x.size(0), self.vit_girds_num, self.vit_girds_num, -1) # [bs, self.vit_girds_num, self.vit_girds_num, -]
        reshape_x = reshape_x.permute(0, 3, 1, 2)
        
        reshape_x = F.interpolate(reshape_x, size=(res_x.size(2), res_x.size(3)), mode='bicubic', align_corners=False) #  [bs, 64, 200, 304]
        merge_x = reshape_x + res_x
        #print('after merge:', merge_x.shape, res_x.shape)
        return merge_x
    
    def preprocess(self, ori_images):
        # handle the test image
        if len(ori_images[0].shape) == 4:
            ori_images = [ori_image.squeeze(dim=0) for ori_image in ori_images]
        if ori_images[0].shape[0] == 3:
            ori_images = [ori_image.permute(1,2,0) for ori_image in ori_images]
        ori_images = [ori_image.cpu().numpy() for ori_image in ori_images]
        #print('in preprocessing', [ele.shape for ele in ori_images])
        
        # list[tensor(800, 1216, 3), tensor(800, 1216, 3)](H, W, C)
        result = []
        for img in ori_images:
            H, W, channel = img.shape
            for h_patch_num, w_patch_num in self.image_grids_size_list:
                patch_H, patch_W = H / h_patch_num, W / w_patch_num
                #print('in preprocess', h_patch_num, w_patch_num, patch_H, patch_W)
                h_pos = [int(patch_H) * i for i in range(h_patch_num + 1)]
                w_pos = [int(patch_W) * i for i in range(w_patch_num + 1)]

                for i in range(h_patch_num):
                    h_start_pos = h_pos[i]
                    h_end_pos = h_pos[i+1]
                    for j in range(w_patch_num):
                        w_start_pos = w_pos[j]
                        w_end_pos = w_pos[j+1]
                        # cropping the img into the patches which size is (H/8) * (W/8)
                        # use the numpy to crop the image
                        # img shape: torch.Size([2, 3, 800, 1088])
                        now_patch = img[h_start_pos: h_end_pos, w_start_pos: w_end_pos, :]
                        PIL_image = Image.fromarray(np.uint8(now_patch))
                        # do the preprocessing
                        new_patch = self._preprocess(PIL_image)
                        result.append(new_patch.unsqueeze(dim=0))        
        result = torch.cat(result, dim=0).cuda()
        return result
    
    def get_per_level_vit_feat(self, all_level_feats, batch_size):
        # the order of the feat should be
        # [(8,8), (4,4), (2,2), (1,1)] * batch size
        
        # conduct transfer per image
        all_image_results = []
        for i in range(batch_size):
            result_per_image = []
            # conduct transfer per level
            for j, (h_patch_num, w_patch_num) in enumerate(self.image_grids_size_list):
                all_result_per_lvl = []
                #conduct transfer per row
                for k in range(h_patch_num):
                    idx_start = i * self.grids_per_img + self.accumu_grids_per_level[j] + k * w_patch_num
                    idx_end = i * self.grids_per_img + self.accumu_grids_per_level[j] + k * w_patch_num + w_patch_num
                    all_feat_per_row = all_level_feats[idx_start: idx_end, :, :]
                    # the shape of all_feat_per_row should be [w_patch_num, self.vit_girds_num*self.vit_girds_num, 768], [8, 50, 768]
                    # remove the first ele [8, 50, 768] => [8, 49, 768]
                    all_feat_per_row = all_feat_per_row[:,1:,:] 
                    # reshape [8, 49, 768] => [8, 7, 7, 768]
                    all_feat_per_row = all_feat_per_row.reshape(all_feat_per_row.shape[0], self.vit_girds_num, 
                                                                self.vit_girds_num, all_feat_per_row.shape[-1])
                    # convert to list list(tensor[7, 7, 768])
                    all_feat_per_row = [ele for ele in all_feat_per_row]
                    # concat along the width dimension (x dim) list(tensor[7, 7, 768]) => tensor[7, 56, 768]
                    all_feat_per_row = torch.cat(all_feat_per_row, dim=-2)
                    all_result_per_lvl.append(all_feat_per_row)
                # concat along the hight dimension (y dim) list(tensor[7, 56, 768]) => tensor[56, 56, 768]
                all_result_per_lvl = torch.cat(all_result_per_lvl, dim=0)
                result_per_image.append(all_result_per_lvl)
            all_image_results.append(result_per_image)
            
        # order the feats first by feat level
        feat_by_level = []
        for i, _ in enumerate(self.image_grids_size_list):
            feat_per_lvl = []
            for j in range(batch_size):
                feat_per_lvl.append(all_image_results[j][i].unsqueeze(dim=0))
            feat_per_lvl = torch.cat(feat_per_lvl, dim=0)
            feat_by_level.append(feat_per_lvl)
        
        return feat_by_level        
    
    def forward(self, img, ori_image):
        # in this version the ori_image should have the same size as the img
        # img and ori_image torch.Size([2, 3, 1280, 800]) (N, C, H, W)
        # in testing the ori_image list[torch.Size([1, 3, 1024, 1024])]
        # for para_name, param in zip(self.clip_visual_model.state_dict(), self.clip_visual_model.parameters()):
        #     if para_name == 'ln_post.bias':
        #         print(para_name, param.requires_grad, param.shape, param)
        with torch.no_grad():
            # split each image into one tensor
            ori_image = [image for image in ori_image]
            ori_image_patches = self.preprocess(ori_image)
            # after preprocessing torch.Size([170, 3, 224, 224])
            
        # ori_image_patches [128, 3, 224, 224] [bs * h_patch_num * w_patch_num, 3, 224, 224]
        # clip forward
        clip1 = self.clip_step_1(ori_image_patches)
        if self.vit_feat_level == 'lvl1':
            vit_feat = clip1
        elif self.vit_feat_level == 'lvl2':
            clip2 = self.clip_step_2(clip1)
            vit_feat = clip2
        elif self.vit_feat_level == 'lvl3':
            clip2 = self.clip_step_2(clip1)
            clip3 = self.clip_step_3(clip2)
            vit_feat = clip3
        else:
            clip2 = self.clip_step_2(clip1)
            clip3 = self.clip_step_3(clip2)
            clip4 = self.clip_step_4(clip3)
            vit_feat = clip4
        bs = len(ori_image)
        
        # clip4 torch.Size([170, 50, 768])
        feat_by_level = self.get_per_level_vit_feat(vit_feat, bs)
        # feat_by_level 4 clip_x [torch.Size([2, 56, 56, 768]), torch.Size([2, 28, 28, 768]), torch.Size([2, 14, 14, 768]), torch.Size([2, 7, 7, 768])]
        
        # clip1: torch.Size([128, 50, 768]) [bs * h_patch_num * w_patch_num, 
        # self.vit_girds_num * self.vit_girds_num + 1, 224, 224]
        res1 = self.res_step_1(img)
        merge1 = self.merge(feat_by_level[0], res1, 1)
        
        outs = []
        res2 = self.res_step(merge1, 0)
        merge2 = self.merge(feat_by_level[1], res2, 2)
        outs.append(merge2)        
        
        res3 = self.res_step(merge2, 1)
        merge3 = self.merge(feat_by_level[2], res3, 3)
        outs.append(merge3)
        
        res4 = self.res_step(merge3, 2)
        merge4 = self.merge(feat_by_level[3], res4, 4)
        outs.append(merge4)
        
        res5 = self.res_step(merge4, 3)
        outs.append(res5)
        
        return outs