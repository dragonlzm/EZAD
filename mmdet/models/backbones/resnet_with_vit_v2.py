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
class ResNetWithVitV2(ResNet):
    ''' This class is for merging the feature from pretrained ViT 
    (For instance, the ViT backbone in the CLIP) 
    with the one from ResNet backbone, to enable the vision backbone with better performance.
    This class ResNetWithVit provide the second version for merging the feature.
    In this version, it merge all ViT feature level to respective ResNet feature level'''    
    
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
                 image_grids_num=(8,8)):
        super(ResNetWithVitV2, self).__init__(
                 depth, in_channels, stem_channels, base_channels,
                 num_stages, strides, dilations, out_indices,
                 style, deep_stem, avg_down, frozen_stages, conv_cfg,
                 norm_cfg, norm_eval, dcn, stage_with_dcn, plugins,
                 with_cp, zero_init_residual, pretrained, init_cfg)
        self.vit_backbone_cfg = vit_backbone_cfg
        self.setup_clip_component()
        self.setup_clip_adapter()
        self.vit_girds_num = int(self.vit_backbone_cfg.input_resolution / self.vit_backbone_cfg.patch_size)
        self.image_grids_num = image_grids_num

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
        h_patch_num, w_patch_num = self.image_grids_num
        # clip1: torch.Size([128, 50, 768]) [bs * h_patch_num * w_patch_num, 
        # self.vit_girds_num * self.vit_girds_num + 1, 224, 224]
        reshape_x = clip_x[:,0,:].reshape(res_x.shape[0], h_patch_num, w_patch_num, clip_x.shape[-1])
        # [bs, self.vit_girds_num*self.vit_girds_num, 1024] / [2, 49, 768]
        #print('before convert:', reshape_x.shape)
        if(step_idx == 1):
            # [2, 49, 768] => [2 * 49, 64]
            reshape_x = self.adapt_mlp_1(reshape_x) # [bs * self.vit_girds_num*self.vit_girds_num, 64]
        elif(step_idx == 2):
            reshape_x = self.adapt_mlp_2(reshape_x) # [bs * self.vit_girds_num*self.vit_girds_num, 256]
        elif(step_idx == 3):
            reshape_x = self.adapt_mlp_3(reshape_x) # [bs * self.vit_girds_num*self.vit_girds_num, 512]
        elif(step_idx == 4):
            reshape_x = self.adapt_mlp_4(reshape_x) # [bs * self.vit_girds_num*self.vit_girds_num, 1024]
        #print('after convert:', reshape_x.shape)
    
        #reshape_x = reshape_x.reshape(clip_x.size(0), self.vit_girds_num, self.vit_girds_num, -1) # [bs, self.vit_girds_num, self.vit_girds_num, -]
        reshape_x = reshape_x.permute(0, 3, 1, 2)
        #print('after reshape:', reshape_x.shape)
        
        reshape_x = F.interpolate(reshape_x, size=(res_x.size(2), res_x.size(3)), mode='bicubic', align_corners=False) # [bs, 200, 304, 64]
        #print('after interpolate', reshape_x.shape)
        #reshape_x = reshape_x.permute(0, 3, 1, 2) # [bs, 64, 200, 304]
        #print('before merge:', reshape_x.shape, res_x.shape)
        merge_x = reshape_x + res_x
        #print('after merge:', merge_x.shape, res_x.shape)
        return merge_x
    
    def preprocess(self, ori_images):
        ori_images = [ori_image for ori_image in ori_images]
        if len(ori_images[0].shape) == 4:
            ori_images = [ori_image.squeeze(dim=0) for ori_image in ori_images]
        if ori_images[0].shape[0] == 3:
            ori_images = [ori_image.permute(1,2,0) for ori_image in ori_images]
        ori_images = [ori_image.cpu().numpy() for ori_image in ori_images]
        #print('in preprocessing', [ele.shape for ele in ori_images])
        
        # list[tensor(800, 1216, 3), tensor(800, 1216, 3)](H, W, C)

        # all_images = []
        # for img in ori_images:
        #     PIL_image = Image.fromarray(np.uint8(img))
        #     # do the preprocessing
        #     new_image = self._preprocess(PIL_image)
        #     all_images.append(new_image.unsqueeze(dim=0))
        
        # all_images = torch.cat(all_images, dim=0).cuda()
        
        result = []
        h_patch_num, w_patch_num = self.image_grids_num
        for img in ori_images:
            H, W, channel = img.shape
            patch_H, patch_W = H / h_patch_num, W / w_patch_num
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
    
    def forward(self, img, ori_image):
        # in this version the ori_image should have the same size as the img
        # img and ori_image torch.Size([2, 3, 1280, 800]) (N, C, H, W)
        # in testing the ori_image list[torch.Size([1, 3, 1024, 1024])]
        # for para_name, param in zip(self.clip_visual_model.state_dict(), self.clip_visual_model.parameters()):
        #     if para_name == 'ln_post.bias':
        #         print(para_name, param.requires_grad, param.shape, param)
        with torch.no_grad():
            ori_image_patches = self.preprocess(ori_image)
            
        # ori_image_patches [128, 3, 224, 224] [bs * h_patch_num * w_patch_num, 3, 224, 224]
        # res0: regular resnet backbone input
        # clip0: clip input with corresponding preprocessing
        clip1 = self.clip_step_1(ori_image_patches)
        # clip1: torch.Size([128, 50, 768]) [bs * h_patch_num * w_patch_num, 
        # self.vit_girds_num * self.vit_girds_num + 1, 224, 224]
        res1 = self.res_step_1(img)
        merge1 = self.merge(clip1, res1, 1)
        
        outs = []
        clip2 = self.clip_step_2(clip1)
        res2 = self.res_step(merge1, 0)
        merge2 = self.merge(clip2, res2, 2)
        outs.append(merge2)
        
        clip3 = self.clip_step_3(clip2)
        res3 = self.res_step(merge2, 1)
        merge3 = self.merge(clip3, res3, 3)
        outs.append(merge3)
        
        clip4 = self.clip_step_4(clip3)
        res4 = self.res_step(merge3, 2)
        merge4 = self.merge(clip4, res4, 4)
        outs.append(merge4)
        
        res5 = self.res_step(merge4, 3)
        outs.append(res5)
        
        return outs