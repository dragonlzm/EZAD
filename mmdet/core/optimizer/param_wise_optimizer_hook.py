# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class ParamWiseOptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        # in ParamWiseOptimizerHook, the grad_clip would be a dict
        # the key of the dict should the key the module has
        # for instance {'encoder': dict(max_norm=10, norm_type=2), 'other': dict(max_norm=10, norm_type=2)}
        self.grad_clip = grad_clip

    def clip_grads(self, params, config):
        # filter out the parameter that needs gradients
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        
        if len(params) > 0:
            # grad_returned is a value:tensor(294.0063, device='cuda:1') should be the norm over all parameters
            grad_returned = clip_grad.clip_grad_norm_(params, **config)
            #print('grad_returned', grad_returned)
            return grad_returned

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        #needed_para = []
        if self.grad_clip is not None:
            searched_param = set()
            all_grads = []
            for param_key in self.grad_clip:
                if param_key == 'other':
                    continue
                now_param_group = []
                for name, module in runner.model.named_modules():
                    if param_key in name:
                        for key, value in module.named_parameters(recurse=False):
                            now_param_group.append(value)
                            # save the assigned params
                        searched_param.add(name)
                #print('param_key', param_key, len(now_param_group))
                if len(now_param_group) == 0:
                    continue
                now_config = self.grad_clip[param_key]
                grad_norm = self.clip_grads(now_param_group, now_config)
                all_grads.append(grad_norm.unsqueeze(dim=0))
                #print('param_key', param_key, 'grad_norm', grad_norm, 'now_config', now_config)
            
            # handle the rest of the parameters
            if 'other' in self.grad_clip:
                now_param_group = []
                for name, module in runner.model.named_modules():
                    if name not in searched_param:
                        for key, value in module.named_parameters(recurse=False):
                            now_param_group.append(value)
                            # save the assigned params
                        searched_param.add(name)
                now_config = self.grad_clip['other']
                grad_norm = self.clip_grads(now_param_group, now_config)
                all_grads.append(grad_norm.unsqueeze(dim=0))
            grad_norm = torch.sum(torch.cat(all_grads, dim=0))
            
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
            #print('grad_norm', grad_norm)
        runner.optimizer.step()
