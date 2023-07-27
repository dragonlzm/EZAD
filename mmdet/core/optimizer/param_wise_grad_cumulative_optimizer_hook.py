# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
import torch
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version

@HOOKS.register_module()
class ParamWiseGradientCumulativeOptimizerHook(Hook):
    # this is for applying different grad clip to different parameter
    # and also cumulate the grad for multiple iterations
    # to simulate the large batch size
    def __init__(self, grad_clip=None, cumulative_iters=1):
        # in ParamWiseOptimizerHook, the grad_clip would be a dict
        # the key of the dict should the key the module has
        # for instance {'encoder': dict(max_norm=10, norm_type=2), 'other': dict(max_norm=10, norm_type=2)}
        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
            f'cumulative_iters only accepts positive int, but got ' \
            f'{type(cumulative_iters)} instead.'
        self.grad_clip = grad_clip
        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def has_batch_norm(self, module):
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def clip_grads(self, params, config):
        # filter out the parameter that needs gradients
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        
        if len(params) > 0:
            # grad_returned is a value:tensor(294.0063, device='cuda:1') should be the norm over all parameters
            grad_returned = clip_grad.clip_grad_norm_(params, **config)
            #print('grad_returned', grad_returned)
            return grad_returned

    def _init(self, runner):
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = runner.max_iters - runner.iter

        self.divisible_iters = (
            residual_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def _grad_clip(self, runner):
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
        return grad_norm

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = runner.outputs['loss']
        loss = loss / loss_factor
        loss.backward()

        if (self.every_n_iters(runner, self.cumulative_iters)
                or self.is_last_iter(runner)):

            if self.grad_clip is not None:
                grad_norm = self._grad_clip(runner)
                
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            runner.optimizer.step()
            runner.optimizer.zero_grad()


