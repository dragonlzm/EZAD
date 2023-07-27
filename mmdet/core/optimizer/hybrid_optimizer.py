import torch
import math
import torch
from mmcv.runner.optimizer import OPTIMIZERS


@OPTIMIZERS.register_module()
class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(HybridOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HybridOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("optimizer", "SGD")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if group["optimizer"] == "SGD":
                    weight_decay = group['weight_decay']
                    momentum = group['momentum']
                    dampening = group['dampening']

                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        d_p = buf
                    p.add_(d_p, alpha=-group['lr'])

                elif group["optimizer"] == "ADAMW":
                    # Perform stepweight decay
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                    # Perform optimization step
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1
                    
                    #print("group['lr']", group['lr'])

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    raise NotImplementedError

        return loss
