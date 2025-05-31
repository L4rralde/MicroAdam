import torch
from torch.optim.optimizer import Optimizer
import math


class MicroAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, compress_fn=None, decompress_fn=None):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.compress_fn = compress_fn  # Should reduce memory (e.g., quantize)
        self.decompress_fn = decompress_fn  # Should recover approximate tensor
        super(MicroAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    # Initialize state
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, device=p.data.device)  # m_0 = 0
                    state['exp_avg_sq'] = torch.zeros_like(p.data, device=p.data.device)  # v_0 = 0
                    state['error_buffer'] = torch.zeros_like(p.data, device=p.data.device) # For feedback. e1 = 0

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] #m_t, v_t
                beta1, beta2 = group['betas']
                state['step'] += 1

                grad = p.grad.data #
                # Optional error feedback mechanism
                if self.decompress_fn and self.compress_fn: #If (must be) provided
                    # Apply error feedback + compression
                    grad += state['error_buffer']
                    compressed_grad = self.compress_fn(grad)
                    grad_hat = self.decompress_fn(compressed_grad)
                    state['error_buffer'] = grad - grad_hat
                    grad = grad_hat  # use decompressed estimate

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
