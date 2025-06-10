import torch
import math

from micro_adam.buffer import SparseGradBuffer


qdtype = torch.uint8

class MicroAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3, betas = (0.9, 0.999), eps=1e-8, k=10,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, k=k)
        super(MicroAdam, self).__init__(params, defaults)

    def _Q(self, x: torch.Tensor, delta: float, Delta: float) -> torch.Tensor:
        #Q procedure from pseudocode
        if Delta == delta:
            quantized = torch.zeros_like(x, dtype=qdtype)
        else:
            u_inv = 15/(Delta - delta + 1e-8)
            quantized = ((x - delta)*u_inv).round().clamp(0, 15).to(qdtype)
        return quantized

    def _Q_inv(self, xq: torch.Tensor, delta: float, Delta: float):
        #Q^{-1} procedure from pseudocode
        xq = xq.to(torch.float32)
        u = (Delta - delta + 1e-8)/15
        x = xq*u + delta

        return x

    @staticmethod
    def adam_stats(
        betas: tuple,
        buffer: SparseGradBuffer,
        step: int,
        grad: torch.Tensor
    ):
        zm = torch.zeros_like(grad).view(-1)
        zv = torch.zeros_like(grad).view(-1)
        m = buffer.m
        beta1, beta2 = betas
        for i in range(min(step, m)):
            r = (step - i)%m
            topk_idx, topk_values = buffer[i]
            zm[topk_idx] += beta1**r * topk_values
            zv[topk_idx] += beta2**r * topk_values**2
        zm = zm.view(grad.shape)
        zv = zv.view(grad.shape)
        return (
            zm * (1 - beta1)/(1 - beta1**step),
            zv * (1 - beta2)/(1 - beta2**step)
        )

    def step(self, closure = None):
        #Closure is added for compatibility. Here is not used at all.
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            #Taking parameters from model's buffers
            beta1, beta2 = group['betas']

            for p in group["params"]:
                if p.grad is None:
                    continue

                #Starts optimization step for each param.
                state = self.state[p]
                if len(state) == 0: 
                    #First step. Initialize adam
                    #Line 2 in pseudocode
                    state["step"] = 0
                    state["ef"] = torch.zeros_like(p.grad.data, dtype=qdtype).view(-1) #e_1. Error feedback
                    state["delta"] = torch.tensor(0.0, device=p.device)
                    state["Delta"] = torch.tensor(0.0, device=p.device)
                    state["g_buffer"] = SparseGradBuffer()

                state["step"] += 1

                #No need to write for loop of line 3. Pytorch performs that.

                #Following snippet is Line 4 from pseudo code
                grad = p.grad.data
                flat_grad = grad.view(-1)
                k = grad.numel()//100 + 1

                flat_ef, delta, Delta = state["ef"], state["delta"], state["Delta"]

                #Line 5 from pseudo code
                a = flat_grad + self._Q_inv(flat_ef, delta, Delta)
                
                #Line 6 of pseudocode
                flat_a = a.view(-1)
                _, topk_idx = torch.topk(flat_a.abs(), k)
                topk_values = flat_a[topk_idx]
                state["g_buffer"].push((topk_idx, topk_values))

                #Line 7 of pseudocode
                mask = torch.zeros_like(flat_a, dtype=torch.bool)
                mask[topk_idx] = True
                #flat_a[mask] = 0
                flat_a = torch.where(mask, torch.tensor(0, dtype=flat_a.dtype, device=flat_a.device), flat_a)
                
                #assert p.device == p.grad.device == flat_a.device, "Device mismatch"
                #assert state["delta"].device == p.device, "delta on wrong device"
                #assert state["Delta"].device == p.device, "Delta on wrong device"

                #Line 8 of pseudocode
                delta = flat_a.min()
                Delta = flat_a.max()

                #Line 9 of pseudocode
                ef = self._Q(flat_a, delta, Delta)
                state["ef"] = ef
                state["delta"] = delta
                state["Delta"] = Delta

                #Adam update
                mt, vt = MicroAdam.adam_stats(
                    (beta1, beta2),
                    state["g_buffer"],
                    state["step"],
                    grad
                )
                new_denom = vt.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(mt, new_denom, value=-step_size)

        return loss #Added for compatibility
