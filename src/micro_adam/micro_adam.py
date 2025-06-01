import torch
import math


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

    def _Q_inv(self, xq: torch.Tensor, delta: float, Delta: float, shape: tuple):
        #Q^{-1} procedure from pseudocode
        xq = xq.to(torch.float32)
        u = (Delta - delta + 1e-8)/15
        x = xq*u + delta

        return x

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
                    state["exp_avg"] = torch.zeros_like(p.data) #m_0
                    state["exp_avg_sq"] = torch.zeros_like(p.data) #v_0
                    state["ef"] = torch.zeros_like(p.grad.data, dtype=qdtype).view(-1) #e_1. Error feedback
                    state["delta"] = torch.tensor(0.0, device=p.device)
                    state["Delta"] = torch.tensor(0.0, device=p.device)

                state["step"] += 1

                #No need to write for loop of line 3. Pytorch performs that.

                #Following snippet is Line 4 from pseudo code
                grad = p.grad.data
                shape = grad.shape
                flat_grad = grad.view(-1)
                k = grad.numel()//100 + 1

                flat_ef, delta, Delta = state["ef"], state["delta"], state["Delta"]

                #Line 5 from pseudo code
                a = flat_grad + self._Q_inv(flat_ef, delta, Delta, shape)
                
                #Line 6 of pseudocode
                flat_a = a.view(-1)
                _, topk_idx = torch.topk(flat_a.abs(), k)
                topk_values = flat_a[topk_idx]
                flat_topk_grad = torch.zeros_like(flat_a)
                flat_topk_grad[topk_idx] = topk_values
                topk_grad = flat_topk_grad.view(shape)

                #Line 7 of pseudocode
                mask = torch.zeros_like(
                    flat_a,
                    dtype=torch.bool,
                    device=flat_a.device
                )
                mask[topk_idx] = True
                flat_a[topk_idx] = 0
                
                #Line 8 of pseudocode
                delta = flat_a.min()
                Delta = flat_a.max()

                #Line 9 of pseudocode
                ef = self._Q(flat_a, delta, Delta)
                state["ef"] = ef
                state["delta"] = delta
                state["Delta"] = Delta

                #Adam update
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(topk_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(topk_grad, topk_grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss #Added for compatibility
