import torch
from torch.optim import Optimizer
import math #really?

qdtype = torch.uint32

class MicroAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3, betas = (0.9, 0.999), eps=1e-8, k=10
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
        x = (xq*u + delta).view(shape)

        return x

    def sparse_grad(self, grad: torch.Tensor, k: int) -> torch.Tensor:
        #flat_grad = grad.clone().view(-1)
        flat_grad = grad.view(-1)
        # Get indices of top-K largest absolute values
        _, topk_idx = torch.topk(flat_grad.abs(), k)
        # Save original values at top-K positions
        topk_vals = flat_grad[topk_idx]
        # Zero entire gradient
        flat_grad.zero_()
        # Restore top-K values to their original positions
        flat_grad[topk_idx] = topk_vals
        sparse_grad = flat_grad.view(grad.shape)
        return sparse_grad

    
    def _quantize(self, x: torch.Tensor, k: int):
        #here x should be $a$ from pseudocode
        flat = x.view(-1)
        topk_vals, topk_idx = torch.topk(flat.abs(), k) #Line 6 of pseudocode

        #Next snippet looks like line 7 from pseudocode, but not actually.
        #Here we are performing required ops to compute Delta and delta on a copy.
        #So, looks like we are duplicating operations. FUTURE: To optimize.
        mask = torch.zeros_like(flat, dtype=torch.bool, device=flat.device)
        mask[topk_idx] = True
        residual = flat.clone()
        residual[mask] = 0

        delta, Delta = residual.min(), residual.max() #Looks

        e_quantized = self._Q(residual, delta, Delta)

        return(
            topk_idx, #I_t in pseudocode
            flat[topk_idx], #V_t in pseudocode
            e_quantized, #e_t in pseudocode
            delta, #\delta
            Delta #\Delta
        )

    def step(self, closure = None):
        #Closure is added for compatibility. Here is not used at all.
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            #Taking parameters from model's buffers
            lr, betas = group['lr'], group['betas'],
            eps, k = group['eps'], group['k']
            beta1, beta2 = betas

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
                    state["ef"] = torch.zeros_like(p.data) #e_1. Error feedback
                    state["delta"] = torch.tensor(0.0, device=p.device)
                    state["Delta"] = torch.tensor(0.0, device=p.device)
        

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                ef, delta, Delta = state["ef"], state["delta"], state["Delta"]
                state["step"] += 1
    
                grad = p.grad.data
                sparse_grad = self.sparse_grad(grad, grad.numel()//100)

                # Adam update
                exp_avg.mul_(beta1).add_(sparse_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(sparse_grad, sparse_grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        
        return loss #Added for compatibility
