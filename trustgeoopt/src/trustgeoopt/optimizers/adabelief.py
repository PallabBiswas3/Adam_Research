import math

import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    """Unrectified Algorithm 2 reference: s=EMA[(g-m_new)^2]+eps.

    Also eps outside sqrt; decoupled weight decay is an explicit extension.
    No AMSGrad/rectification. This is not the authors' full package.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("lr must be finite/nonnegative; eps finite/positive")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be in [0,1)")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("invalid weight decay")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Preflight avoids partial updates when a later tensor has invalid gradients.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and (p.grad.is_sparse or p.is_complex()):
                    raise ValueError("AdaBelief supports real dense gradients only")
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("nonfinite gradient")
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                state["step"] += 1
                t = state["step"]
                m, s = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1-b1)
                residual = g-m
                s.mul_(b2).addcmul_(residual, residual, value=1-b2).add_(group["eps"])
                denom = (s/(1-b2**t)).sqrt().add_(group["eps"])
                p.mul_(1-group["lr"]*group["weight_decay"])
                p.addcdiv_(m, denom, value=-group["lr"]/(1-b1**t))
        return loss
