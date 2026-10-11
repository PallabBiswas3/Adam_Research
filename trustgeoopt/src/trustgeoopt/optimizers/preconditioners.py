import math

import torch
from torch.optim import Optimizer


def symmetric_power(matrix, power, floor):
    """Eigenvalue power of a small SPD factor, with documented numerical floor."""
    values, vectors = torch.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*values.clamp_min(floor).pow(power)) @ vectors.T


class ReferenceShampoo(Optimizer):
    """Small matrix Shampoo, Gupta et al. (2018), Algorithm 1.

    Tensor >=2D is unfolded as first axis by remaining axes. Vectors use diagonal
    AdaGrad, explicitly a hybrid fallback. No grafting/momentum; cumulative factors.
    Max factor dimension is enforced: use a maintained implementation for big models.
    """

    def __init__(self, params, lr=0.01, eps=1e-6, weight_decay=0.0,
                 refresh_interval=1, max_factor_dim=128):
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("invalid lr/eps")
        if refresh_interval < 1 or max_factor_dim < 1:
            raise ValueError("invalid refresh/factor limit")
        if weight_decay < 0 or not math.isfinite(weight_decay):
            raise ValueError("invalid weight decay")
        super().__init__(params, dict(lr=lr, eps=eps, weight_decay=weight_decay,
                                     refresh_interval=refresh_interval, max_factor_dim=max_factor_dim))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim >= 2 and max(p.shape[0], p.numel()//p.shape[0]) > max_factor_dim:
                    raise ValueError("Shampoo factor too large; no silent diagonal fallback")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and (p.grad.is_sparse or p.is_complex()):
                    raise ValueError("real dense gradients required")
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("nonfinite gradient")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state, g = self.state[p], p.grad
                if not state:
                    state["step"] = 0
                    if p.ndim < 2:
                        state["sum_sq"] = torch.full_like(p, group["eps"])
                    else:
                        a, b = p.shape[0], p.numel()//p.shape[0]
                        state["left"] = torch.eye(a, dtype=p.dtype, device=p.device)*group["eps"]
                        state["right"] = torch.eye(b, dtype=p.dtype, device=p.device)*group["eps"]
                state["step"] += 1
                if p.ndim < 2:
                    state["sum_sq"].addcmul_(g, g)
                    direction = g/state["sum_sq"].sqrt()
                else:
                    gm = g.reshape(p.shape[0], -1)
                    state["left"].add_(gm @ gm.T)
                    state["right"].add_(gm.T @ gm)
                    if state["step"] == 1 or (state["step"]-1) % group["refresh_interval"] == 0:
                        state["left_inv"] = symmetric_power(state["left"], -0.25, group["eps"])
                        state["right_inv"] = symmetric_power(state["right"], -0.25, group["eps"])
                    direction = (state["left_inv"] @ gm @ state["right_inv"]).reshape_as(p)
                p.mul_(1-group["lr"]*group["weight_decay"])
                p.add_(direction, alpha=-group["lr"])
        return loss
