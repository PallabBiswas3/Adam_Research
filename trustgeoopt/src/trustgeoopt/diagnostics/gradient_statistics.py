import torch


def cosine(x, y, eps=1e-12):
    if x.shape != y.shape:
        raise ValueError("cosine requires matching shapes")
    denom = x.norm()*y.norm()
    if denom <= eps:
        return float("nan")  # zero direction has no angle
    return float(torch.dot(x.flatten(), y.flatten())/denom)


@torch.no_grad()
def prospective_adam_state(optimizer):
    """Predict current moments without mutating official optimizer state.

    Returns flat grad/mhat/vhat in group order; supported for torch Adam/AdamW
    (including AMSGrad). Includes coupled L2 for Adam, not for AdamW.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return None
    gs, ms, vs = [], [], []
    for group in optimizer.param_groups:
        b1, b2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.detach()
            if group.get("maximize", False):
                g = -g
            if isinstance(optimizer, torch.optim.Adam) and group["weight_decay"]:
                g = g + group["weight_decay"]*p.detach()
            state = optimizer.state.get(p, {})
            step = int(state.get("step", 0))+1
            m = state.get("exp_avg", torch.zeros_like(p))*b1+(1-b1)*g
            v = state.get("exp_avg_sq", torch.zeros_like(p))*b2+(1-b2)*g.square()
            if group.get("amsgrad", False):
                v = torch.maximum(v, state.get("max_exp_avg_sq", torch.zeros_like(p)))
            gs.append(g.flatten())
            ms.append((m/(1-b1**step)).flatten())
            vs.append((v/(1-b2**step)).flatten())
    if not gs:
        return None
    return torch.cat(gs), torch.cat(ms), torch.cat(vs)


@torch.no_grad()
def prospective_adamw_displacement(optimizer):
    """Causal flat displacement, including per-group lr, epsilon and decay.

    Ordinary dense, real AdamW only. Does not initialize or mutate optimizer state.
    This is a research observer, not an additional optimizer step.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("AdamW required")
    state = prospective_adam_state(optimizer)
    if state is None:
        return None
    _, moment, variance = state
    offset, pieces = 0, []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.is_complex() or p.grad.is_sparse:
                raise ValueError("real dense parameters required")
            stop = offset+p.numel()
            direction = moment[offset:stop]/(variance[offset:stop].sqrt()+group["eps"])
            pieces.append(-group["lr"]*(direction+group["weight_decay"]*p.detach().flatten()))
            offset = stop
    return torch.cat(pieces)
