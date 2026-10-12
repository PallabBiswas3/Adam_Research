"""Independent implementations of established equations; no upstream code reuse.

These are toy mechanism adaptations, NOT published benchmark reproductions.
Selectors take observed labels and predictions only; no oracle argument exists.
"""
import math

import torch
import torch.nn.functional as F


def small_loss_mask(losses, observed_y, keep_fraction, classwise=False):
    if losses.ndim != 1 or observed_y.shape != losses.shape or not 0 < keep_fraction <= 1:
        raise ValueError("valid losses/labels/fraction required")
    keep = torch.zeros_like(losses, dtype=torch.bool)
    groups = [torch.arange(len(losses), device=losses.device)]
    if classwise:
        groups = [(observed_y == c).nonzero().flatten() for c in torch.unique(observed_y)]
    for indices in groups:
        count = max(1, math.floor(keep_fraction*len(indices)))
        selected = indices[torch.argsort(losses[indices], stable=True)[:count]]
        keep[selected] = True
    return keep


def coteaching_losses(logits1, logits2, observed_y, forget_rate):
    """Algorithm 1 peer-exchanged mean CE, not official code's extra 1/k factor."""
    if not 0 <= forget_rate < 1:
        raise ValueError("forget rate in [0,1) required")
    loss1, loss2 = [F.cross_entropy(p, observed_y, reduction="none") for p in (logits1, logits2)]
    count = max(1, int((1-forget_rate)*len(observed_y)))
    own1 = torch.argsort(loss1.detach(), stable=True)[:count]
    own2 = torch.argsort(loss2.detach(), stable=True)[:count]
    return loss1[own2].mean(), loss2[own1].mean(), own1, own2


def confidence_gap_z(probability_history, observed_y):
    """AAAI 2025 Eqs.9-11, unadjusted-ties MK convention from its protocol.

    The heuristic trend statistic is NOT a calibrated p-value for dependent
    neural training histories. This small implementation is independently
    derived from the paper; official source is inspection-only.
    """
    if probability_history.ndim != 3 or probability_history.shape[1] != len(observed_y):
        raise ValueError("T by N by C predictions required")
    n, samples, classes = probability_history.shape
    if n < 2:
        return torch.zeros(samples, device=probability_history.device)
    observed = probability_history.gather(2, observed_y[None, :, None].expand(n, samples, 1))
    gaps = observed-probability_history
    statistic = torch.zeros_like(gaps[0])
    for t in range(1, n):
        statistic += torch.sign(gaps[t]-gaps[:t]).sum(0)
    variance = n*(n-1)*(2*n+5)/18
    z = torch.where(statistic > 0, statistic-1,
                    torch.where(statistic < 0, statistic+1, statistic))/math.sqrt(variance)
    z.scatter_(1, observed_y[:, None], float("inf"))
    return z.min(1).values


def classwise_scores(losses, observed_y):
    """Rank within observed class; used for post-hoc operating curves only."""
    scores = torch.zeros_like(losses)
    for c in torch.unique(observed_y):
        indices = (observed_y == c).nonzero().flatten()
        values = losses[indices]
        # Equal losses retain equal rank; do not hide tie-based arbitrary selection.
        for index in indices:
            scores[index] = (values >= losses[index]).float().mean()
    return scores
