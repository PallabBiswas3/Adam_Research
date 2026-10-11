import torch


def history_factor(history, rank, centered=False):
    """history p by h. Return F with FF^T = truncated sample moment/covariance.

    Centered uses population denominator h, explicitly not unbiased h-1.
    """
    if history.ndim != 2 or history.shape[1] == 0 or rank < 1:
        raise ValueError("nonempty p by h history and positive rank required")
    if not torch.isfinite(history).all():
        raise ValueError("finite history required")
    matrix = history-history.mean(dim=1, keepdim=True) if centered else history
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    k = min(rank, s.numel())
    return u[:, :k]*(s[:k]/math_sqrt(history.shape[1])), float(s[k:].square().sum()/history.shape[1])


def math_sqrt(value):
    return value**0.5
