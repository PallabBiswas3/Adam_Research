import math

import torch


def diagonal_low_rank_solve(diagonal, factor, rhs, damping=0.0):
    """Solve (diag(diagonal+damping)+factor@factor.T)x=rhs in O(pk^2+k^3).

    factor p by k may have dependent/zero columns. No p by p matrix allocation.
    """
    if diagonal.ndim != 1 or rhs.shape != diagonal.shape or factor.ndim != 2:
        raise ValueError("expected diagonal/rhs p-vectors and factor p by k")
    if factor.shape[0] != diagonal.numel() or damping < 0 or not math.isfinite(damping):
        raise ValueError("invalid factor shape or damping")
    if any(t.device != diagonal.device or t.dtype != diagonal.dtype for t in (rhs, factor)):
        raise ValueError("all inputs must share device and dtype")
    if not all(torch.isfinite(t).all() for t in (diagonal, factor, rhs)):
        raise ValueError("finite inputs required")
    base = diagonal+damping
    if (base <= 0).any():
        raise ValueError("positive diagonal+damping required")
    inverse_rhs = rhs/base
    if factor.shape[1] == 0:
        return inverse_rhs
    scaled = factor/base[:, None]
    inner = torch.eye(factor.shape[1], dtype=factor.dtype, device=factor.device)+factor.T @ scaled
    chol = torch.linalg.cholesky((inner+inner.T)/2)
    correction = torch.cholesky_solve((factor.T @ inverse_rhs)[:, None], chol).squeeze(1)
    result = inverse_rhs-scaled @ correction
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite Woodbury result")
    return result
