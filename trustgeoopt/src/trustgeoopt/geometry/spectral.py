import torch


def random_orthonormal_basis(dimension, rank, generator=None, dtype=torch.float64, device="cpu"):
    if not 1 <= rank <= dimension:
        raise ValueError("rank must be between 1 and dimension")
    return torch.linalg.qr(torch.randn(dimension, rank, generator=generator,
                                      dtype=dtype, device=device), mode="reduced").Q
