import pytest
import torch

from trustgeoopt.geometry.woodbury import diagonal_low_rank_solve
from trustgeoopt.optimizers import make_optimizer
from trustgeoopt.utils.seeds import set_seed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_baselines_and_geometry():
    set_seed(0)
    for name in ["adamw", "adabelief", "shampoo"]:
        p = torch.nn.Parameter(torch.ones(2, 2, device="cuda"))
        optimizer = make_optimizer([p], dict(name=name, lr=0.01))
        p.square().sum().backward()
        optimizer.step()
        assert torch.isfinite(p).all()
    diagonal = torch.ones(8, device="cuda")
    factor = torch.randn(8, 2, device="cuda")
    rhs = torch.randn(8, device="cuda")
    actual = diagonal_low_rank_solve(diagonal, factor, rhs)
    expected = torch.linalg.solve(torch.diag(diagonal)+factor @ factor.T, rhs)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
