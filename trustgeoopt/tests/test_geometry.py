import pytest
import torch

from trustgeoopt.geometry.woodbury import diagonal_low_rank_solve
from trustgeoopt.geometry.low_rank import GradientHistory
from trustgeoopt.diagnostics.curvature import hessian_vector_product


@pytest.mark.parametrize("rank", [0, 1, 4])
def test_woodbury_matches_dense(rank):
    rng = torch.Generator().manual_seed(3)
    diagonal = torch.rand(8, generator=rng, dtype=torch.float64)+0.2
    factor = torch.randn(8, rank, generator=rng, dtype=torch.float64)
    if rank > 1:
        factor[:, -1] = factor[:, 0]  # dependent columns are legal
    rhs = torch.randn(8, generator=rng, dtype=torch.float64)
    actual = diagonal_low_rank_solve(diagonal, factor, rhs, damping=0.1)
    expected = torch.linalg.solve(torch.diag(diagonal+0.1)+factor @ factor.T, rhs)
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_history_moment_and_covariance():
    samples = torch.tensor([[1., 2., 3.], [2., 1., 4.]], dtype=torch.float64)
    for centered in [False, True]:
        history = GradientHistory(rank=2, window=3, centered=centered)
        for column in samples.T:
            history.update(column)
        factor, tail = history.factor()
        matrix = samples-samples.mean(1, keepdim=True) if centered else samples
        assert torch.allclose(factor @ factor.T, matrix @ matrix.T/3, atol=1e-12)
        assert tail == 0
        history.update(samples[:, 0])
        assert history.history.shape == (2, 3)


def test_hvp_finite_difference_nonlinear():
    x = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    vector = torch.tensor([1., 0.5], dtype=torch.float64)
    function = lambda z: (z**4).sum()+z[0]*z[1]
    actual = hessian_vector_product(function(x), [x], [vector])[0]
    def gradient(z):
        z = z.detach().requires_grad_()
        return torch.autograd.grad(function(z), z)[0]
    expected = (gradient(x+1e-5*vector)-gradient(x-1e-5*vector))/(2e-5)
    assert torch.allclose(actual, expected, atol=1e-8)


def test_woodbury_invalid():
    with pytest.raises(ValueError):
        diagonal_low_rank_solve(torch.zeros(2), torch.zeros(2, 1), torch.ones(2))
