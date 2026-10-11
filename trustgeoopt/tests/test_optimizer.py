import pytest
import torch

from trustgeoopt.optimizers import make_optimizer
from trustgeoopt.optimizers.preconditioners import ReferenceShampoo


def test_adabelief_reference_multistep():
    p = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
    opt = make_optimizer([p], dict(name="adabelief", lr=0.01, betas=(0.5, 0.8), weight_decay=0.1, eps=1e-6))
    reference, m, s = p.detach().clone(), torch.zeros_like(p), torch.zeros_like(p)
    for step, g in enumerate([torch.tensor([2., -3.]), torch.tensor([-1., 1.])], 1):
        p.grad = g.double()
        m = 0.5*m+0.5*p.grad
        s = 0.8*s+0.2*(p.grad-m).square()+1e-6
        reference = reference*(1-0.01*0.1)-0.01*(m/(1-0.5**step))/((s/(1-0.8**step)).sqrt()+1e-6)
        opt.step()
        assert torch.allclose(p, reference, atol=1e-12)


def test_shampoo_dense_diagonal_reference():
    p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    p.grad = torch.diag(torch.tensor([2., 3.], dtype=torch.float64))
    opt = ReferenceShampoo([p], lr=0.1, eps=0.01)
    expected = p.detach()-0.1*torch.diag(torch.tensor([2/(4.01**0.5), 3/(9.01**0.5)], dtype=torch.float64))
    opt.step()
    assert torch.allclose(p, expected, atol=1e-12)


def test_shampoo_dense_kronecker_reference():
    p = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
    opt = ReferenceShampoo([p], lr=0.02, eps=0.1)
    left, right, reference = torch.eye(2, dtype=p.dtype)*0.1, torch.eye(3, dtype=p.dtype)*0.1, p.detach().clone()
    for g in [torch.tensor([[1., 2., 0.], [3., -1., 2.]], dtype=p.dtype), torch.tensor([[0., 1., 2.], [-1., 2., 4.]], dtype=p.dtype)]:
        left, right = left+g @ g.T, right+g.T @ g
        values, basis = torch.linalg.eigh(torch.kron(left, right))
        direction = (basis @ torch.diag(values.pow(-0.25)) @ basis.T @ g.flatten()).reshape_as(p)
        reference -= 0.02*direction
        p.grad = g
        opt.step()
        assert torch.allclose(p, reference, atol=1e-11)


@pytest.mark.parametrize("name", ["sgd", "momentum", "adam", "adamw", "amsgrad", "rmsprop", "adagrad", "radam", "adabelief", "shampoo"])
def test_baseline_quadratic_progress(name):
    p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    opt = make_optimizer([p], dict(name=name, lr=0.01))
    initial = float(p.square().sum())
    for _ in range(20):
        opt.zero_grad()
        p.square().sum().backward()
        opt.step()
    assert torch.isfinite(p).all()
    assert float(p.square().sum()) < initial


def test_gate_and_factor_cap_enforced():
    with pytest.raises(NotImplementedError):
        make_optimizer([torch.nn.Parameter(torch.ones(2))], dict(name="trustgeo"))
    with pytest.raises(ValueError):
        ReferenceShampoo([torch.nn.Parameter(torch.ones(2, 129))])
