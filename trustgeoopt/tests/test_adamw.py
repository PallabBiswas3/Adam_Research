import pytest
import torch

from trustgeoopt.optimizers.optimizer_factory import make_optimizer
from trustgeoopt.diagnostics.gradient_statistics import prospective_adam_state


@pytest.mark.parametrize("amsgrad", [False, True])
def test_adamw_against_scalar_equation(amsgrad):
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = make_optimizer([p], dict(name="adamw", lr=0.03, betas=(0.6, 0.8), eps=1e-5,
                                        weight_decay=0.2, amsgrad=amsgrad))
    reference, m, v, vmax = p.detach().clone(), torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)
    for step, g in enumerate([torch.tensor([2., -1.]), torch.tensor([0.1, 4.]), torch.tensor([-1., 0.])], 1):
        p.grad = g.double()
        m = 0.6*m+0.4*p.grad
        v = 0.8*v+0.2*p.grad.square()
        vmax = torch.maximum(vmax, v)
        vhat = (vmax if amsgrad else v)/(1-0.8**step)
        prospective = prospective_adam_state(optimizer)
        assert torch.allclose(prospective[1], m/(1-0.6**step))
        assert torch.allclose(prospective[2], vhat)
        reference = reference*(1-0.03*0.2)-0.03*(m/(1-0.6**step))/(vhat.sqrt()+1e-5)
        optimizer.step()
        assert torch.allclose(p, reference, atol=1e-12, rtol=1e-12)
