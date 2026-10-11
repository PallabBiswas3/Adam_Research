import pytest
import torch

from trustgeoopt.optimizers import make_optimizer
from trustgeoopt.training.checkpoint import save_checkpoint, load_checkpoint
from trustgeoopt.utils.seeds import set_seed


@pytest.mark.parametrize("name", ["adamw", "adabelief", "shampoo"])
def test_optimizer_checkpoint_continuation(tmp_path, name):
    set_seed(10)
    model = torch.nn.Linear(2, 2).double()
    opt = make_optimizer(model.parameters(), dict(name=name, lr=0.01))
    def step(target, optimizer):
        optimizer.zero_grad()
        target(torch.randn(3, 2, dtype=torch.float64)).square().sum().backward()
        optimizer.step()
    step(model, opt)
    path = tmp_path/"checkpoint.pt"
    save_checkpoint(path, model, opt, 0)
    step(model, opt)
    expected = [p.detach().clone() for p in model.parameters()]
    other = torch.nn.Linear(2, 2).double()
    other_opt = make_optimizer(other.parameters(), dict(name=name, lr=0.01))
    load_checkpoint(path, other, other_opt)
    step(other, other_opt)
    for actual, reference in zip(other.parameters(), expected):
        assert torch.equal(actual, reference)
