import math

import torch

from trustgeoopt.diagnostics.reliability import ReliabilityMonitor
from trustgeoopt.training.metrics import prediction_metrics
from trustgeoopt.optimizers import make_optimizer
from trustgeoopt.diagnostics.gradient_statistics import prospective_adam_state


def test_known_signals_and_zero_gradient():
    monitor = ReliabilityMonitor()
    first = monitor.observe(torch.tensor([2., 0.]), torch.tensor([-1., 0.]), torch.ones(2))
    assert first["scale_mismatch"] == 2
    assert first["momentum_alignment"] == -1
    assert first["flag"] == 1
    zero = monitor.observe(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    assert zero["scale_mismatch"] == 0
    assert math.isnan(zero["momentum_alignment"])


def test_monitor_has_no_effect_on_update():
    p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    a = make_optimizer([p], dict(name="adamw", lr=0.1))
    b = make_optimizer([q], dict(name="adamw_monitor", lr=0.1))
    monitor = ReliabilityMonitor()
    for step in range(10):
        p.grad = p.detach().square()+step
        q.grad = q.detach().square()+step
        monitor.observe(*prospective_adam_state(b))
        a.step()
        b.step()
        assert torch.equal(p, q)


def test_auc_ap_ties_and_single_class():
    result = prediction_metrics([0, 1, 0, 1], [1, 1, 1, 1], threshold=1)
    assert result["roc_auc"] == 0.5
    assert result["average_precision"] == 0.5
    assert prediction_metrics([0, 0], [0, 1], 0.5)["average_precision"] is None
    assert prediction_metrics([1, 1], [0, 1], 0.5)["roc_auc"] is None
