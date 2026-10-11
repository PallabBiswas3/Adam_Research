import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trustgeoopt.training.quadratic_theory import (
    effective_condition, scalar_stationary_loss, second_moment_statistics, stationary_covariance,
)
from trustgeoopt.training.sprint3 import (
    candidate_grid, matched_problem, optimizer_for, preconditioner_matrix,
    quadratic_trial, schedule_multiplier, select_candidates, subspace_metrics,
)


@pytest.mark.parametrize("beta", [0, 0.5, 0.9])
def test_scalar_formula_matches_lyapunov(beta):
    result = stationary_covariance(np.array([[3.0]]), np.array([[2.0]]), np.array([[0.7]]), 0.1, beta)
    assert result["relative_residual"] < 1e-12
    assert result["loss"] == pytest.approx(scalar_stationary_loss(3, 2, 0.7, 0.1, beta))


def test_unstable_recurrence_rejected():
    with pytest.raises(ValueError, match="stable"):
        stationary_covariance(np.eye(2), np.eye(2), np.eye(2), 3)


@pytest.mark.parametrize("noise", ["isotropic", "misaligned", "aligned", "antialigned"])
def test_matched_coordinate_transform(noise):
    a = {"condition": 1000, "dimension": 16, "rotation": "diagonal", "noise": noise}
    h, x, f = matched_problem(a, 30)
    hr, xr, fr = matched_problem({**a, "rotation": "rotated"}, 30)
    generator = torch.Generator().manual_seed(130)
    q = torch.linalg.qr(torch.randn(16, 16, generator=generator, dtype=torch.float64)).Q
    torch.testing.assert_close(hr, q@h@q.T)
    torch.testing.assert_close(xr, q@x)
    torch.testing.assert_close(fr, q@f)
    torch.testing.assert_close(x@h@x, xr@hr@xr)
    assert torch.trace(f@f.T) == pytest.approx(16)


def test_selection_rejects_evaluation_and_incomplete_candidates():
    c = {"regime": "r", "optimizer": {"name": "adamw", "lr": 0.1}, "split": "tuning", "seed": 30}
    records = [{"config": c, "status": "success", "tail_loss": 0.1}]
    assert select_candidates(records, [30, 31], "tail_loss")["r/adamw"] is None
    records.append({"config": {**c, "seed": 31}, "status": "failed"})
    assert select_candidates(records, [30, 31], "tail_loss")["r/adamw"] is None
    with pytest.raises(ValueError, match="evaluation"):
        select_candidates([{**records[0], "config": {**c, "split": "evaluation"}}], [30], "tail_loss")


def test_noise_covariance_is_not_hessian():
    rng = np.random.default_rng(5)
    h = torch.diag(torch.tensor([1., 2., 10., 100.], dtype=torch.float64))
    gradients = [torch.tensor(g, dtype=torch.float64) for g in rng.standard_normal((10000, 4))]
    metrics = subspace_metrics(h, gradients, 1, 8)
    assert metrics["covariance_hessian_cosine"] < 0.7
    assert metrics["oracle_overlap"] == pytest.approx(1)


def test_pinned_reference_hashes_and_float64_projection():
    vendor = Path(__file__).resolve().parents[1]/"src/trustgeoopt/optimizers/vendor"
    provenance = json.loads((vendor/"provenance.json").read_text())
    for name, source in provenance["files"].items():
        assert hashlib.sha256((vendor/name).read_bytes()).hexdigest() == source["sha256"]
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        p = torch.nn.Parameter(torch.arange(16, dtype=torch.float64).reshape(4, 4)/16)
        optimizer = optimizer_for([p], {"name": "soap", "lr": 0.01})
        for _ in range(12):
            p.grad = p.detach().clone()
            optimizer.step()
        b = preconditioner_matrix(optimizer, p)
        torch.testing.assert_close(b, b.T)
        assert torch.linalg.eigvalsh(b).min() > 0
    finally:
        torch.set_default_dtype(previous)


def test_protocol_has_equal_candidates_and_fresh_seeds():
    import yaml
    protocol = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/synthetic/sprint3.yaml").read_text())
    assert set(protocol["development_seeds"]).isdisjoint(protocol["evaluation_seeds"])
    assert not (set(protocol["development_seeds"]+protocol["evaluation_seeds"]) & {10, 11, 20, 21, 22, 23, 24})
    assert {len(candidate_grid(protocol, "stochastic", m)) for m in protocol["stochastic"]["rates"]} == {24}
    p = torch.nn.Parameter(torch.ones(4, 4))
    for method in protocol["stochastic"]["rates"]:
        optimizer_for([p], candidate_grid(protocol, "stochastic", method)[0])
    assert schedule_multiplier("cosine", 99, 100) == pytest.approx(0.01)
    assert second_moment_statistics(2, 0.9)["variance"] == pytest.approx(8*0.1/1.9)


def test_diagonal_scaling_cannot_remove_symmetric_rotation_condition():
    h = np.array([[50.5, 49.5], [49.5, 50.5]])
    for scale in np.logspace(-2, 2, 40):
        assert effective_condition(h, np.diag([scale, 1])) >= 100-1e-8
    assert effective_condition(h, np.linalg.inv(h)) == pytest.approx(1)


def test_budget_censoring_retains_rows(tmp_path):
    import time
    c = {"stage": "geometry", "seed": 40, "objective": {"condition": 10, "rotation": "diagonal"},
         "optimizer": {"name": "adamw", "lr": .01}, "steps": 100, "split": "evaluation",
         "history_window": 8, "rank": 2, "tail_fraction": .2}
    result = quadratic_trial(c, tmp_path, time.perf_counter()-1)
    assert result["status"] == "budget_censored"
    assert result["rows"] == []


def test_short_quadratic_logs_true_objective(tmp_path):
    import time
    import pandas as pd
    c = {"stage": "geometry", "seed": 40, "objective": {"condition": 10, "rotation": "diagonal"},
         "optimizer": {"name": "adamw", "lr": .01}, "steps": 12, "split": "tuning",
         "history_window": 8, "rank": 2, "tail_fraction": .25}
    result = quadratic_trial(c, tmp_path, time.perf_counter()+10)
    rows = pd.read_csv(tmp_path/"metrics.csv")
    assert result["status"] == "success"
    assert len(rows) == 12
    assert result["tail_loss"] == pytest.approx(rows.loss.tail(3).mean())
    assert result["target_censored"] == (result["target"] is None)
    assert result["state_tensor_bytes"] > 0
