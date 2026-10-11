import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from trustgeoopt.diagnostics.gradient_statistics import prospective_adamw_displacement
from trustgeoopt.training.reliability_evaluation import lead_labels, fit_logistic, predict_logistic, paired_cluster_interval
from trustgeoopt.training.metrics import prediction_metrics
from trustgeoopt.training.sprint2 import published_counterexample, select_rates, run_trial, trial_config
from trustgeoopt.optimizers import make_optimizer


@pytest.mark.parametrize("amsgrad,maximize", [(False, False), (True, False), (False, True)])
def test_proposal_matches_actual_multiple_groups(amsgrad, maximize):
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))
    opt = torch.optim.AdamW([dict(params=[p], lr=.03, eps=1e-7, weight_decay=.2),
                            dict(params=[q], lr=.1, eps=1e-5, weight_decay=.01)], amsgrad=amsgrad, maximize=maximize)
    for i in range(10):
        p.grad = torch.tensor([(-1.)**i, .4+i], dtype=torch.float64)
        q.grad = torch.tensor([i+1.], dtype=torch.float64)
        before = torch.cat([p.detach(), q.detach()]).clone()
        state_before = copy.deepcopy(opt.state_dict())
        proposed = prospective_adamw_displacement(opt)
        for old, new in zip(state_before["state"].values(), opt.state_dict()["state"].values()):
            for key in old:
                assert torch.equal(old[key], new[key])
        opt.step()
        torch.testing.assert_close(torch.cat([p, q])-before, proposed, rtol=1e-12, atol=1e-15)


def test_lead_does_not_cross_transition_or_run_end():
    frame = pd.DataFrame(dict(step=range(7), harmful=[0, 1, 0, 1, 0, 0, 1]))
    result = lead_labels(frame, 3, transitions=[4])
    assert result.step.tolist() == [0]
    assert result.label.tolist() == [1]


def test_cluster_bootstrap_retains_seed_effect_and_pairing():
    frame = pd.DataFrame(dict(seed=[1]*20+[2]*20, label=[0, 1]*20,
                              first=[0., 1.]*20, second=[1., 0.]*20))
    interval = paired_cluster_interval(frame, "first", "second", 20)
    assert interval["seed_clusters"] == 2
    assert interval["estimate"] > .4
    assert interval["ci95"][0] == interval["ci95"][1]
    reverse = paired_cluster_interval(frame, "second", "first", 20)
    assert interval["estimate"] == -reverse["estimate"]


def test_weighted_bootstrap_matches_explicit_whole_seed_duplication():
    frame = pd.DataFrame(dict(seed=[1, 1, 2, 2, 3, 3], label=[0, 1, 1, 0, 0, 1],
                              first=[.2, .7, .3, .3, .8, .9], second=[.1, .4, .6, .2, .2, .3]))
    result = paired_cluster_interval(frame, "first", "second", 100, 4)
    rng = np.random.default_rng(4)
    groups = [g for _, g in frame.groupby("seed")]
    values = []
    for _ in range(100):
        sample = pd.concat([groups[i] for i in rng.integers(3, size=3)])
        a = prediction_metrics(sample.label, sample["first"], .5)["average_precision"]
        b = prediction_metrics(sample.label, sample["second"], .5)["average_precision"]
        values.append(a-b)
    np.testing.assert_allclose(result["ci95"], np.quantile(values, [.025, .975]))


def test_fit_uses_development_scaling_and_rejects_single_class():
    x = np.arange(-10, 10)[:, None]
    model = fit_logistic(x, (x[:, 0] > 0).astype(int))
    saved_mean = model["mean"].copy()
    scores = predict_logistic(model, np.array([[-1000], [1000]]))
    assert scores[0] < .01 and scores[1] > .99
    assert model["mean"] == saved_mean
    with pytest.raises(ValueError):
        fit_logistic(x, np.zeros(20))


def test_selection_disqualifies_failed_rate_and_forbids_eval():
    records = [dict(config=dict(split="tuning", regime="r", optimizer=dict(name="adamw", lr=rate)),
                    status=status, tail_loss=loss) for rate, status, loss in [(1, "failed", 0), (.1, "success", 2), (.1, "success", 3)]]
    assert select_rates(records) == {"r/adamw": .1}
    records[0]["config"]["split"] = "evaluation"
    with pytest.raises(ValueError):
        select_rates(records)


def test_incomplete_development_candidate_cannot_win_selection():
    record = dict(config=dict(split="tuning", regime="r", seed=10, optimizer=dict(name="adamw", lr=.1)),
                  status="success", tail_loss=0.0)
    assert select_rates([record], [10, 11]) == {"r/adamw": None}


def test_published_failure_cycle_and_regret_bound():
    with open("configs/synthetic/sprint2.yaml") as handle:
        config = yaml.safe_load(handle)["published_counterexample"]
    config["steps"] = 300
    frame = pd.DataFrame(published_counterexample(config))
    adam = frame[frame.method == "adam_paper"]
    assert (adam.maximum_cycle_start_error == 0).all()
    assert adam.minimum_preupdate_x.min() > 0
    assert adam.average_regret.min() >= (2*config["C"]-4)/3
    config["bias_correction"] = True
    with pytest.raises(ValueError):
        published_counterexample(config)


def test_sprint2_quadratic_oracle_and_immutable_attempts(tmp_path):
    protocol = dict(parameter_shape=[2, 2], tail_fraction=.2, target_relative_error=.01)
    c = trial_config(protocol, "toy", dict(kind="rotated", dimension=4, condition=10), "adamw", .1, 12, "reliability_development", 20, True)
    a = run_trial(c, tmp_path, {})
    b = run_trial(c, tmp_path, {})
    assert a["run_id"] != b["run_id"]
    frame = pd.read_csv(tmp_path/"raw"/a["run_id"]/"metrics.csv")
    other = pd.read_csv(tmp_path/"raw"/b["run_id"]/"metrics.csv")
    pd.testing.assert_frame_equal(frame, other)
    np.testing.assert_allclose(frame.loss_increase, frame.oracle_quadratic_change, atol=1e-13)
    assert frame.proposal_error.max() < 1e-14
    assert a["tail_loss"] is not None
    assert json.loads((tmp_path/"raw"/a["run_id"]/"metadata.json").read_text())["status"] == "success"


def test_step_clip_limits_displacement_without_changing_moments():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    a = make_optimizer([p], dict(name="adamw_step_clip", lr=.3, step_cap=.1))
    b = make_optimizer([q], dict(name="adamw", lr=.3))
    p.grad = q.grad = torch.ones_like(p)
    before = p.detach().clone()
    a.step()
    b.step()
    assert abs(float((p-before).norm())-.1) < 1e-14
    for key in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(a.state[p][key], b.state[q][key])
    assert a.correction_events == 1


def test_momentum_reset_turns_uphill_observed_direction_downhill():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    a = make_optimizer([p], dict(name="adamw_momentum_reset", lr=.1))
    p.grad = -torch.ones_like(p)
    for _ in range(10):
        a.step()
    p.grad = torch.ones_like(p)
    assert torch.dot(p.grad, prospective_adamw_displacement(a)) > 0
    before = p.detach().clone()
    a.step()
    assert torch.dot(p.grad, p-before) < 0
    assert a.correction_events == 1
    assert int(a.state[p]["step"]) == 11


@pytest.mark.parametrize("name", ["adamw_step_clip", "adamw_momentum_reset", "adamw_lr_safeguard"])
def test_correction_state_preserves_continuation(name):
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    a = make_optimizer([p], dict(name=name, lr=.1))
    for _ in range(5):
        p.grad = torch.tensor([.5, -.7], dtype=torch.float64)
        a.step()
    q = torch.nn.Parameter(p.detach().clone())
    b = make_optimizer([q], dict(name=name, lr=.1))
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    for _ in range(5):
        p.grad = q.grad = torch.tensor([-.7, .9], dtype=torch.float64)
        a.step()
        b.step()
        assert torch.equal(p, q)
        assert a.correction_events == b.correction_events
        assert a.param_groups[0]["lr"] == b.param_groups[0]["lr"] == .1
