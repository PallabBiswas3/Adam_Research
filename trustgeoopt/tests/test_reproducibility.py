import copy
import json

import pandas as pd
import pytest
import torch

from trustgeoopt.datasets.synthetic import SyntheticObjective
from trustgeoopt.training.synthetic_runner import run_synthetic
from trustgeoopt.training.trainer import train
from trustgeoopt.models import make_model


def config():
    return dict(name="test", seed=3, device="cpu", steps=8, checkpoint_interval=4,
                objective=dict(kind="noisy", dimension=4, condition=10, noise_std=0.1),
                optimizer=dict(name="adamw", lr=0.03))


def test_same_seed_and_runner_resume(tmp_path):
    cfg = config()
    a = run_synthetic(cfg, tmp_path)
    b = run_synthetic(cfg, tmp_path)
    frame_a, frame_b = pd.read_csv(a/"metrics.csv"), pd.read_csv(b/"metrics.csv")
    columns = [col for col in frame_a if not col.endswith("seconds")]
    pd.testing.assert_frame_equal(frame_a[columns], frame_b[columns])
    short = copy.deepcopy(cfg)
    short["steps"] = 4
    prefix = run_synthetic(short, tmp_path)
    resumed = run_synthetic(cfg, tmp_path, prefix/"checkpoint.pt")
    continuation = pd.read_csv(resumed/"metrics.csv")
    pd.testing.assert_frame_equal(frame_a.iloc[4:][columns].reset_index(drop=True), continuation[columns])


@pytest.mark.parametrize("kind", ["quadratic", "rotated", "rosenbrock", "nonstationary", "noisy"])
def test_objective_gradient_finite_difference(kind):
    obj = SyntheticObjective(dict(kind=kind, dimension=4, condition=10, change_step=2), 1)
    x = torch.tensor([0.2, 0.1, -0.4, 0.3], dtype=torch.float64)
    grad = obj.gradient(x, 3)
    for i in range(4):
        perturbation = torch.zeros_like(x)
        perturbation[i] = 1e-6
        approx = (obj.loss(x+perturbation, 3)-obj.loss(x-perturbation, 3))/2e-6
        assert torch.allclose(grad[i], approx, atol=1e-6, rtol=1e-6)


def test_nonstationary_labels_same_objective(tmp_path):
    cfg = config()
    cfg["objective"].update(kind="nonstationary", change_step=4, optimum_shift=10)
    result = run_synthetic(cfg, tmp_path)
    frame = pd.read_csv(result/"metrics.csv")
    assert frame.relative_error.isna().all()
    assert (frame.harmful == (frame.objective_after-frame.objective_before > 1e-12*frame.objective_before.abs().clip(lower=1))).all()
    assert json.loads((result/"metadata.json").read_text())["summary"]["tracking"] is True


def test_trainer_resume_and_no_test_leakage(tmp_path):
    cfg = dict(name="trainer-test", seed=1, epochs=2, batch_size=16, device="cpu",
        dataset=dict(name="synthetic", dimension=4, classes=2, train_size=32, validation_size=16, test_size=16),
        model=dict(name="mlp", hidden=[8]), optimizer=dict(name="adamw", lr=0.01))
    full = train(cfg, tmp_path)
    prefix_cfg = copy.deepcopy(cfg)
    prefix_cfg["epochs"] = 1
    prefix = train(prefix_cfg, tmp_path)
    resumed = train(cfg, tmp_path, prefix/"checkpoint.pt")
    full_state = torch.load(full/"checkpoint.pt", weights_only=True)["model"]
    resumed_state = torch.load(resumed/"checkpoint.pt", weights_only=True)["model"]
    assert all(torch.equal(full_state[key], resumed_state[key]) for key in full_state)
    meta = json.loads((full/"metadata.json").read_text())
    assert meta["test_evaluated"] is False
    assert meta["summary"]["test"] is None


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_image_model_forward_backward(name):
    model = make_model(dict(name=name, channels=3), dict(classes=10))
    prediction = model(torch.randn(2, 3, 32, 32))
    assert prediction.shape == (2, 10)
    prediction.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
