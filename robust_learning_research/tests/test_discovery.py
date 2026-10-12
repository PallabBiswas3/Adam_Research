"""Tests specified before the controlled campaign implementation."""
import numpy as np
import pytest
import torch

from robust_learning.data import make_problem
from robust_learning.baselines import coteaching_losses, confidence_gap_z, small_loss_mask
from robust_learning.metrics import classification_metrics, operating_curve


@pytest.mark.parametrize("kind", ["symmetric", "class_conditional", "instance"])
def test_frozen_corruption_and_separate_rngs(kind):
    a = make_problem(100, 200, noise_kind=kind)
    b = make_problem(100, 200, noise_kind=kind)
    np.testing.assert_array_equal(a.train.x, b.train.x)
    np.testing.assert_array_equal(a.train.observed_y, b.train.observed_y)
    assert np.array_equal(a.train.observed_y != a.oracle.true_y, a.oracle.flipped)
    changed = make_problem(100, 201, noise_kind=kind)
    np.testing.assert_array_equal(a.train.x, changed.train.x)
    assert not np.array_equal(a.train.observed_y, changed.train.observed_y)
    assert a.train.observed_y.max() < 3


def test_zero_and_full_corruption():
    clean = make_problem(101, 200, noise_rate=0)
    assert not clean.oracle.flipped.any()
    noisy = make_problem(101, 200, noise_kind="symmetric", noise_rate=1)
    assert noisy.oracle.flipped.all()
    assert not hasattr(noisy.train, "oracle")
    assert not hasattr(noisy.train, "true_y")
    assert not hasattr(noisy.train, "flipped")
    with pytest.raises(ValueError):
        make_problem(0, 0, noise_kind="invented")


def test_class_conditional_targets_and_split_ids():
    data = make_problem(103, 400, noise_kind="class_conditional", noise_rate=1)
    np.testing.assert_array_equal(data.train.observed_y, (data.oracle.true_y+1)%3)
    assert not set(data.train.ids)&set(data.validation.ids)
    assert not set(data.train.ids)&set(data.test.ids)
    assert not set(data.validation.ids)&set(data.test.ids)
    assert data.validation.y.shape == (360,)
    assert data.test.y.shape == (360,)
    # Changing noise never changes clean validation/test examples or labels.
    other = make_problem(103, 499, noise_kind="instance")
    np.testing.assert_array_equal(data.validation.x, other.validation.x)
    np.testing.assert_array_equal(data.test.y, other.test.y)


def test_bayes_difficulty_is_predeclared_and_readonly():
    data = make_problem(107, 200)
    assert np.all((data.oracle.ambiguity >= 0)&(data.oracle.ambiguity <= 1))
    assert np.isfinite(data.oracle.ambiguity).all()
    with pytest.raises(ValueError):
        data.oracle.true_y[0] = 2


def test_cross_selection_and_mean_normalization():
    # Networks prefer different observed examples. Each receives its peer's set.
    first = torch.tensor([[4., 0.], [0., 4.], [2., 0.], [0., 2.]], requires_grad=True)
    second = torch.tensor([[0., 4.], [4., 0.], [0., 2.], [2., 0.]], requires_grad=True)
    y = torch.zeros(4, dtype=torch.long)
    l1, l2, own1, own2 = coteaching_losses(first, second, y, .5)
    assert set(own1.tolist()) == {0, 2}
    assert set(own2.tolist()) == {1, 3}
    expected1 = torch.nn.functional.cross_entropy(first[own2], y[own2])
    expected2 = torch.nn.functional.cross_entropy(second[own1], y[own1])
    assert torch.allclose(l1, expected1)
    assert torch.allclose(l2, expected2)
    l1.backward()
    assert torch.equal(first.grad[own1], torch.zeros_like(first.grad[own1]))


def test_mann_kendall_confidence_gap_agrees_with_direct_equations():
    series = torch.tensor([[[.2, .5, .3]], [[.3, .4, .3]], [[.4, .3, .3]], [[.5, .25, .25]], [[.6, .2, .2]]])
    z = confidence_gap_z(series, torch.tensor([0]))
    variance = 5*4*(2*5+5)/18
    assert torch.allclose(z, torch.tensor([(10-1)/variance**.5]))
    assert confidence_gap_z(series.flip(0), torch.tensor([0]))[0] < 0
    assert confidence_gap_z(torch.ones_like(series)/3, torch.tensor([0]))[0] == 0


def test_controls_and_exact_metric_values():
    losses = torch.tensor([1., 4., 2., 3.])
    labels = torch.tensor([0, 0, 1, 1])
    assert small_loss_mask(losses, labels, .5, False).tolist() == [True, False, True, False]
    assert small_loss_mask(losses, labels, .5, True).tolist() == [True, False, True, False]
    metrics = classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]), np.eye(3))
    assert metrics["macro_f1"] == 1
    assert metrics["balanced_accuracy"] == 1
    assert metrics["ece"] == 0


def test_curve_ties_and_empty_strata_are_explicit():
    data = make_problem(108, 200, noise_rate=0, counts=(9, 9, 9))
    curve = operating_curve(np.ones(27), data.oracle, [.1, .5, 1.])
    # Equal scores stay together; index tie breaking must not invent a PR curve.
    assert all(r["accepted"] == 27 for r in curve)
    assert all(r["rejected_noise_precision"] is None for r in curve)


@pytest.mark.parametrize("method", ["ce", "global_small_loss", "classwise_small_loss", "coteaching_toy", "confidence_tracking_toy"])
def test_training_determinism_and_oracle_isolation(method):
    import dataclasses
    import inspect
    import time
    from robust_learning.data import Oracle
    from robust_learning.training import run_training
    data = make_problem(109, 201, counts=(12, 9, 6))
    alternative = dataclasses.replace(data, oracle=Oracle(
        true_y=(data.oracle.true_y+1)%3, flipped=~data.oracle.flipped,
        ambiguity=1-data.oracle.ambiguity, flip_probability=1-data.oracle.flip_probability))
    config = dict(seed=500, hidden=8, method=method, lr=.003, epochs=5,
                  batch_size=9, warmup=1, keep_fraction=.7,
                  nominal_noise_rate=.3, confidence_z_threshold=1.6448536)
    # An incompatible oracle must have no route into training or clean validation.
    a, model_a = run_training(data.train, data.validation, config, time.perf_counter()+20)
    b, model_b = run_training(alternative.train, alternative.validation, config, time.perf_counter()+20)
    assert a["status"] == b["status"] == "success"
    for name, value in model_a.state_dict().items():
        assert torch.equal(value, model_b.state_dict()[name])
    np.testing.assert_array_equal(a["scores"], b["scores"])
    np.testing.assert_array_equal(a["actual_keep"], b["actual_keep"])
    assert "oracle" not in inspect.signature(run_training).parameters


def test_matched_precision_feasibility_and_whole_run_uncertainty():
    from robust_learning.analysis import bootstrap_mean, matched_point
    def point(precision, hard, tail=.5):
        return dict(rejected_noise_precision=precision, hard_clean_retention=hard,
                    tail_clean_retention=tail, acceptance_fraction=.7)
    curve = [point(None, 1), point(.4, .9), point(.7, .6), point(.9, .2)]
    assert matched_point(curve, .5)["hard_clean_retention"] == .6
    assert matched_point(curve, .95) is None
    a = bootstrap_mean([-.2, .1, .4], 1000, 8)
    assert a == bootstrap_mean([-.2, .1, .4], 1000, 8)
    assert a["n"] == 3 and a["mean"] == pytest.approx(.1)
    assert a["lower"] <= a["mean"] <= a["upper"]
    assert bootstrap_mean([.25]*5)["lower"] == .25
    with pytest.raises(ValueError):
        bootstrap_mean([])


def test_partial_confusion_macro_f1_and_calibration():
    probability = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .8, .1]])
    result = classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 1]), probability)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert result["macro_f1"] == pytest.approx((1+2/3)/3)
    assert result["balanced_accuracy"] == pytest.approx(2/3)
    assert result["ece"] == pytest.approx(.8-2/3)
