"""Registered equation/data tests, before independent strong-baseline modules."""
import numpy as np
import pytest
import torch

from robust_learning.strong.equations import (
    adjusted_probability, candidate_classes, ccs_loss, mann_kendall_z,
    refine_targets, sharpen, semisupervised_loss, fit_clean_probability,
    classwise_keep, peer_policies,
)
from robust_learning.strong.cifar import make_splits, corrupt, ImageTrainingView


def test_prior_aware_adjustment_and_positive_prior_validation():
    p = torch.tensor([[.6, .3, .1], [.2, .4, .4]])
    marginal = torch.tensor([.5, .3, .2])
    prior = torch.tensor([.7, .2, .1])
    expected = p*prior/marginal
    expected /= expected.sum(1, keepdim=True)
    assert torch.allclose(adjusted_probability(p, marginal, prior), expected)
    assert torch.allclose(adjusted_probability(p, marginal, torch.ones(3)/3), (p/marginal)/(p/marginal).sum(1, keepdim=True))
    with pytest.raises(ValueError):
        adjusted_probability(p, marginal, torch.tensor([1., 0., 0.]))


def test_ccs_exclusion_has_zero_gradient_and_minibatch_mean():
    logits = torch.tensor([[1., 3., 2.], [2., 1., 3.]], requires_grad=True)
    y = torch.tensor([0, 2]); exclude = torch.tensor([[False, True, False], [False, True, False]])
    loss = ccs_loss(logits, y, exclude, torch.tensor([True, False]))
    expected = torch.nn.functional.cross_entropy(logits[:1, [0, 2]], torch.tensor([0]))/2
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[0, 1] == 0 and torch.equal(logits.grad[1], torch.zeros(3))
    assert not torch.allclose(loss, torch.nn.functional.cross_entropy(logits[:1], y[:1]))
    with pytest.raises(ValueError):
        ccs_loss(logits, y, torch.ones_like(exclude), torch.ones(2, dtype=torch.bool))


def test_class_trends_are_not_confidence_gap_trends():
    history = torch.tensor([[[.1, .4, .5]], [[.2, .3, .5]], [[.3, .2, .5]], [[.4, .1, .5]]])
    z = mann_kendall_z(history)
    variance = 4*3*13/18
    assert z[0, 0] == pytest.approx(5/variance**.5)
    assert z[0, 1] < 0 and z[0, 2] == 0
    candidates = candidate_classes(history, torch.tensor([1]), 1.2815515655446004)
    assert candidates.tolist() == [[True, False, False]]
    assert not candidate_classes(history, torch.tensor([0]), 1.2815515655446004)[0, 0]


def test_refinement_sharpening_and_ssl_exact_values():
    predictions = torch.tensor([[.2, .8], [.8, .2]])
    y = torch.tensor([0, 1]); w = torch.tensor([1., 0.])
    target = refine_targets(y, predictions, w, .5)
    assert torch.equal(target[0], torch.tensor([1., 0.]))
    assert torch.allclose(target[1], torch.tensor([16/17, 1/17]))
    assert torch.allclose(sharpen(predictions, 1.), predictions)
    assert sharpen(predictions, .5)[0, 1] > predictions[0, 1]
    assert sharpen(predictions, 2.)[0, 1] < predictions[0, 1]
    lx, lu, reg = semisupervised_loss(torch.zeros(2, 2), torch.eye(2), torch.zeros(2, 2), torch.eye(2))
    assert lx == pytest.approx(np.log(2)) and lu == pytest.approx(.25) and reg == pytest.approx(0)


def test_gmm_orientation_constant_failure_and_peer_exchange():
    loss = np.concatenate([np.linspace(.05, .15, 50), np.linspace(.8, 1., 50)])
    probability = fit_clean_probability(loss, 600)
    assert probability[:50].mean() > .99 and probability[50:].mean() < .01
    with pytest.raises(ValueError):
        fit_clean_probability(np.ones(20), 600)
    first = (torch.tensor([True, False]), torch.tensor([[False, True], [True, False]]))
    second = (torch.tensor([False, True]), torch.tensor([[True, False], [False, True]]))
    peers = peer_policies(first, second)
    assert peers[0] is second and peers[1] is first


@pytest.mark.parametrize("kind", ["symmetric_including_original", "asymmetric"])
def test_corruption_replay_and_actual_flip_fidelity(kind):
    labels = np.tile(np.arange(10), 100)
    y, flips, intervention = corrupt(labels, 44, .5, kind)
    yy, ff, ii = corrupt(labels, 44, .5, kind)
    np.testing.assert_array_equal(y, yy); np.testing.assert_array_equal(flips, ff)
    assert np.array_equal(flips, y != labels)
    assert np.array_equal(intervention, ii)
    assert np.all(~flips | intervention)
    if kind == "symmetric_including_original":
        assert flips.sum() < intervention.sum()
    else:
        assert np.array_equal(y[labels == 0], labels[labels == 0])
    assert not corrupt(labels, 0, 0, kind)[1].any()


def test_split_ids_longtail_prior_and_no_oracle_training_fields():
    labels = np.repeat(np.arange(10), 100)
    split = make_splits(labels, 123, validation_per_class=30, imbalance_factor=10, prior_majority=10, model_val_per_class=10)
    train, validation, prior = split["train"], split["validation"], split["prior"]
    assert not set(train)&set(validation) and not set(train)&set(prior) and not set(validation)&set(prior)
    counts = np.bincount(labels[train], minlength=10)
    assert counts[0] == 70 and counts[-1] == 7
    assert np.bincount(labels[validation], minlength=10).tolist() == [10]*10
    assert np.bincount(labels[prior], minlength=10)[-1] == 1
    images = np.zeros((len(train), 32, 32, 3), dtype=np.uint8)
    view = ImageTrainingView(images, np.zeros(len(train), dtype=np.int64), train)
    assert not hasattr(view, "true_y") and not hasattr(view, "flipped") and not hasattr(view, "oracle")
    with pytest.raises(ValueError):
        view.observed_y[0] = 3


def test_classwise_control_uses_observed_classes_only():
    keep = classwise_keep(torch.tensor([1., 3., 4., 2.]), torch.tensor([0, 0, 1, 1]), .5)
    assert keep.tolist() == [True, False, False, True]


def test_incremental_trend_state_and_mixup_beta_reflection():
    from robust_learning.strong.kernels import DSSState, mixup
    state = DSSState(2, 3, torch.ones(3)/3)
    for p in [torch.tensor([[.2, .5, .3], [.6, .1, .3]]), torch.tensor([[.3, .4, .3], [.5, .2, .3]]),
              torch.tensor([[.4, .3, .3], [.4, .3, .3]]), torch.tensor([[.5, .2, .3], [.3, .4, .3]])]:
        state.observe(torch.arange(2), p); state.finish_epoch()
    expected = candidate_classes(torch.stack(state.history), torch.tensor([1, 0]), 1.2815515655446004)
    assert torch.equal(state.policy(torch.tensor([1, 0]))[1], expected)
    class RNG:
        def beta(self, a, b):
            return .2
    images, targets = torch.tensor([[0.], [1.], [2.], [3.]]), torch.eye(4)
    mixed, target = mixup(images, targets, 4., RNG(), torch.Generator().manual_seed(3))
    permutation = torch.randperm(4, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(mixed, .8*images+.2*images[permutation])
    assert torch.allclose(target.sum(1), torch.ones(4))


def test_peer_receives_no_gradients_in_dividemix_update():
    from robust_learning.strong.kernels import dividemix_update
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(3*32*32, 10)
        def forward(self, x):
            return self.linear(x.flatten(1))
    from robust_learning.strong.cifar import image_transforms
    view = ImageTrainingView(np.zeros((8, 32, 32, 3), dtype=np.uint8), np.arange(8), np.arange(8))
    first, peer = Tiny(), Tiny()
    optimizer = torch.optim.SGD(first.parameters(), lr=.02)
    config = dict(temperature=.5, mixup_alpha=4., lambda_u=25., regularizer_weight=1.)
    result = dividemix_update(first, peer, optimizer, view, np.array([0, 1]), np.array([2, 3]),
        np.full(8, .7), image_transforms()[0], config, np.random.default_rng(7), torch.Generator().manual_seed(8))
    assert np.isfinite(result["loss"])
    assert all(p.grad is None for p in peer.parameters())
    assert any(p.grad is not None for p in first.parameters())
    assert not peer.training


def test_dividemix_paper_and_code_mse_conventions_are_explicit():
    x = torch.zeros(2, 2); target = torch.eye(2)
    code = semisupervised_loss(x, target, x, target)[1]
    paper = semisupervised_loss(x, target, x, target, mse_class_reduction="sum")[1]
    assert paper == 2*code


def test_real_backbone_shape_and_license_attribution():
    from pathlib import Path
    from robust_learning.strong.vendor_dividemix import PreResNet
    torch.set_num_threads(2)
    model = PreResNet.ResNet18(num_classes=10).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) > 10_000_000
    license_text = (Path(PreResNet.__file__).parent/"LICENSE").read_text()
    assert "MIT License" in license_text and "Junnan Li" in license_text
