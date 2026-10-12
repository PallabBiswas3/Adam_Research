import math

import pytest
import torch

from trustgeoopt.geometry.noise_curvature import (
    RandomizedCurvature, autograd_hvp, central_gradient_difference,
    paired_noise_factor, pca_basis)
from trustgeoopt.training.sprint4 import layer_objects, schedule, select_policies
from trustgeoopt.models.mlp import MLP
from torch.func import functional_call


def test_paired_noise_scaling_and_mean_cancellation():
    # Exact balanced population, deterministic signal cancels from each pair.
    a = torch.tensor([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]], dtype=torch.float64)
    b = torch.zeros_like(a)
    f = paired_noise_factor(a+19, b+19)
    assert torch.allclose(f@f.T, a.T@a/8)
    with pytest.raises(ValueError):
        paired_noise_factor(a, b[:2])


def test_paired_noise_gaussian_independent_identifies_sigma():
    rng = torch.Generator().manual_seed(1200)
    sigma = torch.tensor([1., 4.], dtype=torch.float64)
    a = torch.randn(20000, 2, generator=rng, dtype=torch.float64)*sigma.sqrt()+30
    b = torch.randn(20000, 2, generator=rng, dtype=torch.float64)*sigma.sqrt()+30
    f = paired_noise_factor(a, b)
    assert torch.allclose(f@f.T, torch.diag(sigma), atol=.09)


def test_shared_noise_secants_and_autograd_hvp():
    h = torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64)
    x, v = torch.tensor([.2, .3], dtype=torch.float64), torch.tensor([1., -.5], dtype=torch.float64)
    noise = torch.tensor([200., -100.], dtype=torch.float64)
    delta = central_gradient_difference(lambda z: h@z+noise, x, v, .01)
    assert torch.allclose(delta, h@v, atol=1e-10)
    assert torch.allclose(autograd_hvp(lambda z: z@h@z/2, x, v), h@v)
    with pytest.raises(ValueError):
        central_gradient_difference(lambda z: z, x, v, 0)


def test_range_ritz_psd_reconstruction_and_counts():
    h = torch.diag(torch.tensor([0., 0., 2., 8., 20.], dtype=torch.float64))
    estimator = RandomizedCurvature(rank=3, oversample=2, power_steps=1)
    result = estimator.refresh(lambda v: h@v, torch.zeros(5, dtype=torch.float64), seed=22)
    assert result.operator_calls == 15
    assert torch.allclose(result.basis.T@result.basis, torch.eye(3, dtype=h.dtype), atol=1e-12)
    assert torch.allclose(result.basis@torch.diag(result.eigenvalues)@result.basis.T, h, atol=1e-10)
    assert result.retained_bytes == 8*(5*3+3)


def test_refresh_and_invalid_operator():
    estimator = RandomizedCurvature(2, 0, 0, 5)
    x = torch.zeros(8)
    a = estimator.observe(0, lambda v: v, x)
    assert estimator.observe(4, lambda v: 2*v, x) is a
    with pytest.raises(ValueError):
        estimator.observe(3, lambda v: v, x)
    assert estimator.observe(5, lambda v: 2*v, x) is not a
    with pytest.raises(ValueError):
        estimator.observe(3, lambda v: v, x)
    with pytest.raises(ValueError):
        estimator.refresh(lambda v: v*math.nan, x)
    with pytest.raises(ValueError):
        RandomizedCurvature(rank=0)


def test_subtraction_counterexample_does_not_recover_hessian():
    h = torch.diag(torch.tensor([1., 10.], dtype=torch.float64))
    excitation = torch.diag(torch.tensor([200., .001], dtype=torch.float64))
    signal = h@excitation@h
    assert signal.diag().argmax() == 0  # H's leading direction is the other axis.
    samples = torch.tensor([[2., 0.], [2., 1.], [2., -1.]], dtype=torch.float64)
    assert abs(float(pca_basis(samples, 1, True)[1, 0])) == 1


def test_schedules_endpoints_and_reject_unknown():
    assert schedule("conservative", 0, 100) == 1
    assert schedule("conservative", 99, 100) == .1
    assert schedule("cosine", 99, 100) == .01
    with pytest.raises(ValueError):
        schedule("adaptive_unfrozen", 0, 100)


def test_selection_rejects_evaluation_and_failed_candidate():
    protocol = {"development_seeds": [50, 51], "candidates": {"adamw": [0, 1]},
                "selection_floor": 1e-12, "selection_sd_penalty": .5}
    rows = []
    for candidate in [0, 1]:
        for seed in [50, 51]:
            for regime in ["rising", "falling"]:
                rows.append({"config": {"seed": seed, "regime": regime, "split": "tuning",
                    "candidate_id": candidate, "optimizer": {"name": "adamw", "lr": candidate+1}},
                    "status": "success", "tail_loss": 1. if candidate == 0 else .1})
    rows[-1]["status"] = "failed"
    selection, scores = select_policies(rows, protocol)
    assert selection["adamw"]["candidate_id"] == 0
    assert not scores[1]["valid"]
    rows[0]["config"]["split"] = "evaluation"
    with pytest.raises(ValueError):
        select_policies(rows, protocol)


def test_layer_ggn_matches_autograd_hessian_block_and_per_example_gradient():
    torch.manual_seed(777)
    model = MLP(3, 2, [4], initialization="kaiming").double()
    x = torch.randn(7, 3, dtype=torch.float64)
    y = torch.randint(2, (7,))
    objects = layer_objects(model, x, y)
    mapping = dict(model.named_parameters())
    for obj in objects:
        size = math.prod(obj["weight_shape"])
        def loss(point):
            state = {**mapping, obj["prefix"]+".weight": point[:size].reshape(obj["weight_shape"]),
                     obj["prefix"]+".bias": point[size:]}
            return torch.nn.functional.cross_entropy(functional_call(model, state, x), y)
        h = torch.autograd.functional.hessian(loss, obj["point"])
        g = torch.autograd.functional.jacobian(loss, obj["point"])
        assert torch.allclose(h, obj["ggn"], atol=1e-10)
        assert torch.allclose(g, obj["per_example"].mean(0), atol=1e-10)


def test_known_excitation_identifiability_formula():
    # Noncommuting SPD H and C, exactly known R=H C H.
    h = torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64)
    c = torch.tensor([[3., -.5], [-.5, 1.]], dtype=torch.float64)
    def root(a, exponent):
        values, vectors = torch.linalg.eigh(a)
        return (vectors*values.pow(exponent))@vectors.T
    r = h@c@h
    chalf, cinvhalf = root(c, .5), root(c, -.5)
    recovered = cinvhalf@root(chalf@r@chalf, .5)@cinvhalf
    assert torch.allclose(recovered, h, atol=1e-12)


def test_population_gap_perturbation_bound_for_ordered_reference():
    h = torch.diag(torch.tensor([1., 2., 5., 9.], dtype=torch.float64))
    reference = .2*h@h+3*torch.eye(4, dtype=h.dtype)
    generator = torch.Generator().manual_seed(30)
    noise = torch.randn(4, 4, generator=generator, dtype=h.dtype)
    error = .01*(noise+noise.T)
    u = torch.linalg.eigh(reference).eigenvectors[:, -2:]
    v = torch.linalg.eigh(reference+error).eigenvectors[:, -2:]
    sin_f = (2-(u.T@v).square().sum()).clamp_min(0).sqrt()
    bound = 2*min(2**.5*torch.linalg.matrix_norm(error, 2), error.norm())/(.2*(5**2-2**2))
    assert sin_f <= bound


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optional CUDA unavailable")
def test_range_ritz_cuda():
    x = torch.zeros(8, device="cuda")
    result = RandomizedCurvature(2).refresh(lambda v: v, x)
    assert result.basis.is_cuda
