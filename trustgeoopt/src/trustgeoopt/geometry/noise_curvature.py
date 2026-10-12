"""Established noise differences and randomized range/Ritz diagnostics.

No optimizer. No dense p-by-p allocation in the standalone routines.
Operators must represent a fixed symmetric PSD matrix during each refresh.
"""
from dataclasses import dataclass
import math
import time

import torch


def paired_noise_factor(first, second):
    """Columns D/sqrt(2n) estimate minibatch noise covariance via FF^T.

Rows are independent pairs at identical parameters, equal batch sizes and
conditionally iid draws. No centering: E[D]=0 by construction. Does not
identify curvature. Cross-pair independence is needed for iid error bounds.
"""
    if first.ndim != 2 or first.shape != second.shape or len(first) == 0:
        raise ValueError("nonempty equally shaped sample-by-parameter matrices required")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("finite paired gradients required")
    return (first-second).T/math.sqrt(2*len(first))


def pca_basis(samples, rank, centered=False):
    if samples.ndim != 2 or len(samples) == 0 or not 1 <= rank <= min(samples.shape):
        raise ValueError("valid samples and rank required")
    if not torch.isfinite(samples).all():
        raise ValueError("finite samples required")
    x = samples-samples.mean(0) if centered else samples
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[:rank].T


def central_gradient_difference(gradient, point, vector, delta):
    """Closure must preserve its minibatch to get common-random-number secants."""
    if delta <= 0 or not math.isfinite(delta) or point.shape != vector.shape:
        raise ValueError("positive finite delta and matching shapes required")
    return (gradient(point+delta*vector)-gradient(point-delta*vector))/(2*delta)


def autograd_hvp(loss, point, vector):
    """Fresh graph per call. Loss closure must keep data/parameters fixed."""
    if point.shape != vector.shape:
        raise ValueError("matching point/vector shapes required")
    with torch.enable_grad():
        x = point.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(loss(x), x, create_graph=True)[0]
        return torch.autograd.grad((g*vector).sum(), x)[0].detach()


@dataclass
class SpectralEstimate:
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    operator_calls: int
    refresh_seconds: float
    retained_bytes: int
    workspace_tensor_bytes_estimate: int

    def apply(self, vector):
        return self.basis@(self.eigenvalues*(self.basis.T@vector))


class RandomizedCurvature:
    """Classical randomized range finder + Rayleigh–Ritz, not a new estimator.

    Retains O(p r) state, workspace O(p(r+s)+(r+s)^2); PSD eigenvalues
    clipped at zero, QR reorthogonalizes each pass. Independent fixed probes
    and fixed operator are required. Negative curvature is deliberately not
    targeted; use a PSD GGN for networks. Reference: Halko et al. (2011).
    """
    def __init__(self, rank=4, oversample=4, power_steps=1, refresh_frequency=100):
        if rank < 1 or oversample < 0 or power_steps < 0 or refresh_frequency < 1:
            raise ValueError("invalid rank/oversampling/power/refresh")
        self.rank, self.oversample = rank, oversample
        self.power_steps, self.refresh_frequency = power_steps, refresh_frequency
        self.estimate, self.last_step = None, None
        self.last_observed_step = None

    def refresh(self, operator, template, seed=0):
        if template.ndim != 1 or len(template) < self.rank or not template.is_floating_point():
            raise ValueError("floating vector template and feasible rank required")
        if template.is_cuda:
            torch.cuda.synchronize(template.device)
        start = time.perf_counter()
        width = min(len(template), self.rank+self.oversample)
        rng = torch.Generator(device=template.device).manual_seed(seed)
        q = torch.randn(len(template), width, generator=rng,
                        dtype=template.dtype, device=template.device)
        calls = 0

        def multiply(matrix):
            nonlocal calls
            columns = []
            for column in matrix.T:
                y = operator(column).detach()
                if y.shape != template.shape or not torch.isfinite(y).all():
                    raise ValueError("operator returned invalid vector")
                columns.append(y)
                calls += 1
            return torch.stack(columns, 1)

        for _ in range(self.power_steps+1):
            q = torch.linalg.qr(multiply(q), mode="reduced").Q
        aq = multiply(q)
        small = q.T@aq
        values, rotation = torch.linalg.eigh((small+small.T)/2)
        basis = q@rotation[:, -self.rank:]
        values = values[-self.rank:].clamp_min(0)
        item = template.element_size()
        if template.is_cuda:
            torch.cuda.synchronize(template.device)
        self.estimate = SpectralEstimate(basis, values, calls, time.perf_counter()-start,
            item*(basis.numel()+values.numel()),
            item*(4*len(template)*width+3*width*width+len(template)*self.rank))
        return self.estimate

    def observe(self, step, operator, template, seed=0):
        if self.last_observed_step is not None and step < self.last_observed_step:
            raise ValueError("steps must be monotone")
        if self.estimate is None or step-self.last_step >= self.refresh_frequency:
            self.refresh(operator, template, seed)
            self.last_step = step
        self.last_observed_step = step
        return self.estimate


def basis_metrics(matrix, basis, oracle_basis=None):
    """Dense matrices permitted ONLY for small analysis controls/layer oracles."""
    rank = basis.shape[1]
    if oracle_basis is None:
        oracle_basis = torch.linalg.eigh(matrix).eigenvectors[:, -rank:]
    compressed = basis.T@matrix@basis
    reconstruction = basis@compressed@basis.T
    return {"overlap": float((basis.T@oracle_basis).square().sum()/rank),
            "capture": float(torch.trace(compressed)/torch.trace(matrix).clamp_min(1e-30)),
            "oracle_calibrated_error": float((matrix-reconstruction).norm()/matrix.norm().clamp_min(1e-30))}
