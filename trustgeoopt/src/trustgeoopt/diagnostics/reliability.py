import math

import torch

from .gradient_statistics import cosine


class ReliabilityMonitor:
    """Causal observer; accepts no losses, future updates, or future labels."""

    def __init__(self, eps=1e-12, mismatch_threshold=2.0, alignment_threshold=0.0):
        if not math.isfinite(eps) or eps <= 0 or mismatch_threshold <= 0:
            raise ValueError("invalid reliability threshold/epsilon")
        self.eps = eps
        self.mismatch_threshold = mismatch_threshold
        self.alignment_threshold = alignment_threshold
        self.previous_gradient = None
        self.previous_scale = None

    @torch.no_grad()
    def observe(self, gradient, momentum=None, second_moment=None, independent_gradient=None):
        g = gradient.detach().flatten()
        if not torch.isfinite(g).all():
            raise FloatingPointError("nonfinite diagnostic gradient")
        signals = {"gradient_norm": float(g.norm()), "scale_mismatch": float("nan"),
                   "momentum_alignment": float("nan"), "preconditioner_drift": float("nan"),
                   "history_consistency": float("nan"), "independent_consistency": float("nan")}
        if momentum is not None:
            signals["momentum_alignment"] = cosine(g, momentum.flatten(), self.eps)
        if second_moment is not None:
            v = second_moment.detach().flatten()
            if v.shape != g.shape or not torch.isfinite(v).all() or (v < 0).any():
                raise ValueError("second moment must match gradient and be finite/nonnegative")
            signals["scale_mismatch"] = float(g.square().sum()/(v.sum()+self.eps))
            scale = v.sqrt()
            if self.previous_scale is not None and self.previous_scale.shape == scale.shape:
                signals["preconditioner_drift"] = float((scale-self.previous_scale).norm() /
                                                       (self.previous_scale.norm()+self.eps))
            self.previous_scale = scale.clone()
        if self.previous_gradient is not None and self.previous_gradient.shape == g.shape:
            signals["history_consistency"] = cosine(g, self.previous_gradient, self.eps)
        if independent_gradient is not None:
            signals["independent_consistency"] = cosine(g, independent_gradient.flatten(), self.eps)
        self.previous_gradient = g.clone()
        signals["flag"] = int(signals["scale_mismatch"] > self.mismatch_threshold or
                              signals["momentum_alignment"] < self.alignment_threshold)
        return signals

    def state_dict(self):
        return dict(previous_gradient=self.previous_gradient, previous_scale=self.previous_scale)

    def load_state_dict(self, state):
        self.previous_gradient = state["previous_gradient"]
        self.previous_scale = state["previous_scale"]
