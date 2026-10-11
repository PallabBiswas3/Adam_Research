"""Established first-order safeguards adapted to dense AdamW research trials.

No novelty or convergence guarantee is claimed. Norm clipping acts on the UPDATE,
not on the gradient; a reset zeroes only the first moment and keeps its global
step count and second moment. No objective, Hessian, or future label is consulted.
"""
import math

import torch

from ..diagnostics.gradient_statistics import prospective_adamw_displacement


class CheapAdamW(torch.optim.AdamW):
    def __init__(self, params, correction, step_cap=0.1, safeguard_scale=200.0, **kwargs):
        if correction not in {"step_clip", "momentum_reset", "lr_safeguard"}:
            raise ValueError("unknown correction")
        if not math.isfinite(step_cap) or step_cap <= 0 or not math.isfinite(safeguard_scale) or safeguard_scale <= 0:
            raise ValueError("positive finite cap and schedule scale required")
        super().__init__(params, **kwargs)
        self.correction = correction
        self.step_cap = step_cap
        self.safeguard_scale = safeguard_scale
        self.correction_step = 0
        self.correction_events = 0
        self.base_rates = [group["lr"] for group in self.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("research corrections do not support closures")
        active = [p for group in self.param_groups for p in group["params"] if p.grad is not None]
        if any(p.is_complex() or p.grad.is_sparse for p in active):
            raise ValueError("real dense gradients required")
        before = [p.detach().clone() for p in active] if self.correction == "step_clip" else None
        if self.correction == "momentum_reset" and active:
            proposed = prospective_adamw_displacement(self)
            gradient = torch.cat([p.grad.flatten() for p in active])
            if torch.dot(gradient, proposed) > 0:
                for p in active:
                    if "exp_avg" in self.state.get(p, {}):
                        self.state[p]["exp_avg"].zero_()
                self.correction_events += 1
        if self.correction == "lr_safeguard":
            factor = 1/math.sqrt(1+self.correction_step/self.safeguard_scale)
            for group, rate in zip(self.param_groups, self.base_rates):
                group["lr"] = rate*factor
            self.correction_events += int(self.correction_step > 0 and bool(active))
        result = super().step()
        if self.correction == "lr_safeguard":
            for group, rate in zip(self.param_groups, self.base_rates):
                group["lr"] = rate
        if before:
            displacement = torch.cat([(p-old).flatten() for p, old in zip(active, before)])
            norm = float(displacement.norm())
            if norm > self.step_cap:
                factor = self.step_cap/norm
                for p, old in zip(active, before):
                    p.copy_(old+(p-old)*factor)
                self.correction_events += 1
        self.correction_step += 1
        return result

    def state_dict(self):
        result = super().state_dict()
        result["cheap_correction"] = dict(correction=self.correction, step_cap=self.step_cap,
            safeguard_scale=self.safeguard_scale, correction_step=self.correction_step,
            correction_events=self.correction_events, base_rates=self.base_rates)
        return result

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        correction = state.pop("cheap_correction")
        if (correction["correction"], correction["step_cap"], correction["safeguard_scale"]) != (
                self.correction, self.step_cap, self.safeguard_scale):
            raise ValueError("correction configuration changed")
        super().load_state_dict(state)
        self.correction_step = correction["correction_step"]
        self.correction_events = correction["correction_events"]
        self.base_rates = correction["base_rates"]
