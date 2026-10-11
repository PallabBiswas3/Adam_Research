import math
import time

import torch
from torch import nn

from ..datasets.synthetic import SyntheticObjective
from ..diagnostics.gradient_statistics import cosine, prospective_adam_state
from ..diagnostics.reliability import ReliabilityMonitor
from ..optimizers import make_optimizer
from ..utils.hardware import synchronize
from ..utils.logging import RunLog
from ..utils.seeds import set_seed
from .checkpoint import save_checkpoint, load_checkpoint


class ToyParameter(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.x = nn.Parameter(value)


def run_synthetic(config, output_root="results", resume=None):
    set_seed(config["seed"], config.get("deterministic", True), config.get("threads", 1))
    device = torch.device(config.get("device", "cpu"))
    dtype = getattr(torch, config.get("dtype", "float64"))
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("synthetic experiments require float32 or float64")
    objective = SyntheticObjective(config["objective"], config["seed"], dtype, device)
    shape = config.get("parameter_shape", [objective.dimension])
    if math.prod(shape) != objective.dimension:
        raise ValueError("parameter_shape must have objective.dimension elements")
    initial = torch.randn(objective.dimension, dtype=dtype, device=device)*config.get("initial_scale", 1.0)
    if objective.kind == "rosenbrock":
        initial.fill_(-1.2)
        initial[1::2] = 1.0
    model = ToyParameter(initial.reshape(shape))
    optimizer = make_optimizer(model.parameters(), config["optimizer"])
    monitor = ReliabilityMonitor(**config.get("diagnostics", {}))
    noise_rng = torch.Generator().manual_seed(config["seed"]+700)
    first_step = 0
    initial_loss = float(objective.loss(model.x, 0))
    total_optimizer_seconds, total_diagnostic_seconds = 0.0, 0.0
    target_step = target_seconds = None
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        extra = checkpoint["extra"]
        if extra["config"] != config:
            # Budget extension alone is allowed. Other changes require a fresh trial.
            saved, current = dict(extra["config"]), dict(config)
            saved.pop("steps", None)
            current.pop("steps", None)
            if saved != current:
                raise ValueError("resume config changed beyond steps budget")
        first_step = checkpoint["step"]+1
        monitor.load_state_dict(extra["monitor"])
        noise_rng.set_state(extra["noise_rng"].cpu())
        initial_loss = extra["initial_loss"]
        target_step, target_seconds = extra["target_step"], extra["target_seconds"]
        total_optimizer_seconds = extra["optimizer_seconds"]
        total_diagnostic_seconds = extra["diagnostic_seconds"]
    log = RunLog(config, output_root, device)
    log.metadata["resume_from"] = str(resume) if resume else None
    try:
        previous_runtime = checkpoint["extra"]["elapsed_seconds"] if resume else 0.0
        if first_step >= config.get("steps", 200):
            raise ValueError("resumed checkpoint already meets the steps budget")
        spikes = harmful_count = 0
        last_loss = None
        for step in range(first_step, config.get("steps", 200)):
            optimizer.zero_grad(set_to_none=True)
            before = model.x.detach().clone()
            loss_before = float(objective.loss(before, step))
            true_gradient = objective.gradient(before, step)
            observed_gradient = true_gradient + objective.sample_noise(step, noise_rng).reshape_as(before)
            if not torch.isfinite(observed_gradient).all() or not math.isfinite(loss_before):
                raise FloatingPointError("nonfinite objective or gradient")
            model.x.grad = observed_gradient.clone()
            unclipped_norm = float(observed_gradient.norm())
            if config.get("gradient_clip") is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            synchronize(device)
            begin = time.perf_counter()
            state = prospective_adam_state(optimizer)
            signals = monitor.observe(model.x.grad) if state is None else monitor.observe(*state)
            synchronize(device)
            diagnostic_seconds = time.perf_counter()-begin
            total_diagnostic_seconds += diagnostic_seconds
            begin = time.perf_counter()
            optimizer.step()
            synchronize(device)
            optimizer_seconds = time.perf_counter()-begin
            total_optimizer_seconds += optimizer_seconds
            displacement = model.x.detach()-before
            loss_after = float(objective.loss(model.x, step))
            if not math.isfinite(loss_after) or not torch.isfinite(displacement).all():
                raise FloatingPointError("nonfinite objective or update")
            tolerance = 1e-12*max(1.0, abs(loss_before))
            harmful = int(loss_after-loss_before > tolerance)
            spike = int(loss_after-loss_before > config.get("spike_fraction", 0.1)*max(abs(loss_before), 1e-12))
            harmful_count += harmful
            spikes += spike
            elapsed = log.elapsed()+previous_runtime
            relative_error = loss_after/max(initial_loss, 1e-12) if objective.kind != "nonstationary" else float("nan")
            if target_step is None and math.isfinite(relative_error) and relative_error <= config.get("target_relative_error", 0.01):
                target_step, target_seconds = step+1, elapsed
            moment = state[2] if state is not None else None
            row = dict(step=step, objective_before=loss_before, objective_after=loss_after,
                optimization_error=loss_after, relative_error=relative_error,
                distance_to_optimum=float((model.x.flatten()-objective.optimum(step)).norm()),
                true_gradient_norm=float(true_gradient.norm()), unclipped_gradient_norm=unclipped_norm,
                step_norm=float(displacement.norm()), descent_alignment=cosine(-displacement, true_gradient),
                update_direction_curvature=objective.curvature(before, displacement, step),
                second_moment_mean=float(moment.mean()) if moment is not None else float("nan"),
                second_moment_min=float(moment.min()) if moment is not None else float("nan"),
                second_moment_max=float(moment.max()) if moment is not None else float("nan"),
                harmful=harmful, loss_spike=spike, loss_increase=loss_after-loss_before,
                finite=1, optimizer_seconds=optimizer_seconds, diagnostic_seconds=diagnostic_seconds,
                elapsed_seconds=elapsed, **signals)
            log.write(row)
            last_loss = loss_after
            if config.get("checkpoint_interval", 0) and (step+1) % config["checkpoint_interval"] == 0:
                save_checkpoint(log.path/"checkpoint.pt", model, optimizer, step, dict(config=config,
                    monitor=monitor.state_dict(), noise_rng=noise_rng.get_state(), initial_loss=initial_loss,
                    target_step=target_step, target_seconds=target_seconds,
                    elapsed_seconds=elapsed, optimizer_seconds=total_optimizer_seconds,
                    diagnostic_seconds=total_diagnostic_seconds))
        summary = dict(initial_loss=initial_loss, final_loss=last_loss, target_step=target_step,
                       target_seconds=target_seconds, target_censored=target_step is None,
                       tracking=objective.kind == "nonstationary", harmful_updates=harmful_count,
                       loss_spikes=spikes, optimizer_seconds=total_optimizer_seconds,
                       diagnostic_seconds=total_diagnostic_seconds)
        return log.finish(summary=summary)
    except Exception as error:
        log.finish(status="failed", error=error)
        raise
