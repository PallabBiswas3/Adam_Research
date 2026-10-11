"""Bounded CPU evidence campaign. Reuses the existing objectives and optimizers."""

import copy
import csv
import hashlib
import json
import math
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

from ..datasets.synthetic import SyntheticObjective
from ..diagnostics.gradient_statistics import cosine, prospective_adam_state, prospective_adamw_displacement
from ..diagnostics.reliability import ReliabilityMonitor
from ..optimizers import make_optimizer
from ..utils.hardware import environment_metadata
from ..utils.logging import clean_json, write_json
from ..utils.seeds import set_seed


def create_campaign(protocol, destination):
    set_seed(0, True, 1)
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    (path/"raw").mkdir()
    (path/"processed").mkdir()
    (path/"protocol.yaml").write_text(yaml.safe_dump(protocol), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(path/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts", "tests", "configs"):
            for source in sorted((root/folder).rglob("*")):
                if source.suffix in {".py", ".yaml"}:
                    archive.write(source, source.relative_to(root))
    state = dict(protocol=protocol, experiment_seconds=0.0, stages=[], trials=0,
                 environment=environment_metadata("cpu"),
                 snapshot_sha256=hashlib.sha256((path/"source_snapshot.zip").read_bytes()).hexdigest())
    write_json(path/"campaign.json", state)
    return path


def trial_config(protocol, regime, objective, name, rate, seed, split, steps, diagnostic=False):
    optimizer = dict(name=name, lr=rate, weight_decay=0.0)
    if name.startswith("adamw_") and name != "adamw_monitor":
        optimizer.update(step_cap=0.1, safeguard_scale=200.0)
    return dict(name=f"sprint2-{split}-{regime}-{name}", seed=seed, device="cpu", dtype="float64",
                threads=1, steps=steps, parameter_shape=protocol["parameter_shape"],
                objective=copy.deepcopy(objective), optimizer=optimizer,
                regime=regime, split=split, diagnostic=diagnostic,
                tail_fraction=protocol["tail_fraction"], target_relative_error=protocol["target_relative_error"])


@torch.no_grad()
def run_trial(config, destination, environment, deadline=None):
    """Stream-free buffered toy loop; partial rows survive numerical/budget failures.

    Analytic quadratic gradient is identical to SyntheticObjective.gradient but
    avoids autograd allocation. Baseline logs do not pay for the causal monitor.
    Timing excludes imports and CSV serialization; campaign timing includes setup/I/O.
    """
    if config["device"] != "cpu":
        raise ValueError("Sprint 2 is CPU-only")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run_id = f"{config['name']}-{digest[:10]}-{uuid.uuid4().hex[:8]}"
    path = Path(destination)/"raw"/run_id
    path.mkdir(parents=True, exist_ok=False)
    (path/"config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    meta = dict(experiment_id=run_id, config=config, config_sha256=digest,
                environment=environment, status="running", errors=[], rows=0)
    write_json(path/"metadata.json", meta)
    start, rows = time.perf_counter(), []
    target_step, changes, optimizer_seconds, diagnostic_seconds = None, 0, 0.0, 0.0
    status, error = "success", None
    initial_loss = None
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            status = "budget_not_started"
        else:
            set_seed(config["seed"], True, 1)
            obj = SyntheticObjective(config["objective"], config["seed"])
            if obj.kind == "rosenbrock":
                raise ValueError("Sprint 2 analytic runner only supports quadratic objectives")
            if math.prod(config["parameter_shape"]) != obj.dimension:
                raise ValueError("parameter_shape mismatch")
            p = torch.nn.Parameter(torch.randn(obj.dimension, dtype=torch.float64).reshape(config["parameter_shape"]))
            optimizer = make_optimizer([p], config["optimizer"])
            monitor = ReliabilityMonitor()
            rng = torch.Generator().manual_seed(config["seed"]+700)
            initial_loss = float(obj.loss(p, 0))
            for step in range(config["steps"]):
                if step % 64 == 0 and deadline is not None and time.perf_counter() >= deadline:
                    status = "budget_censored"
                    break
                before = p.detach().clone()
                matrix, center = obj.geometry(step)
                e = before.flatten()-center
                true_g = matrix @ e
                observed = true_g+obj.sample_noise(step, rng)
                loss_before = float(0.5*torch.dot(e, true_g))
                if not torch.isfinite(observed).all() or not math.isfinite(loss_before) or loss_before > 1e100:
                    raise FloatingPointError("nonfinite/divergent objective (>1e100) or gradient")
                p.grad = observed.reshape_as(p).clone()
                signals = {}
                if config["diagnostic"]:
                    begin = time.perf_counter()
                    state = prospective_adam_state(optimizer)
                    proposed = prospective_adamw_displacement(optimizer)
                    signals = monitor.observe(*state)
                    norm = float(proposed.norm())
                    directional = float(torch.dot(proposed, matrix @ proposed))
                    linear = float(torch.dot(true_g, proposed))
                    signals.update(proposed_step_norm=norm,
                        observed_descent_alignment=cosine(-proposed, observed),
                        oracle_descent_alignment=cosine(-proposed, true_g),
                        oracle_directional_curvature=directional/max(norm**2, 1e-30),
                        oracle_linear_change=linear, oracle_quadratic_change=linear+0.5*directional,
                        distance_before=float(e.norm()), learning_rate=config["optimizer"]["lr"],
                        noise_scale=obj.config.get("noise_std_after", obj.config.get("noise_std", 0.0))
                            if step >= obj.config.get("noise_change_step", 10**12) else obj.config.get("noise_std", 0.0),
                        phase=step/config["steps"])
                    diagnostic_seconds += time.perf_counter()-begin
                begin = time.perf_counter()
                optimizer.step()
                optimizer_seconds += time.perf_counter()-begin
                displacement = (p.detach()-before).flatten()
                after_e = p.detach().flatten()-center
                loss_after = float(0.5*torch.dot(after_e, matrix @ after_e))
                if not torch.isfinite(p).all() or not math.isfinite(loss_after):
                    raise FloatingPointError("nonfinite update")
                tolerance = 1e-12*max(1.0, abs(loss_before))
                if target_step is None and obj.kind != "nonstationary" and loss_after <= config["target_relative_error"]*initial_loss:
                    target_step = step+1
                row = dict(step=step, objective_before=loss_before, objective_after=loss_after,
                           loss_increase=loss_after-loss_before, harmful=int(loss_after-loss_before > tolerance),
                           harmful_tolerance=tolerance, distance_after=float(after_e.norm()),
                           step_norm=float(displacement.norm()),
                           true_descent_alignment=cosine(-displacement, true_g),
                           true_directional_curvature=float(torch.dot(displacement, matrix @ displacement))/max(float(displacement.square().sum()), 1e-30),
                           **signals)
                if config["diagnostic"]:
                    row["proposal_error"] = float((displacement-proposed).abs().max())
                rows.append(row)
    except Exception as exception:
        status, error = "failed", f"{type(exception).__name__}: {exception}"
    if rows:
        with (path/"metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    tail = rows[max(0, int(config["steps"]*(1-config["tail_fraction"]))):]
    summary = dict(initial_loss=initial_loss, final_loss=rows[-1]["objective_after"] if rows else None,
                   tail_loss=float(np.mean([r["objective_after"] for r in tail])) if tail else None,
                   tail_distance=float(np.mean([r["distance_after"] for r in tail])) if tail else None,
                   harmful_fraction=float(np.mean([r["harmful"] for r in rows])) if rows else None,
                   target_step=target_step, target_censored=target_step is None,
                   target_applicable=config["objective"]["kind"] != "nonstationary",
                   optimizer_seconds=optimizer_seconds, diagnostic_seconds=diagnostic_seconds,
                   correction_events=getattr(optimizer, "correction_events", changes) if rows else changes)
    meta.update(status=status, rows=len(rows), runtime_seconds=time.perf_counter()-start,
                errors=[error] if error else [], summary=summary)
    write_json(path/"metadata.json", meta)
    return dict(run_id=run_id, path=str(path), status=status, config=config, **summary)


def select_rates(records, required_seeds=None):
    """No evaluation rows may enter tuning, failed/censored candidates disqualified."""
    if any(r["config"]["split"] != "tuning" for r in records):
        raise ValueError("selection accepts tuning records only")
    grouped = {}
    for record in records:
        c = record["config"]
        key = (c["regime"], c["optimizer"]["name"], c["optimizer"]["lr"])
        grouped.setdefault(key, []).append(record)
    candidates = {}
    for (regime, method, rate), group in grouped.items():
        score = math.inf
        complete = required_seeds is None or (
            {r["config"]["seed"] for r in group} == set(required_seeds) and len(group) == len(required_seeds))
        if complete and all(r["status"] == "success" and r["tail_loss"] is not None and math.isfinite(r["tail_loss"]) for r in group):
            score = float(np.mean([np.log10(r["tail_loss"]+1e-12) for r in group]))
        candidates.setdefault((regime, method), []).append((score, rate))
    return {f"{regime}/{method}": min(values)[1] if math.isfinite(min(values)[0]) else None
            for (regime, method), values in candidates.items()}


def execute_stage(destination, stage):
    path = Path(destination)
    state = json.loads((path/"campaign.json").read_text())
    protocol = state["protocol"]
    if stage in state["stages"]:
        raise ValueError("stage already executed; create a new campaign to repeat")
    if stage not in {"baseline", "reliability", "corrections"}:
        raise ValueError("unknown stage")
    set_seed(0, True, 1)
    # Every later stage captures the code actually imported for that stage.
    root = Path(__file__).resolve().parents[3]
    snapshot = path/f"source_snapshot_{stage}.zip"
    with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts", "tests", "configs"):
            for source in sorted((root/folder).rglob("*")):
                if source.suffix in {".py", ".yaml"}:
                    archive.write(source, source.relative_to(root))
    environment = environment_metadata("cpu")
    environment.update(source_snapshot=snapshot.name,
                       snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    begin = time.perf_counter()
    deadline = begin+max(0, protocol["max_experiment_seconds"]-state["experiment_seconds"])
    records = []

    def execute(config):
        if state["trials"] >= protocol["max_trials"]:
            raise ValueError("max_trials exceeded")
        result = run_trial(config, path, environment, deadline)
        records.append(result)
        state["trials"] += 1
        with (path/"trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean_json(result), allow_nan=False)+"\n")
        if len(records) % 25 == 0:
            print(f"{stage}: {len(records)} trials, {time.perf_counter()-begin:.1f}s", flush=True)
        return result

    if stage in {"baseline", "corrections"}:
        rates_by_method = protocol["learning_rates"] if stage == "baseline" else {
            f"adamw_{mechanism}": protocol["learning_rates"]["adamw"] for mechanism in ("step_clip", "momentum_reset", "lr_safeguard")}
        for regime, objective in protocol["regimes"].items():
            for method, rates in rates_by_method.items():
                for rate in rates:
                    for seed in protocol["development_seeds"]:
                        execute(trial_config(protocol, regime, objective, method, rate, seed, "tuning", protocol["steps"]))
        selections = select_rates(records, protocol["development_seeds"])
        write_json(path/"processed"/("selected_rates.json" if stage == "baseline" else "correction_selected_rates.json"), selections)
        for regime, objective in protocol["regimes"].items():
            for method in rates_by_method:
                rate = selections[f"{regime}/{method}"]
                if rate is not None:
                    for seed in protocol["evaluation_seeds"]:
                        execute(trial_config(protocol, regime, objective, method, rate, seed,
                                             "evaluation" if stage == "baseline" else "correction_evaluation", protocol["steps"]))
    else:
        for split, seeds in (("development", protocol["development_seeds"]), ("evaluation", protocol["evaluation_seeds"])):
            regimes = dict(protocol["regimes"])
            if split == "evaluation":
                regimes.update(protocol["held_out_regimes"])
            for regime, original in regimes.items():
                objective = copy.deepcopy(original)
                if regime in protocol["regimes"]:
                    for key in ("noise_change_step", "change_step"):
                        if key in objective:
                            objective[key] = protocol["reliability_steps"]//2
                for rate in protocol["reliability_learning_rates"]:
                    for seed in seeds:
                        execute(trial_config(protocol, regime, objective, "adamw", rate, seed,
                                             f"reliability_{split}", protocol["reliability_steps"], True))
    state["experiment_seconds"] += time.perf_counter()-begin
    state["stages"].append(stage)
    write_json(path/"campaign.json", state)
    return records


def published_counterexample(config):
    """Reddi et al. Appendix A: no bias correction, epsilon zero, 1D projection."""
    c, alpha, steps = config["C"], config["alpha"], config["steps"]
    beta2 = 1/(1+c*c)
    if not (c > 2 and 0 < alpha < math.sqrt(1-beta2) and steps > 0 and steps % 3 == 0):
        raise ValueError("Appendix A requires C>2, alpha<sqrt(1-beta2), complete cycles")
    if config["beta1"] != 0 or config["beta2"] != beta2 or config["epsilon"] != 0 or config["bias_correction"]:
        raise ValueError("published assumptions changed")
    if config["x1"] != 1 or config["domain"] != [-1.0, 1.0] or config["schedule"] != "alpha/sqrt(t)":
        raise ValueError("published initial point/domain/schedule changed")
    rows = []
    for method in ("adam_paper", "amsgrad_paper", "sgd", "adagrad"):
        x, v, maximum, total_square, regret = 1.0, 0.0, 0.0, 0.0, 0.0
        min_x, cycle_error = 1.0, 0.0
        for t in range(1, steps+1):
            g = c if t % 3 == 1 else -1.0
            regret += g*(x+1)  # loss incurred BEFORE update, comparator x*=-1
            min_x = min(min_x, x)
            if t % 3 == 1:
                cycle_error = max(cycle_error, abs(x-1))
            v = beta2*v+(1-beta2)*g*g
            maximum = max(maximum, v)
            total_square += g*g
            denom = {"adam_paper": math.sqrt(v), "amsgrad_paper": math.sqrt(maximum),
                     "sgd": 1.0, "adagrad": math.sqrt(total_square/t)}[method]
            x = max(-1.0, min(1.0, x-alpha*g/(math.sqrt(t)*denom)))
            if t % 3 == 0:
                rows.append(dict(method=method, t=t, x_next=x, average_regret=regret/t,
                                 minimum_preupdate_x=min_x, maximum_cycle_start_error=cycle_error))
    return rows
