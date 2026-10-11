"""Mechanism study. Existing optimizers only; immutable attempts and split-safe tuning."""
import csv
import hashlib
import itertools
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.synthetic import SyntheticObjective, supervised_splits
from ..models.mlp import MLP
from ..optimizers import make_optimizer
from ..optimizers.vendor.soap import SOAP
from ..utils.hardware import environment_metadata
from ..utils.logging import write_json
from ..utils.seeds import set_seed
from .quadratic_theory import effective_condition
from .sprint2 import create_campaign, run_trial, trial_config


def schedule_multiplier(name, step, steps):
    if name == "constant":
        return 1.0
    if name == "cosine":
        return 0.01+0.99*(1+math.cos(math.pi*min(step, steps-1)/max(steps-1, 1)))/2
    raise ValueError(name)


def optimizer_for(params, candidate):
    options = dict(candidate)
    options.pop("schedule", None)
    if options["name"] == "soap":
        options.pop("name")
        # Preserve the external reference byte-for-byte. It allocates factors in
        # default dtype, so float64 toy callers temporarily set default dtype.
        return SOAP(params, weight_decay=0, betas=(0.95, 0.95),
                    max_precond_dim=128, precondition_frequency=10, **options)
    return make_optimizer(params, options)


def matched_problem(config, seed):
    """A paired coordinate transform, NOT two independent random objectives."""
    d, k = config.get("dimension", 16), config["condition"]
    generator = torch.Generator().manual_seed(seed+100)
    q = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64)).Q
    coordinate = q if config["rotation"] == "rotated" else torch.eye(d, dtype=torch.float64)
    spectrum = torch.logspace(0, math.log10(k), d, dtype=torch.float64)
    h = coordinate@torch.diag(spectrum)@coordinate.T
    generator.manual_seed(seed)
    x = coordinate@torch.randn(d, generator=generator, dtype=torch.float64)
    mode = config.get("noise", "isotropic")
    if mode == "isotropic":
        factor = torch.eye(d, dtype=torch.float64)
    else:
        values = torch.logspace(0, 2, d, dtype=torch.float64)
        values *= d/values.sum()  # equal trace, not equal largest eigenvalue
        if mode == "antialigned":
            values = values.flip(0)
        if mode == "misaligned":
            generator.manual_seed(seed+102)
            basis = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64)).Q
        elif mode in {"aligned", "antialigned"}:
            basis = torch.eye(d, dtype=torch.float64)
        else:
            raise ValueError(mode)
        factor = basis@torch.diag(values.sqrt())
    return h, x, coordinate@factor


def subspace_metrics(h, gradients, rank, seed):
    """Centered and uncentered history; covariance is never called a Hessian."""
    d = len(h)
    _, hu = torch.linalg.eigh(h)
    top_h = hu[:, -rank:]
    rng = torch.Generator().manual_seed(seed+333)
    random = torch.linalg.qr(torch.randn(d, rank, generator=rng, dtype=h.dtype)).Q
    diagonal = torch.eye(d, dtype=h.dtype)[:, torch.argsort(h.diag())[-rank:]]
    g = torch.stack(gradients)
    centered = g-g.mean(0)
    bases = {"random": random, "diagonal": diagonal, "oracle": top_h}
    for name, samples in (("history", g), ("centered_history", centered)):
        _, u = torch.linalg.eigh(samples.T@samples/len(samples))
        bases[name] = u[:, -rank:]
    result = {}
    for name, u in bases.items():
        result[f"{name}_overlap"] = float((u.T@top_h).square().sum()/rank)
        result[f"{name}_curvature_capture"] = float(torch.trace(u.T@h@u)/torch.trace(h))
    # Direct comparison of covariances, with scale removed.
    c = centered.T@centered/len(centered)
    result["covariance_hessian_cosine"] = float((c*h).sum()/(c.norm()*h.norm()).clamp_min(1e-30))
    return result


def preconditioner_matrix(optimizer, p):
    """Frozen linear scaling at a snapshot; excludes momentum/LR dynamics."""
    d, group, state = p.numel(), optimizer.param_groups[0], optimizer.state[p]
    if isinstance(optimizer, SOAP):
        if "Q" not in state or state.get("step", 0) == 0:
            return torch.eye(d, dtype=p.dtype)
        denom = (state["exp_avg_sq"]/(1-group["betas"][1]**state["step"])).sqrt()+group["eps"]
        columns = []
        for e in torch.eye(d, dtype=p.dtype):
            projected = optimizer.project(e.reshape_as(p), state)
            columns.append(optimizer.project_back(projected/denom, state).flatten())
        return torch.stack(columns, dim=1)
    if "exp_avg_sq" in state:
        step = float(state["step"])
        v = state.get("max_exp_avg_sq", state["exp_avg_sq"])/(1-group["betas"][1]**step)
        return torch.diag(1/(v.flatten().sqrt()+group["eps"]))
    return torch.eye(d, dtype=p.dtype)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


@torch.no_grad()
def quadratic_trial(config, path, deadline):
    set_seed(config["seed"], True, 1)
    seed, objective = config["seed"], config["objective"]
    if config["stage"] == "stochastic":
        obj = SyntheticObjective(objective, seed)
        h = obj.a
        p = nn.Parameter(torch.randn(16, dtype=torch.float64).reshape(4, 4))
        factor = obj.noise_factor
    else:
        h, x, factor = matched_problem(objective, seed)
        p = nn.Parameter(x.reshape(4, 4))
        obj = None
    optimizer = optimizer_for([p], config["optimizer"])
    rng = torch.Generator().manual_seed(seed+700)
    rows, snapshots, history = [], [], []
    initial = float(p.flatten()@h@p.flatten()/2)
    start = time.perf_counter()
    target = None
    steps = config["steps"]
    for step in range(steps):
        if step % 64 == 0 and time.perf_counter() >= deadline:
            return {"status": "budget_censored", "rows": rows, "snapshots": snapshots}
        true = h@p.flatten()
        scale = objective.get("noise_std", 0)
        if step >= objective.get("noise_change_step", steps+1):
            scale = objective.get("noise_std_after", scale)
        noise = scale*(factor@torch.randn(16, generator=rng, dtype=torch.float64))
        observed = true+noise
        if not torch.isfinite(observed).all():
            error = FloatingPointError("nonfinite gradient")
            error.partial_rows = rows
            raise error
        p.grad = observed.reshape_as(p)
        lr = config["optimizer"]["lr"]*schedule_multiplier(config["optimizer"].get("schedule", "constant"), step, steps)
        optimizer.param_groups[0]["lr"] = lr
        before = p.detach().clone()
        try:
            optimizer.step()
        except Exception as error:
            error.partial_rows = rows
            raise
        loss = float(p.flatten()@h@p.flatten()/2)
        if not math.isfinite(loss) or loss > 1e100:
            error = FloatingPointError("nonfinite/divergent loss (>1e100)")
            error.partial_rows = rows
            raise error
        elapsed = time.perf_counter()-start
        displacement = (p-before).flatten()
        norm2 = float(displacement.square().sum())
        if target is None and loss <= initial*0.01:
            target = {"step": step+1, "seconds": elapsed}
        state = optimizer.state[p]
        v = state.get("exp_avg_sq")
        vmean = float(v.mean()/(1-optimizer.param_groups[0]["betas"][1]**float(state["step"]))) if v is not None and float(state.get("step", 0)) > 0 else None
        rows.append({"step": step+1, "loss": loss, "lr": lr, "noise_std": scale,
            "elapsed_seconds": elapsed, "step_norm": math.sqrt(norm2), "second_moment_mean": vmean,
            "directional_curvature": float(displacement@h@displacement)/max(norm2, 1e-30)})
        # Diagnostic work is restricted to evaluation, separately timed below.
        if config["stage"] == "geometry" and config["split"] != "tuning":
            history.append(observed.clone())
            history = history[-config["history_window"]:]
            if step+1 in {100, steps//2, steps}:
                begin = time.perf_counter()
                b = preconditioner_matrix(optimizer, p)
                b = (b+b.T)/2
                metrics = subspace_metrics(h, history, config["rank"], seed)
                metrics["hessian_is_isotropic"] = objective["condition"] == 1
                _, noise_u = torch.linalg.eigh(factor@factor.T)
                _, h_u = torch.linalg.eigh(h)
                metrics.update(noise_hessian_overlap=float((noise_u[:, -config["rank"]:].T@h_u[:, -config["rank"]:]).square().sum()/config["rank"]),
                    effective_condition=effective_condition(h.numpy(), b.numpy()),
                    preconditioned_hessian_scale_mismatch=float((b@h-torch.eye(16)*torch.trace(b@h)/16).norm()/(b@h).norm()),
                    diagnostic_seconds=time.perf_counter()-begin, step=step+1)
                snapshots.append(metrics)
    with (path/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(path/"geometry.json", snapshots)
    tail = [r["loss"] for r in rows[int(steps*(1-config["tail_fraction"])):]]
    return {"status": "success", "steps_completed": steps, "tail_loss": float(np.mean(tail)),
            "final_loss": rows[-1]["loss"], "initial_loss": initial, "target": target,
            "target_censored": target is None, "loop_seconds": time.perf_counter()-start,
            "diagnostic_seconds": sum(s["diagnostic_seconds"] for s in snapshots),
            "state_tensor_bytes": tensor_bytes(optimizer.state), "geometry": snapshots}


def neural_data(config):
    if config["model"] != "fashion":
        return supervised_splits({**config["dataset"], "data_seed": config["data_seed"]}, config["seed"])[:2]
    from torchvision.datasets import FashionMNIST
    dataset = FashionMNIST(root=config["data_root"], train=True, download=False)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(700))
    a, b = config["fashion_train_size"], config["fashion_validation_size"]
    # Disjoint subsets of the official TRAIN set; official test is never inspected.
    inputs = (dataset.data[indices[:a+b]].float()/255-0.2860)/0.3530
    labels = dataset.targets[indices[:a+b]]
    return TensorDataset(inputs[:a], labels[:a]), TensorDataset(inputs[a:], labels[a:])


def neural_trial(config, path, deadline):
    set_seed(config["seed"], True, 1)
    train, validation = neural_data(config)
    model = MLP(784 if config["model"] == "fashion" else config["dataset"]["dimension"],
                10 if config["model"] == "fashion" else 2, config["hidden"], initialization="kaiming")
    optimizer = optimizer_for(model.parameters(), config["optimizer"])
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]+900))
    criterion = nn.CrossEntropyLoss()
    process = psutil.Process()
    max_rss, peak_state, target, rows = process.memory_info().rss, 0, None, []
    begin = time.perf_counter()
    total_steps = len(loader)*config["epochs"]
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for batch, (x, y) in enumerate(loader):
            if time.perf_counter() >= deadline:
                return {"status": "budget_censored", "epochs_completed": epoch, "rows": rows}
            optimizer.param_groups[0]["lr"] = config["optimizer"]["lr"]*schedule_multiplier(
                config["optimizer"].get("schedule", "constant"), epoch*len(loader)+batch, total_steps)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                error = FloatingPointError("nonfinite neural loss")
                error.partial_rows = rows
                raise error
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())*len(y)
        model.eval()
        with torch.no_grad():
            vx, vy = validation.tensors
            logits = model(vx)
            value = float(criterion(logits, vy))
            accuracy = float((logits.argmax(1) == vy).float().mean())
        if not math.isfinite(value):
            error = FloatingPointError("nonfinite validation loss")
            error.partial_rows = rows
            raise error
        elapsed = time.perf_counter()-begin
        if target is None and value <= config["target_loss"]:
            target = {"epoch": epoch+1, "seconds": elapsed}
        max_rss = max(max_rss, process.memory_info().rss)
        peak_state = max(peak_state, tensor_bytes(optimizer.state))
        rows.append({"epoch": epoch+1, "validation_loss": value, "accuracy": accuracy,
                     "train_loss": train_loss/len(train), "elapsed_seconds": elapsed})
    with (path/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "success", "epochs_completed": config["epochs"],
            "validation_loss": rows[-1]["validation_loss"], "accuracy": rows[-1]["accuracy"],
            "target": target, "target_censored": target is None, "loop_seconds": time.perf_counter()-begin,
            "sampled_process_rss_bytes": max_rss, "state_tensor_bytes": peak_state,
            "parameter_tensor_bytes": tensor_bytes(list(model.parameters()))}


def select_candidates(records, seeds, metric, floor=1e-12):
    if any(r["config"]["split"] != "tuning" for r in records):
        raise ValueError("evaluation data cannot be used for selection")
    groups = {}
    for r in records:
        c = r["config"]
        key = (c["regime"], c["optimizer"]["name"], json.dumps(c["optimizer"], sort_keys=True))
        groups.setdefault(key, []).append(r)
    scores = {}
    for (regime, name, candidate), group in groups.items():
        complete = len(group) == len(seeds) and {r["config"]["seed"] for r in group} == set(seeds)
        score = math.inf
        if complete and all(r["status"] == "success" and r.get(metric) is not None and math.isfinite(r[metric]) for r in group):
            score = float(np.mean([np.log10(r[metric]+floor) for r in group]))
        scores.setdefault(f"{regime}/{name}", []).append((score, candidate))
    return {key: json.loads(min(values)[1]) if math.isfinite(min(values)[0]) else None
            for key, values in scores.items()}


def candidate_grid(protocol, stage, name):
    section = protocol[stage]
    if stage != "stochastic":
        return [{"name": name, "lr": lr} for lr in section["rates"][name]]
    knobs = {"adamw": ("betas", [(0.9, b) for b in section["beta2"]]),
             "momentum": ("momentum", section["momenta"]),
             "shampoo": ("refresh", section["refresh"])}
    key, values = knobs[name]
    # Reference Shampoo's actual option name is refresh_interval.
    if name == "shampoo":
        key = "refresh_interval"
    return [{"name": name, "lr": lr, key: v, "schedule": schedule}
            for lr, v, schedule in itertools.product(section["rates"][name], values, section["schedules"])]


def run_stage(destination, stage):
    path = Path(destination)
    state = json.loads((path/"campaign.json").read_text())
    if stage in state["stages"]:
        raise ValueError("stage already executed; preserve attempts in a new campaign")
    protocol = state["protocol"]
    if set(protocol["development_seeds"]) & set(protocol["evaluation_seeds"]) or (set(protocol["development_seeds"]) | set(protocol["evaluation_seeds"])) & {10, 11, 20, 21, 22, 23, 24}:
        raise ValueError("Sprint 3 seeds must be fresh and split-disjoint")
    set_seed(0, True, 1)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64 if stage != "neural" else torch.float32)
    start = time.perf_counter()
    deadline = start+max(0, protocol["max_experiment_seconds"]-state["experiment_seconds"])
    environment = state["environment"]
    # Reject source changes between campaign creation and stages.
    if environment_metadata("cpu")["source_sha256"] != environment["source_sha256"]:
        raise ValueError("source changed since campaign creation; create a new campaign")
    records = []

    def execute(config):
        if state["trials"] >= protocol["max_trials"]:
            raise ValueError("trial cap exceeded")
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        trial_path = path/"raw"/f"{stage}-{config_hash[:12]}-{uuid.uuid4().hex[:8]}"
        trial_path.mkdir()
        write_json(trial_path/"config.json", config)
        before = time.perf_counter()
        result = {"status": "budget_not_started"}
        if before < deadline:
            try:
                result = (neural_trial if stage == "neural" else quadratic_trial)(config, trial_path, deadline)
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                if hasattr(error, "partial_rows"):
                    write_json(trial_path/"partial_rows.json", error.partial_rows)
        result.update(config=config, config_sha256=config_hash, path=str(trial_path),
                      runtime_seconds=time.perf_counter()-before, source_sha256=environment["source_sha256"])
        # Budget-censored buffered rows are preserved; failed trials retain config/error.
        if "rows" in result:
            write_json(trial_path/"partial_rows.json", result.pop("rows"))
        write_json(trial_path/"metadata.json", result)
        records.append(result)
        state["trials"] += 1
        with (path/"trials.jsonl").open("a") as handle:
            handle.write(json.dumps(result, allow_nan=False)+"\n")
        if len(records) % 20 == 0:
            print(f"{stage}: {len(records)} attempts, {time.perf_counter()-start:.1f}s", flush=True)

    def config(regime, objective, candidate, seed, split):
        c = {"stage": stage, "regime": regime, "objective": objective, "optimizer": candidate,
             "seed": seed, "split": split, "steps": protocol["steps"], "tail_fraction": protocol["tail_fraction"],
             "rank": protocol["geometry"]["rank"], "history_window": protocol["geometry"]["history_window"]}
        if stage == "neural":
            model = regime.split("/")[0]
            c.update(protocol["neural"], model=model, hidden=protocol["neural"]["models"][model],
                     target_loss=protocol["neural"]["targets"][model],
                     data_root=str(Path(__file__).resolve().parents[3]/"data"))
        return c

    try:
        if stage == "replication":
            old = json.loads((Path(__file__).resolve().parents[3]/"results/sprint2/campaign-20261012/campaign.json").read_text())["protocol"]
            rates = {"adamw": 0.01, "momentum": 0.0001, "shampoo": 0.3}
            for name, lr in rates.items():
                for seed in old["evaluation_seeds"]:
                    c = trial_config(old, "noisy", old["regimes"]["noisy"], name, lr, seed, "replication_only", old["steps"])
                    result = run_trial(c, path, environment, deadline)
                    result.update(config=c, stage="replication")
                    with (path/"replication.jsonl").open("a") as handle:
                        handle.write(json.dumps(result, allow_nan=False)+"\n")
                    state["trials"] += 1
        else:
            if stage == "stochastic":
                regimes = protocol[stage]["regimes"]
            elif stage == "geometry":
                regimes = {f"{name}/{rotation}": {**obj, "condition": protocol[stage]["condition"], "rotation": rotation}
                           for name, obj in protocol[stage]["regimes"].items() for rotation in ("diagonal", "rotated")}
                regimes["isotropic/diagonal"] = {"condition": 1, "rotation": "diagonal", "noise_std": 1, "noise": "isotropic"}
            elif stage == "neural":
                regimes = {f"{model}/task700": {} for model in protocol[stage]["models"]}
            else:
                raise ValueError(stage)
            for regime, objective in regimes.items():
                for name in protocol[stage]["rates"]:
                    for candidate in candidate_grid(protocol, stage, name):
                        for seed in protocol["development_seeds"]:
                            execute(config(regime, objective, candidate, seed, "tuning"))
            chosen = select_candidates(records, protocol["development_seeds"],
                                       "validation_loss" if stage == "neural" else "tail_loss")
            write_json(path/"processed"/f"{stage}_selection.json", chosen)
            for regime, objective in regimes.items():
                for name in protocol[stage]["rates"]:
                    candidate = chosen[f"{regime}/{name}"]
                    if candidate is None:
                        continue
                    for seed in protocol["evaluation_seeds"]:
                        execute(config(regime, objective, candidate, seed, "evaluation"))
            if stage == "geometry":
                for noise in protocol[stage]["held_out_noise"]:
                    for rotation in ("diagonal", "rotated"):
                        objective = {"condition": protocol[stage]["condition"], "rotation": rotation, "noise_std": 1, "noise": noise}
                        for name in protocol[stage]["rates"]:
                            candidate = chosen[f"isotropic_noise/{rotation}/{name}"]
                            if candidate:
                                for seed in protocol["evaluation_seeds"]:
                                    execute(config(f"{noise}/{rotation}", objective, candidate, seed, "held_out"))
            if stage == "neural":
                for model in ("small", "deep"):
                    for name in protocol[stage]["rates"]:
                        candidate = chosen[f"{model}/task700/{name}"]
                        if candidate:
                            for seed in protocol["evaluation_seeds"]:
                                c = config(f"{model}/task701", {}, candidate, seed, "held_out")
                                c["data_seed"] = protocol[stage]["held_out_data_seed"]
                                execute(c)
    finally:
        torch.set_default_dtype(old_dtype)
        state["experiment_seconds"] += time.perf_counter()-start
        state["stages"].append(stage)
        write_json(path/"campaign.json", state)
    return records
