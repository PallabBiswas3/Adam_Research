"""Bounded, split-safe noise/curvature experiments using existing optimizers."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader

from ..datasets.synthetic import supervised_splits
from ..geometry.noise_curvature import (RandomizedCurvature, autograd_hvp,
    basis_metrics, central_gradient_difference, paired_noise_factor, pca_basis)
from ..models.mlp import MLP
from ..utils.hardware import environment_metadata
from ..utils.logging import write_json
from ..utils.seeds import set_seed
from .sprint2 import create_campaign
from .sprint3 import matched_problem, optimizer_for, tensor_bytes


def schedule(name, step, steps):
    if name == "constant":
        return 1.
    if name == "cosine":
        return .01+.99*(1+math.cos(math.pi*step/max(steps-1, 1)))/2
    if name == "conservative":
        return 1/(1+9*step/max(steps-1, 1))  # ends at 10%, no noise-change knowledge
    raise ValueError(name)


def noise_scale(regime, step, steps, protocol):
    high, low = protocol["noise_high"], protocol["noise_low"]
    if regime == "constant":
        return high
    before, after = (low, high) if regime == "rising" else (high, low)
    if regime not in {"rising", "falling"}:
        raise ValueError(regime)
    return before if step < steps//2 else after


def save_rows(path, name, rows):
    if not rows:
        return
    with (path/name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_trial(config, path, deadline, protocol):
    seed, steps = config["seed"], config["steps"]
    set_seed(seed, True, 1)
    h, x, factor = matched_problem({"condition": protocol["condition"],
        "rotation": config["rotation"], "noise": "isotropic"}, seed)
    p = nn.Parameter(x.reshape(4, 4))
    optimizer = optimizer_for([p], config["optimizer"])
    rng = torch.Generator().manual_seed(seed+700)
    rows, status, error = [], "success", None
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for step in range(steps):
                if step % 32 == 0 and time.perf_counter() >= deadline:
                    status = "budget_censored"
                    break
                scale = noise_scale(config["regime"], step, steps, protocol)
                gradient = h@p.flatten()+scale*(factor@torch.randn(16, generator=rng, dtype=p.dtype))
                optimizer.param_groups[0]["lr"] = config["optimizer"]["lr"]*schedule(
                    config["optimizer"]["schedule"], step, steps)
                p.grad = gradient.reshape_as(p)
                optimizer.step()
                loss = float(p.flatten()@h@p.flatten()/2)
                if not math.isfinite(loss) or loss > 1e100:
                    raise FloatingPointError("nonfinite/divergent quadratic loss")
                rows.append({"step": step+1, "loss": loss,
                    "seconds": time.perf_counter()-start, "noise_std": scale,
                    "lr": optimizer.param_groups[0]["lr"]})
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter()-start
    save_rows(path, "trajectory.csv", rows)
    result = {"status": status, "error": error, "steps_completed": len(rows),
        "loop_seconds": elapsed, "state_bytes": tensor_bytes(optimizer.state)}
    if status == "success":
        tail = np.array([r["loss"] for r in rows[int(steps*(1-protocol["tail_fraction"])):]])
        w, target = protocol["target_window"], None
        for end in range(w-1, len(rows)):
            if max(r["loss"] for r in rows[end-w+1:end+1]) <= protocol["target_loss"]:
                target = rows[end]["seconds"]
                break
        result.update(tail_loss=float(tail.mean()), final_loss=rows[-1]["loss"],
            tail_sd=float(tail.std(ddof=1)), recovery_loss=float(np.mean([
                r["loss"] for r in rows[steps//2:steps//2+200]])),
            sustained_target_seconds=target, target_censored=target is None)
    return result


def select_policies(records, protocol):
    if any(r["config"]["split"] != "tuning" for r in records):
        raise ValueError("evaluation must never enter selection")
    expected = {(seed, regime) for seed in protocol["development_seeds"] for regime in ("rising", "falling")}
    groups = {}
    for r in records:
        c = r["config"]
        groups.setdefault((c["optimizer"]["name"], c["candidate_id"]), []).append(r)
    candidates = []
    for (method, candidate_id), group in groups.items():
        complete = len(group) == len(expected) and {
            (r["config"]["seed"], r["config"]["regime"]) for r in group} == expected
        score = None
        if complete and all(r["status"] == "success" for r in group):
            values = np.array([math.log10(r["tail_loss"]+protocol["selection_floor"]) for r in group])
            score = float(values.mean()+protocol["selection_sd_penalty"]*values.std(ddof=1))
        candidates.append({"method": method, "candidate_id": candidate_id,
            "optimizer": group[0]["config"]["optimizer"], "score": score,
            "valid": score is not None, "n_success": sum(r["status"] == "success" for r in group)})
    selection = {}
    for method in protocol["candidates"]:
        valid = [r for r in candidates if r["method"] == method and r["valid"]]
        if not valid:
            raise ValueError(f"no complete candidate for {method}")
        selection[method] = min(valid, key=lambda r: (r["score"], r["candidate_id"]))
    return selection, candidates


def sample_covariance(rows):
    centered = rows-rows.mean(0)
    return centered.T@centered/len(rows)


def synthetic_factor(mode, h, seed, step, steps):
    d = len(h)
    hu = torch.linalg.eigh(h).eigenvectors
    if mode == "isotropic_h":
        return torch.eye(d, dtype=h.dtype)
    if mode == "switching":
        mode = "aligned" if step < steps//2 else "antialigned"
    values = torch.logspace(0, 2, d, dtype=h.dtype)
    values *= d/values.sum()
    if mode == "antialigned":
        values = values.flip(0)
    basis = hu
    if mode == "misaligned":
        rng = torch.Generator().manual_seed(seed+177)
        basis = torch.linalg.qr(torch.randn(d, d, generator=rng, dtype=h.dtype)).Q
    if mode == "partial":
        mixing = torch.eye(d, dtype=h.dtype)
        for i in range(d//2):
            j = d-1-i
            mixing[i, i] = mixing[j, j] = 2**-.5
            mixing[i, j], mixing[j, i] = 2**-.5, -2**-.5
        basis = hu@mixing
    return basis@torch.diag(values.sqrt())


def compare_estimators(h, history, positions, first, second, operator,
                       shared_fd, independent_fd, settings, seed):
    """Small dense analysis controls; standalone probes remain matrix-free."""
    rank = settings["rank"]
    rows = []
    covariance = sample_covariance(history)
    start = time.perf_counter()
    f = paired_noise_factor(first, second)
    noise_estimate = f@f.T
    noise_seconds = time.perf_counter()-start
    controls = {
        "history": lambda: pca_basis(history, rank),
        "centered_history": lambda: pca_basis(history, rank, True),
        "paired_noise": lambda: pca_basis((first-second), rank),
        "noise_subtraction": lambda: torch.linalg.eigh(covariance-noise_estimate).eigenvectors[:, -rank:],
        "oracle_subtraction": lambda: torch.linalg.eigh(h@sample_covariance(positions)@h).eigenvectors[:, -rank:],
        "diagonal": lambda: torch.eye(len(h), dtype=h.dtype)[:, torch.argsort(h.diag())[-rank:]],
        "oracle": lambda: torch.linalg.eigh(h).eigenvectors[:, -rank:],
    }
    rng = torch.Generator().manual_seed(seed+222)
    controls["random"] = lambda: torch.linalg.qr(torch.randn(len(h), rank, generator=rng, dtype=h.dtype)).Q
    bases = {}
    for method, compute in controls.items():
        start = time.perf_counter()
        basis = compute()
        duration = time.perf_counter()-start
        bases[method] = basis
        rows.append({"method": method, **basis_metrics(h, basis),
            "refresh_seconds": duration+(noise_seconds if method == "noise_subtraction" else 0),
            "operator_calls": 0, "extra_gradient_calls": 2*len(first) if method in {"paired_noise", "noise_subtraction"} else 0,
            "retained_bytes": basis.numel()*basis.element_size(), "actual_reconstruction_error": None,
            "workspace_tensor_bytes_estimate": None})
    for method, op in (("hvp_ritz", operator), ("shared_fd_ritz", shared_fd), ("independent_fd_ritz", independent_fd)):
        estimator = RandomizedCurvature(rank, settings["oversample"], settings["power_steps"], settings["refresh_frequency"])
        result = estimator.refresh(op, torch.zeros(len(h), dtype=h.dtype), seed+223)
        basis = result.basis
        bases[method] = basis
        reconstruction = basis@torch.diag(result.eigenvalues)@basis.T
        rows.append({"method": method, **basis_metrics(h, basis),
            "refresh_seconds": result.refresh_seconds, "operator_calls": result.operator_calls,
            "extra_gradient_calls": 2*result.operator_calls if "fd" in method else 0,
            "retained_bytes": result.retained_bytes,
            "workspace_tensor_bytes_estimate": result.workspace_tensor_bytes_estimate,
            "actual_reconstruction_error": float((h-reconstruction).norm()/h.norm().clamp_min(1e-30))})
    return rows, bases, noise_estimate


def estimator_trial(config, path, deadline, protocol):
    set_seed(config["seed"], True, 1)
    settings = protocol["estimators"]
    seed, steps, mode = config["seed"], settings["steps"], config["regime"]
    h, x, _ = matched_problem({"condition": 1 if mode == "isotropic_h" else protocol["condition"],
                              "rotation": "rotated"}, seed)
    p = nn.Parameter(x.reshape(4, 4))
    optimizer = optimizer_for([p], {"name": "adamw", "lr": .03, "betas": [.9, .99], "eps": 1e-8})
    rng = torch.Generator().manual_seed(seed+700)
    probe_rng = torch.Generator().manual_seed(seed+20000)
    gradients, positions, rows, metrics = [], [], [], []
    start, train_seconds, diagnostic_seconds = time.perf_counter(), 0., 0.
    for step in range(steps):
        if step % 32 == 0 and time.perf_counter() >= deadline:
            save_rows(path, "trajectory.csv", rows)
            save_rows(path, "estimators.csv", metrics)
            return {"status": "budget_censored", "steps_completed": step}
        begin = time.perf_counter()
        factor = synthetic_factor(mode, h, seed, step, steps)
        noise_factor = protocol["noise_high"]*factor
        with torch.no_grad():
            theta = p.flatten().clone()
            gradient = h@theta+noise_factor@torch.randn(16, generator=rng, dtype=h.dtype)
            gradients.append(gradient)
            positions.append(theta)
            gradients, positions = gradients[-settings["history_window"]:], positions[-settings["history_window"]:]
            p.grad = gradient.reshape_as(p)
            optimizer.param_groups[0]["lr"] = .03*schedule("conservative", step, steps)
            optimizer.step()
            loss = float(p.flatten()@h@p.flatten()/2)
        train_seconds += time.perf_counter()-begin
        rows.append({"step": step+1, "loss": loss, "seconds": time.perf_counter()-start})
        if step+1 not in settings["phases"]:
            continue
        begin = time.perf_counter()
        point = p.flatten().detach().clone()
        n = settings["paired_samples"]
        draw_start = time.perf_counter()
        first = (h@point)[None, :]+torch.randn(n, 16, generator=probe_rng, dtype=h.dtype)@noise_factor.T
        second = (h@point)[None, :]+torch.randn(n, 16, generator=probe_rng, dtype=h.dtype)@noise_factor.T
        draw_seconds = time.perf_counter()-draw_start
        def op(v):
            return autograd_hvp(lambda z: z@h@z/2, point, v)
        fixed_noise = noise_factor@torch.randn(16, generator=probe_rng, dtype=h.dtype)
        def shared(v):
            return central_gradient_difference(lambda z: h@z+fixed_noise, point, v, settings["delta"])
        def independent(v):
            return central_gradient_difference(lambda z: h@z+noise_factor@torch.randn(16, generator=probe_rng, dtype=h.dtype),
                                               point, v, settings["delta"])
        comparisons, bases, noise_estimate = compare_estimators(h, torch.stack(gradients), torch.stack(positions),
            first, second, op, shared, independent, settings, seed)
        sigma = noise_factor@noise_factor.T
        true_noise_basis = torch.linalg.eigh(sigma).eigenvectors[:, -settings["rank"]:]
        for row in comparisons:
            if row["method"] in {"paired_noise", "noise_subtraction"}:
                row["refresh_seconds"] += draw_seconds
            row.update(phase=step+1, hessian_is_isotropic=mode == "isotropic_h",
                noise_overlap=float((bases[row["method"]].T@true_noise_basis).square().sum()/settings["rank"]),
                noise_covariance_relative_error=float((noise_estimate-sigma).norm()/sigma.norm()),
                subtraction_min_eigenvalue=float(torch.linalg.eigvalsh(sample_covariance(torch.stack(gradients))-noise_estimate).min()),
                signal_noise_norm_ratio=float((h@sample_covariance(torch.stack(positions))@h).norm()/sigma.norm()))
            metrics.append(row)
        # Keep the actual observations and small oracle objects, not just aggregates.
        np.savez_compressed(path/f"snapshot-{step+1}.npz", h=h.numpy(), sigma=sigma.numpy(),
            history=torch.stack(gradients).numpy(), positions=torch.stack(positions).numpy(),
            paired_first=first.numpy(), paired_second=second.numpy(),
            **{f"basis_{k}": v.numpy() for k, v in bases.items()})
        diagnostic_seconds += time.perf_counter()-begin
    save_rows(path, "trajectory.csv", rows)
    save_rows(path, "estimators.csv", metrics)
    return {"status": "success", "steps_completed": steps, "loop_seconds": time.perf_counter()-start,
        "train_seconds": train_seconds, "diagnostic_seconds": diagnostic_seconds,
        "state_bytes": tensor_bytes(optimizer.state), "metrics": metrics}


def layer_objects(model, inputs, labels):
    """Binary two-layer ReLU block GGN/features at a fixed parameter point.

    Layer block Hessians equal these GGN blocks away from ReLU kinks. Cross-layer
    Hessian residuals are NOT included. Population oracle uses training only.
    """
    first, last = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        pre = first(inputs)
        activation = pre.relu()
        logits = last(activation)
        probability = logits.softmax(1)[:, 1]
        output_difference = last.weight[1]-last.weight[0]
        jhidden = (pre > 0)*output_difference
        zfirst = torch.cat([(jhidden[:, :, None]*inputs[:, None, :]).flatten(1), jhidden], 1)
        zlast = torch.cat([-activation, activation, -torch.ones(len(inputs), 1), torch.ones(len(inputs), 1)], 1)
        # Flatten order matches weight.reshape(-1), then bias.
        objects = []
        for index, (z, layer, a) in enumerate(((zfirst, first, inputs), (zlast, last, activation))):
            w = probability*(1-probability)
            ggn = z.T@(w[:, None]*z)/len(z)
            per_example = (probability-labels)[:, None]*z
            point = torch.cat([layer.weight.flatten(), layer.bias])
            objects.append({"layer": index, "z": z, "w": w, "ggn": ggn,
                "per_example": per_example, "activation_covariance": sample_covariance(a),
                "activation_moment": a.T@a/len(a), "point": point,
                "prefix": "network.1" if index == 0 else "network.3",
                "weight_shape": layer.weight.shape})
        return objects


def neural_probe(model, train, history, epoch, seed, settings, path):
    x, y = train.tensors
    objects = layer_objects(model, x, y)
    rng = torch.Generator().manual_seed(seed+30000+epoch*111)
    first_draws = torch.randint(len(x), (settings["paired_samples"], settings["batch_size"]), generator=rng)
    second_draws = torch.randint(len(x), first_draws.shape, generator=rng)
    # Real autograd gradients at EXACTLY the same parameters for every pair.
    params = list(model.parameters())
    samples = [[], []]
    begin = time.perf_counter()
    for group, indices in enumerate((first_draws, second_draws)):
        for batch_indices in indices:
            loss = nn.functional.cross_entropy(model(x[batch_indices]), y[batch_indices])
            grad = torch.autograd.grad(loss, params)
            samples[group].append([torch.cat([grad[0].flatten(), grad[1]]).detach(),
                                   torch.cat([grad[2].flatten(), grad[3]]).detach()])
    gradient_seconds = time.perf_counter()-begin
    base_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    probe_indices = torch.randint(len(x), (settings["probe_batch_size"],), generator=rng)
    comparisons = []
    for obj in objects:
        layer = obj["layer"]
        first = torch.stack([g[layer] for g in samples[0]])
        second = torch.stack([g[layer] for g in samples[1]])
        hist = torch.stack(history[layer][-64:]) if history[layer] else first
        # No excitation matrix is available for historical nonlinear GGN;
        # oracle_subtraction is excluded rather than claiming H Cov(theta) H.
        positions = torch.zeros_like(hist)
        point = obj["point"]
        shape = obj["weight_shape"]
        count = math.prod(shape)
        def loss_at(value, indices):
            mapping = dict(base_parameters)
            mapping[obj["prefix"]+".weight"] = value[:count].reshape(shape)
            mapping[obj["prefix"]+".bias"] = value[count:]
            return nn.functional.cross_entropy(functional_call(model, mapping, x[indices]), y[indices])
        def hvp(v):
            return autograd_hvp(lambda value: loss_at(value, probe_indices), point, v)
        def gradient(value, indices):
            value = value.detach().requires_grad_(True)
            return torch.autograd.grad(loss_at(value, indices), value)[0].detach()
        def shared(v):
            return central_gradient_difference(lambda value: gradient(value, probe_indices), point, v, .01)
        def independent(v):
            return central_gradient_difference(lambda value: gradient(value,
                torch.randint(len(x), (settings["probe_batch_size"],), generator=rng)), point, v, .01)
        comparisons_layer, bases, estimated_noise = compare_estimators(obj["ggn"].double(), hist.double(), positions.double(),
            first.double(), second.double(), lambda v: hvp(v.float()).double(),
            lambda v: shared(v.float()).double(), lambda v: independent(v.float()).double(), settings, seed+layer)
        # Frozen-parameter gradient-covariance is a distinct control from history.
        frozen_basis = pca_basis(first.double(), settings["rank"], True)
        comparisons_layer.append({"method": "frozen_gradient_covariance", **basis_metrics(obj["ggn"].double(), frozen_basis),
            "refresh_seconds": gradient_seconds, "operator_calls": 0,
            "extra_gradient_calls": settings["paired_samples"], "retained_bytes": frozen_basis.numel()*8,
            "actual_reconstruction_error": None, "workspace_tensor_bytes_estimate": None})
        g = obj["per_example"].double()
        noise_oracle = sample_covariance(g)/settings["batch_size"]
        for row in comparisons_layer:
            if row["method"] == "oracle_subtraction":
                continue
            if row["method"] in {"paired_noise", "noise_subtraction"}:
                row["refresh_seconds"] += gradient_seconds  # shared across layers, conservative attribution
            row.update(epoch=epoch, layer=layer, history_kind="frozen_initial" if epoch == 0 else "training_history",
                noise_covariance_relative_error=float((estimated_noise-noise_oracle).norm()/noise_oracle.norm().clamp_min(1e-30)),
                activation_covariance_trace=float(torch.trace(obj["activation_covariance"])),
                activation_covariance_condition=float(torch.linalg.cond(obj["activation_covariance"])),
                gradient_sampling_seconds=gradient_seconds,
                parameter_dimension=len(point), oracle_trace=float(torch.trace(obj["ggn"])))
            comparisons.append(row)
        np.savez_compressed(path/f"neural-{epoch}-layer{layer}.npz", ggn=obj["ggn"].numpy(),
            activation_covariance=obj["activation_covariance"].numpy(), activation_moment=obj["activation_moment"].numpy(),
            noise_oracle=noise_oracle.numpy(), history=hist.numpy(), first=first.numpy(), second=second.numpy(),
            **{f"basis_{k}": v.numpy() for k, v in bases.items() if k != "oracle_subtraction"})
    return comparisons


def neural_trial(config, path, deadline, protocol):
    set_seed(config["seed"], True, 1)
    settings = {**protocol["estimators"], **protocol["neural"]}
    train, val, _ = supervised_splits({**settings["dataset"], "data_seed": config["task_seed"]}, config["seed"])
    model = MLP(settings["dataset"]["dimension"], 2, settings["hidden"], initialization="kaiming")
    optimizer = optimizer_for(model.parameters(), settings["optimizer"])
    loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]+900))
    history, rows, diagnostics = [[], []], [], []
    steps, train_seconds, diagnostic_seconds = 0, 0., 0.
    for epoch in range(settings["epochs"]+1):
        if time.perf_counter() >= deadline:
            save_rows(path, "trajectory.csv", rows)
            save_rows(path, "estimators.csv", diagnostics)
            return {"status": "budget_censored", "epochs_completed": epoch-1}
        if epoch > 0:
            start = time.perf_counter()
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(bx), by)
                loss.backward()
                parameters = list(model.parameters())
                for layer in (0, 1):
                    history[layer].append(torch.cat([parameters[2*layer].grad.flatten(), parameters[2*layer+1].grad]).detach().clone())
                    history[layer] = history[layer][-64:]
                optimizer.param_groups[0]["lr"] = settings["optimizer"]["lr"]*schedule("cosine", steps, settings["epochs"]*len(loader))
                optimizer.step()
                steps += 1
            train_seconds += time.perf_counter()-start
        with torch.no_grad():
            value = float(nn.functional.cross_entropy(model(val.tensors[0]), val.tensors[1]))
        rows.append({"epoch": epoch, "validation_loss": value, "train_seconds": train_seconds})
        if epoch in settings["phases"]:
            start = time.perf_counter()
            diagnostics += neural_probe(model, train, history, epoch, config["seed"], settings, path)
            diagnostic_seconds += time.perf_counter()-start
    save_rows(path, "trajectory.csv", rows)
    save_rows(path, "estimators.csv", diagnostics)
    return {"status": "success", "epochs_completed": settings["epochs"], "train_seconds": train_seconds,
        "diagnostic_seconds": diagnostic_seconds, "mean_step_seconds": train_seconds/steps,
        "validation_loss": rows[-1]["validation_loss"], "state_bytes": tensor_bytes(optimizer.state),
        "metrics": diagnostics}


def earlier_inventory(root):
    inventory = []
    for sprint in ("sprint2", "sprint3"):
        for item in sorted((root/"results"/sprint).rglob("*")):
            if item.is_file():
                inventory.append({"path": item.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(item.read_bytes()).hexdigest(), "bytes": item.stat().st_size})
    return inventory


def initialize(protocol, destination):
    dev, evaluation = set(protocol["development_seeds"]), set(protocol["evaluation_seeds"])
    if dev != set(range(50, 60)) or evaluation != set(range(60, 70)) or dev & evaluation:
        raise ValueError("required fresh development/evaluation seeds 50–59 / 60–69")
    if protocol["device"] != "cpu" or protocol["max_trials"] > 1000 or protocol["max_experiment_seconds"] > 1800:
        raise ValueError("Sprint 4 CPU budget limits exceeded")
    if protocol["dimension"] != 16 or protocol["max_experiment_seconds"] <= 0:
        raise ValueError("this matched quadratic runner requires dimension 16 and positive budget")
    if any(len(v) != 8 for v in protocol["candidates"].values()):
        raise ValueError("equal eight-candidate budgets required")
    path = create_campaign(protocol, destination)
    inventory = earlier_inventory(Path(__file__).resolve().parents[3])
    write_json(path/"raw"/"earlier_inventory.json", inventory)
    write_json(path/"preservation_before.json", {"files": len(inventory),
        "aggregate_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()})
    return path


def run_stage(destination, stage):
    path = Path(destination)
    state = json.loads((path/"campaign.json").read_text())
    protocol = state["protocol"]
    if stage in state["stages"] or (path/f"{stage}.started.json").exists():
        raise ValueError("stage already started; do not overwrite; use a new campaign")
    if environment_metadata("cpu")["source_sha256"] != state["environment"]["source_sha256"]:
        raise ValueError("source changed since freeze; create new campaign")
    write_json(path/f"{stage}.started.json", {"stage": stage, "source_sha256": state["environment"]["source_sha256"]})
    set_seed(0, True, 1)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32 if stage == "neural" else torch.float64)
    start = time.perf_counter()
    deadline = start+max(0, protocol["max_experiment_seconds"]-state["experiment_seconds"])
    records = []

    def execute(config):
        if state["trials"] >= protocol["max_trials"]:
            raise ValueError("maximum trial count reached")
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        trial = path/"raw"/f"{stage}-{digest[:16]}"
        trial.mkdir(exist_ok=False)
        write_json(trial/"config.json", config)
        before = time.perf_counter()
        result = {"status": "budget_not_started"}
        if before < deadline:
            try:
                function = neural_trial if stage == "neural" else estimator_trial if stage == "estimators" else baseline_trial
                result = function(config, trial, deadline, protocol)
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        # Detailed estimator rows live in CSV instead of inflating the trial index.
        result.pop("metrics", None)
        result.update(config=config, config_sha256=digest, path=trial.relative_to(path).as_posix(),
            runtime_seconds=time.perf_counter()-before, source_sha256=state["environment"]["source_sha256"])
        write_json(trial/"metadata.json", result)
        records.append(result)
        with (path/"trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False)+"\n")
        state["trials"] += 1
        checkpoint = {**state, "experiment_seconds": state["experiment_seconds"]+time.perf_counter()-start,
                      "interrupted_stage": stage}
        write_json(path/"campaign.json", checkpoint)
        if len(records) % 20 == 0:
            print(f"{stage}: {len(records)} attempts / {time.perf_counter()-start:.1f}s", flush=True)

    def baseline_config(seed, method, candidate_id, options, regime, rotation, split, steps=None):
        return {"stage": stage, "seed": seed, "optimizer": {"name": method, **options},
            "candidate_id": candidate_id, "regime": regime, "rotation": rotation,
            "split": split, "steps": steps or protocol["steps"]}
    try:
        if stage == "pilot":
            for method in protocol["candidates"]:
                options = protocol["candidates"][method][0]
                execute(baseline_config(50, method, 0, options, "falling", "rotated", "pilot", 128))
        elif stage == "tuning":
            # Interleave methods and seeds to reduce temporal timing confounding.
            for candidate_id in range(8):
                for seed in protocol["development_seeds"]:
                    for regime in ("rising", "falling"):
                        for method, candidates in protocol["candidates"].items():
                            execute(baseline_config(seed, method, candidate_id, candidates[candidate_id], regime,
                                "diagonal" if seed % 2 == 0 else "rotated", "tuning"))
            selection, scores = select_policies(records, protocol)
            write_json(path/"processed"/"selection.json", selection)
            write_json(path/"processed"/"development_scores.json", scores)
        elif stage == "evaluation":
            selection = json.loads((path/"processed"/"selection.json").read_text())
            for seed in protocol["evaluation_seeds"]:
                for regime in ("rising", "falling", "constant"):
                    for rotation in ("diagonal", "rotated"):
                        for method, chosen in selection.items():
                            execute(baseline_config(seed, method, chosen["candidate_id"], chosen["optimizer"], regime, rotation, "evaluation"))
        elif stage == "estimators":
            for seed in protocol["evaluation_seeds"]:
                for mode in protocol["estimators"]["noise_modes"]:
                    execute({"stage": stage, "seed": seed, "regime": mode, "split": "evaluation"})
        elif stage == "neural":
            for seed in protocol["evaluation_seeds"]:
                for task_seed in protocol["neural"]["task_seeds"]:
                    execute({"stage": stage, "seed": seed, "task_seed": task_seed, "split": "evaluation"})
        else:
            raise ValueError(stage)
        state["stages"].append(stage)
    finally:
        state["experiment_seconds"] += time.perf_counter()-start
        write_json(path/"campaign.json", state)
        torch.set_default_dtype(previous_dtype)
    print(f"Completed {stage}: {state['trials']} total attempts, {state['experiment_seconds']:.2f}s", flush=True)
    return records
