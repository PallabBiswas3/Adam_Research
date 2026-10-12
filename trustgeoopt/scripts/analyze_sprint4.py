"""Postprocessing and whole-seed comparisons; no optimizer tuning."""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

script = Path(__file__).resolve()
project = next(p for p in script.parents if (p/"pyproject.toml").exists())
sys.path.insert(0, str(project/"src"))
from trustgeoopt.training.sprint4 import select_policies, earlier_inventory
from trustgeoopt.utils.logging import write_json as _write_json


def write_json(path, value):
    """Normalize pandas/numpy scalar types without changing numerical values."""
    normalized = json.loads(json.dumps(value, default=lambda x: x.item()))
    _write_json(path, normalized)


def interval(differences, rng, replicates):
    values = np.asarray(differences, dtype=float)
    sampled = values[rng.integers(0, len(values), (replicates, len(values)))].mean(1)
    return {"n_pairs": len(values), "mean_difference": float(values.mean()),
        "ci95": np.quantile(sampled, [.025, .975]).tolist(),
        "paired_differences": values.tolist()}


def compare_groups(frame, keys, metric, baseline, rng, replicates, log=False):
    contrasts = []
    for key, group in frame.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        base = group[group.method == baseline].set_index("seed")[metric]
        for method, rows in group.groupby("method"):
            if method == baseline:
                continue
            other = rows.set_index("seed")[metric]
            paired = pd.concat([other.rename("other"), base.rename("base")], axis=1).dropna()
            if len(paired) == 0:
                continue
            differences = (np.log10(paired.other+1e-12)-np.log10(paired.base+1e-12)) if log else paired.other-paired.base
            contrasts.append({**dict(zip(keys, key)), "method": method, "baseline": baseline,
                "metric": metric, "scale": "log10 ratio" if log else "difference",
                "paired_seeds": paired.index.tolist(), "complete_ten_seed_pairs": len(paired) == 10,
                **interval(differences, rng, replicates)})
    return contrasts


def analyze(path):
    started = time.perf_counter()
    state = json.loads((path/"campaign.json").read_text())
    protocol = state["protocol"]
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    replicates = protocol["bootstrap_replicates"]
    records = [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]
    out = path/"processed"
    if (out/"analysis_provenance.json").exists():
        raise ValueError("already analyzed; preserve results")
    write_json(out/"status_counts.json", dict(Counter(r["status"] for r in records)))
    write_json(out/"stage_costs.json", {stage: {"attempts": sum(r["config"]["stage"] == stage for r in records),
        "sum_trial_seconds": sum(r["runtime_seconds"] for r in records if r["config"]["stage"] == stage)}
        for stage in sorted({r["config"]["stage"] for r in records})})
    tuning = [r for r in records if r["config"]["stage"] == "tuning"]
    selection, scores = select_policies(tuning, protocol)
    assert selection == json.loads((out/"selection.json").read_text())
    # Development-only resampling: both regimes from the same seed remain together.
    selection_frequency = []
    dev = protocol["development_seeds"]
    for method in protocol["candidates"]:
        values = []
        for candidate_id in range(8):
            runs = [r for r in tuning if r["config"]["optimizer"]["name"] == method and r["config"]["candidate_id"] == candidate_id]
            mapping = {(r["config"]["seed"], r["config"]["regime"]): r for r in runs}
            if len(runs) != 20 or any(r["status"] != "success" for r in runs):
                values.append(None)
            else:
                values.append(np.array([[np.log10(mapping[s, regime]["tail_loss"]+1e-12)
                    for regime in ("rising", "falling")] for s in dev]))
        counts = Counter()
        for _ in range(replicates):
            indices = rng.integers(0, 10, 10)
            sampled_scores = [float(a[indices].mean()+protocol["selection_sd_penalty"]*a[indices].std(ddof=1))
                              if a is not None else np.inf for a in values]
            counts[int(np.argmin(sampled_scores))] += 1
        selection_frequency.append({"method": method, "chosen_id": selection[method]["candidate_id"],
            "bootstrap_candidate_counts": dict(counts), "replicates": replicates})
    write_json(out/"selection_uncertainty.json", selection_frequency)
    boundary = []
    for method, selected in selection.items():
        rate = selected["optimizer"]["lr"]
        rates = [r["lr"] for r in protocol["candidates"][method]]
        boundary.append({"method": method, "selected_rate": rate, "lower": min(rates), "upper": max(rates),
            "at_rate_boundary": rate in {min(rates), max(rates)},
            "note": "joint candidate design; beta/epsilon boundaries remain, not global optimization"})
    write_json(out/"grid_boundaries.json", boundary)
    baseline = []
    for r in records:
        c = r["config"]
        if c["stage"] == "evaluation":
            baseline.append({"seed": c["seed"], "regime": c["regime"], "rotation": c["rotation"],
                "method": c["optimizer"]["name"], "path": r["path"], **{k: v for k, v in r.items() if k not in {"config", "path"}}})
    frame = pd.DataFrame(baseline)
    # Explicit fixed common-time horizon per noise regime, all rotations/methods/seeds.
    horizons = frame[frame.status == "success"].groupby("regime").loop_seconds.min().to_dict()
    for i, row in frame.iterrows():
        if row.status != "success":
            continue
        trajectory = pd.read_csv(path/row.path/"trajectory.csv")
        horizon = horizons[row.regime]
        available = trajectory[trajectory.seconds <= horizon]
        frame.loc[i, "common_time_horizon"] = horizon
        frame.loc[i, "common_time_steps"] = len(available)
        frame.loc[i, "common_time_loss"] = available.loss.tail(100).mean()
        target = row.get("sustained_target_seconds", np.nan)
        reached = pd.notna(target) and target <= horizon
        frame.loc[i, "common_time_target_reached"] = int(reached)
        frame.loc[i, "restricted_time_to_target"] = target if reached else horizon
    frame.to_csv(out/"baseline_runs.csv", index=False)
    summaries = []
    for key, group in frame.groupby(["regime", "rotation", "method"]):
        good = group[group.status == "success"]
        summaries.append({**dict(zip(["regime", "rotation", "method"], key)), "attempts": len(group), "success": len(good),
            **{f"mean_{metric}": good[metric].mean() for metric in ["tail_loss", "final_loss", "tail_sd", "recovery_loss",
                "runtime_seconds", "loop_seconds", "state_bytes", "common_time_loss", "restricted_time_to_target"]},
            "run_sd_tail_loss": good.tail_loss.std(ddof=1),
            "sustained_target_reached": int(good.sustained_target_seconds.notna().sum()),
            "common_time_target_reached": int(good.common_time_target_reached.sum()),
            "common_time_horizon": horizons[key[0]]})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/"baseline_summary.csv", index=False)
    contrasts = []
    for baseline_method in ("adamw", "momentum"):
        for metric in ("tail_loss", "common_time_loss", "runtime_seconds", "recovery_loss"):
            contrasts += compare_groups(frame, ["regime", "rotation"], metric, baseline_method, rng, replicates, True)
        contrasts += compare_groups(frame, ["regime", "rotation"], "restricted_time_to_target", baseline_method, rng, replicates)
    # Rotation pair contrasts under the SAME frozen policy.
    rotation_frame = frame.copy()
    rotation_frame["optimizer"] = rotation_frame.method
    rotation_frame["method"] = rotation_frame.rotation
    contrasts += compare_groups(rotation_frame, ["regime", "optimizer"], "tail_loss", "diagonal", rng, replicates, True)
    write_json(out/"baseline_comparisons.json", contrasts)
    synthetic, neural = [], []
    for r in records:
        c = r["config"]
        if c["stage"] not in {"estimators", "neural"} or not (path/r["path"]/"estimators.csv").exists():
            continue
        measurements = pd.read_csv(path/r["path"]/"estimators.csv")
        for row in measurements.to_dict("records"):
            row.update(seed=c["seed"], run_status=r["status"], run_path=r["path"])
            if c["stage"] == "estimators":
                row.update(regime=c["regime"])
                synthetic.append(row)
            else:
                row.update(task_seed=c["task_seed"], mean_step_seconds=r.get("mean_step_seconds"),
                           train_seconds=r.get("train_seconds"), total_diagnostic_seconds=r.get("diagnostic_seconds"))
                row["amortized_refresh_over_step"] = row["refresh_seconds"]/(100*r["mean_step_seconds"])
                neural.append(row)
    synthetic, neural = pd.DataFrame(synthetic), pd.DataFrame(neural)
    synthetic.to_csv(out/"synthetic_estimators.csv", index=False)
    neural.to_csv(out/"neural_estimators.csv", index=False)
    estimator_summaries = []
    estimator_contrasts = []
    for stage, measurements, keys in (("synthetic", synthetic, ["regime", "phase"]),
                                      ("neural", neural, ["task_seed", "epoch", "layer"])):
        metrics = ["capture", "overlap", "oracle_calibrated_error", "refresh_seconds", "retained_bytes",
                   "noise_covariance_relative_error", "actual_reconstruction_error"]
        if stage == "neural":
            metrics += ["amortized_refresh_over_step"]
        for key, group in measurements.groupby(keys+["method"]):
            estimator_summaries.append({"stage": stage, **dict(zip(keys+["method"], key)), "n_runs": len(group),
                **{f"mean_{metric}": float(group[metric].mean()) if group[metric].notna().any() else None for metric in metrics},
                **{f"sd_{metric}": float(group[metric].std(ddof=1)) if group[metric].notna().sum() > 1 else None
                   for metric in ["capture", "overlap", "noise_covariance_relative_error"]}})
        inferential = measurements if stage == "neural" else measurements[measurements.regime != "isotropic_h"]
        for control in ("centered_history", "random"):
            for metric in ("capture", "overlap"):
                for contrast in compare_groups(inferential, keys, metric, control, rng, replicates):
                    estimator_contrasts.append({"stage": stage, **contrast})
    pd.DataFrame(estimator_summaries).to_csv(out/"estimator_summary.csv", index=False)
    write_json(out/"estimator_comparisons.json", estimator_contrasts)
    # Gate: mean layer capture at final epoch, paired by complete seed.
    endpoint = neural[neural.epoch == protocol["neural"]["epochs"]].groupby(["task_seed", "seed", "method"])["capture"].mean().reset_index()
    gate_contrasts = compare_groups(endpoint, ["task_seed"], "capture", "centered_history", rng, replicates)
    gate = []
    for method in ("hvp_ritz", "shared_fd_ritz", "noise_subtraction"):
        rows = [r for r in gate_contrasts if r["method"] == method]
        gain_pass = len(rows) == 2 and all(r["mean_difference"] >= .05 and r["ci95"][0] > 0 for r in rows)
        timing = neural[neural.method == method].groupby(["task_seed", "seed", "epoch"])["amortized_refresh_over_step"].sum()
        cost = float(timing.mean())
        gate.append({"method": method, "capture_contrasts": rows, "capture_gate": gain_pass,
            "mean_two_layer_amortized_cost_fraction": cost, "cost_gate": cost <= .1,
            "residual_equal_cost_optimizer_need_established": False,
            "expand_deep_fashion_or_optimizer": False})
    write_json(out/"neural_gate.json", gate)
    # Held-out epoch-to-epoch basis drift; oracle overlap compares changing spaces.
    drift = []
    for r in records:
        c = r["config"]
        if c["stage"] != "neural" or r["status"] != "success":
            continue
        for layer in (0, 1):
            snapshots = {phase: np.load(path/r["path"]/f"neural-{phase}-layer{layer}.npz") for phase in protocol["neural"]["phases"]}
            for first_phase, second_phase in zip(protocol["neural"]["phases"][:-1], protocol["neural"]["phases"][1:]):
                for method in ("oracle", "centered_history", "paired_noise", "hvp_ritz", "shared_fd_ritz"):
                    a = snapshots[first_phase][f"basis_{method}"]
                    b = snapshots[second_phase][f"basis_{method}"]
                    drift.append({"seed": c["seed"], "task_seed": c["task_seed"], "layer": layer,
                        "first_epoch": first_phase, "second_epoch": second_phase, "method": method,
                        "overlap": float(np.square(a.T@b).sum()/a.shape[1])})
    pd.DataFrame(drift).to_csv(out/"neural_subspace_drift.csv", index=False)
    # Pilot manifests, frozen selections and earlier evidence preservation.
    root = project
    old = json.loads((path/"raw"/"earlier_inventory.json").read_text())
    changed = [entry["path"] for entry in old if not (root/entry["path"]).exists()
        or hashlib.sha256((root/entry["path"]).read_bytes()).hexdigest() != entry["sha256"]]
    if changed:
        raise AssertionError(f"earlier evidence changed: {changed[:5]}")
    write_json(out/"preservation_after.json", {"files_verified": len(old), "changed": changed,
        "aggregate_sha256": hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()})
    plots(path, summary, synthetic, neural, pd.DataFrame(drift))
    write_json(out/"analysis_provenance.json", {"analysis_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "experiment_source_sha256": state["environment"]["source_sha256"],
        "postprocessing_seconds": time.perf_counter()-started,
        "selection_unchanged": True, "bootstrap_unit": "whole optimization seed, paired within condition",
        "warnings": ["exploratory comparisons without multiplicity correction", "conditional on frozen joint tuning",
            "one-machine short-run wall timing", "layer-block neural oracle is not network-wide Hessian"]})
    print(json.dumps({"attempts": len(records), "statuses": dict(Counter(r["status"] for r in records)),
        "experiment_seconds": state["experiment_seconds"], "analysis_seconds": time.perf_counter()-started}, indent=2))


def plots(path, summary, synthetic, neural, drift):
    folder = path/"plots"
    folder.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric, title in zip(axes, ["mean_tail_loss", "mean_common_time_loss"], ["Equal calls (2,400)", "Equal observed loop time"]):
        table = summary.copy()
        table["condition"] = table.regime+"/"+table.rotation
        pivot = table.pivot(index="condition", columns="method", values=metric)
        pivot.plot.bar(ax=axis, logy=True)
        axis.set_ylabel("True quadratic loss")
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(folder/"baseline_quality_cost.png", dpi=170)
    plt.close(fig)
    methods = ["centered_history", "paired_noise", "noise_subtraction", "hvp_ritz", "shared_fd_ritz", "independent_fd_ritz", "random", "oracle"]
    last = synthetic[(synthetic.phase == synthetic.phase.max()) & (synthetic.regime != "isotropic_h")]
    pivot = last.groupby(["regime", "method"]).capture.mean().unstack()[methods]
    fig, axis = plt.subplots(figsize=(12, 5))
    pivot.plot.bar(ax=axis)
    axis.set_ylabel("Recovered fraction of Hessian trace")
    axis.set_title("Held-out noise orientation; rank 4, final phase")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/"synthetic_subspace_capture.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for task, axis in zip(sorted(neural.task_seed.unique()), axes):
        for method in ["centered_history", "frozen_gradient_covariance", "paired_noise", "noise_subtraction", "hvp_ritz", "shared_fd_ritz", "random", "oracle"]:
            group = neural[(neural.task_seed == task) & (neural.method == method)].groupby("epoch").capture.mean()
            axis.plot(group.index, group, marker="o", label=method)
        axis.set_title(f"Training task {task}; mean across layer blocks")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("GGN trace fraction")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.03, 1))
    fig.tight_layout()
    fig.savefig(folder/"neural_geometry_phases.png", dpi=170)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 4))
    costs = neural.groupby("method").amortized_refresh_over_step.mean().sort_values()
    costs.plot.bar(ax=axis, logy=True)
    axis.axhline(.1, color="red", linestyle="--", label="10% per-layer cost marker")
    axis.set_ylabel("One layer refresh / 100 measured training steps")
    axis.set_title("Probe cost excludes small full-training oracle analysis")
    axis.legend()
    fig.tight_layout()
    fig.savefig(folder/"neural_estimation_cost.png", dpi=170)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 4))
    drift.groupby(["first_epoch", "second_epoch", "method"]).overlap.mean().unstack().plot.bar(ax=axis)
    axis.set_ylabel("Subspace overlap across epochs")
    fig.tight_layout()
    fig.savefig(folder/"neural_subspace_drift.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    analyze(Path(args.campaign))
