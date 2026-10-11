"""Whole-run paired resampling, compact results, provenance audit and plots."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path, PureWindowsPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from trustgeoopt.training.quadratic_theory import scalar_stationary_loss, stationary_covariance
from trustgeoopt.utils.logging import write_json
from trustgeoopt.utils.hardware import environment_metadata


def trial_folder(path, record):
    return path/"raw"/PureWindowsPath(record["path"]).name


def paired_interval(a, b, seeds, rng, replicates, floor):
    if set(a) != set(seeds) or set(b) != set(seeds):
        return {"status": "incomplete paired runs", "n": len(set(a)&set(b))}
    differences = np.array([np.log10(a[s]+floor)-np.log10(b[s]+floor) for s in seeds])
    values = differences[rng.integers(0, len(seeds), (replicates, len(seeds)))].mean(1)
    return {"status": "complete", "n": len(seeds), "log10_ratio": float(differences.mean()),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(), "paired_differences": differences.tolist()}


def analyze(path):
    started = time.perf_counter()
    state = json.loads((path/"campaign.json").read_text())
    protocol = state["protocol"]
    records = [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    summaries, comparisons, boundaries, geometry = [], [], [], []
    groups = {}
    for r in records:
        c = r["config"]
        if c["split"] != "tuning":
            key = (c["stage"], c["regime"], c["split"], c["optimizer"]["name"])
            groups.setdefault(key, []).append(r)
            for snapshot in r.get("geometry", []):
                geometry.append({"regime": c["regime"], "split": c["split"], "method": c["optimizer"]["name"],
                                 "seed": c["seed"], **snapshot})
    for (stage, regime, split, method), group in groups.items():
        metric = "validation_loss" if stage == "neural" else "tail_loss"
        success = [r for r in group if r["status"] == "success"]
        values = [r[metric] for r in success]
        reached = [r["target"]["seconds"] for r in success if r.get("target")]
        summaries.append({"stage": stage, "regime": regime, "split": split, "method": method,
            "n_success": len(success), "n_attempts": len(group), "mean_loss": np.mean(values) if values else None,
            "sd_loss": np.std(values, ddof=1) if len(values) > 1 else None,
            "mean_runtime_seconds": np.mean([r["runtime_seconds"] for r in success]) if success else None,
            "target_reached": len(reached), "target_time_mean_reached_only": np.mean(reached) if reached else None,
            "mean_state_bytes": np.mean([r["state_tensor_bytes"] for r in success]) if success else None,
            "max_sampled_process_rss_bytes": max((r.get("sampled_process_rss_bytes", 0) for r in success), default=0)})
        baseline = groups.get((stage, regime, split, "adamw"), [])
        a = {r["config"]["seed"]: r[metric] for r in success}
        b = {r["config"]["seed"]: r[metric] for r in baseline if r["status"] == "success"}
        if method != "adamw":
            comparisons.append({"stage": stage, "regime": regime, "split": split, "method": method,
                "baseline": "adamw", **paired_interval(a, b, protocol["evaluation_seeds"], rng,
                    protocol["bootstrap_replicates"], protocol["selection_floor"])})
    # Coordinate dependence: pair entire diagonal/rotated runs using latent seed.
    for (stage, regime, split, method), group in groups.items():
        if stage != "geometry" or not regime.endswith("/rotated"):
            continue
        diagonal = groups.get((stage, regime.removesuffix("/rotated")+"/diagonal", split, method), [])
        a = {r["config"]["seed"]: r["tail_loss"] for r in group if r["status"] == "success"}
        b = {r["config"]["seed"]: r["tail_loss"] for r in diagonal if r["status"] == "success"}
        comparisons.append({"stage": stage, "regime": regime, "split": split, "method": method,
            "baseline": "matched_diagonal", **paired_interval(a, b, protocol["evaluation_seeds"], rng,
                protocol["bootstrap_replicates"], protocol["selection_floor"])})
    for stage in ("stochastic", "geometry", "neural"):
        selection = json.loads((path/"processed"/f"{stage}_selection.json").read_text())
        for regime_method, candidate in selection.items():
            if not candidate:
                continue
            rates = protocol[stage]["rates"][candidate["name"]]
            if candidate["lr"] in (min(rates), max(rates)):
                boundaries.append({"stage": stage, "key": regime_method, "candidate": candidate,
                                   "boundary": "lower" if candidate["lr"] == min(rates) else "upper"})
    out = path/"processed"
    pd.DataFrame(summaries).to_csv(out/"summary.csv", index=False)
    pd.DataFrame(geometry).to_csv(out/"geometry_snapshots.csv", index=False)
    write_json(out/"paired_comparisons.json", comparisons)
    write_json(out/"grid_boundaries.json", boundaries)
    write_json(out/"status_counts.json", pd.Series([r["status"] for r in records]).value_counts().to_dict())
    development_details(path, records)
    theory_validation(path)
    plots(path, summaries, geometry, records)
    snapshot = path/"source_snapshot_analysis.zip"
    with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
        root = Path(__file__).resolve().parents[1]
        for source in sorted(list((root/"src").rglob("*.py"))+list((root/"scripts").glob("*.py"))):
            archive.write(source, source.relative_to(root))
    write_json(out/"analysis_provenance.json", {"environment": environment_metadata("cpu"),
        "snapshot": snapshot.name, "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "runtime_seconds": time.perf_counter()-started,
        "note": "Postprocessing source captured separately; optimizer-run code in the original source_snapshot.zip is unchanged"})
    # Manifest verifies every individual attempt, compressed snapshots and raw rows.
    manifest = []
    for source in sorted(path.rglob("*")):
        if source.is_file() and source.name != "artifact_manifest.json":
            manifest.append({"path": source.relative_to(path).as_posix(), "bytes": source.stat().st_size,
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    write_json(path/"artifact_manifest.json", manifest)
    print(f"Analyzed {len(records)} attempts; {state['experiment_seconds']:.1f} campaign seconds", flush=True)


def development_details(path, records):
    groups = {}
    for r in records:
        c = r["config"]
        if c["split"] == "tuning":
            key = (c["stage"], c["regime"], json.dumps(c["optimizer"], sort_keys=True))
            groups.setdefault(key, []).append(r)
    rows = []
    for (stage, regime, candidate), group in groups.items():
        c = json.loads(candidate)
        metric = "validation_loss" if stage == "neural" else "tail_loss"
        values = [r.get(metric) for r in group if r["status"] == "success"]
        valid = len(values) == 2
        rows.append({"stage": stage, "regime": regime, "method": c["name"], "lr": c["lr"],
            "schedule": c.get("schedule", "constant"), "beta2": c.get("betas", [None, None])[1],
            "momentum": c.get("momentum"), "refresh_interval": c.get("refresh_interval"),
            "n_success": len(values), "n_attempts": len(group),
            "mean_loss": float(np.mean(values)) if valid else None,
            "mean_log10_loss": float(np.mean(np.log10(np.array(values)+1e-12))) if valid else None,
            "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in group])),
            "candidate": candidate})
    df = pd.DataFrame(rows)
    df.to_csv(path/"processed/development_candidates.csv", index=False)
    df[df.stage == "stochastic"].sort_values("mean_log10_loss").groupby(
        ["regime", "method", "schedule"]).head(1).to_csv(path/"processed/development_schedule_best.csv", index=False)
    # Actual normalized feature covariance after train-only preprocessing.
    from trustgeoopt.datasets.synthetic import supervised_splits
    protocol = json.loads((path/"campaign.json").read_text())["protocol"]
    feature_controls = []
    for seed in (700, 701):
        training = supervised_splits({**protocol["neural"]["dataset"], "data_seed": seed}, 0)[0]
        x = training.tensors[0].double()
        covariance = torch_covariance(x)
        values = np.linalg.eigvalsh(covariance)
        feature_controls.append({"data_seed": seed, "normalized_training_covariance_eigenvalues": values.tolist(),
                                 "condition": float(values[-1]/values[0])})
    write_json(path/"processed/neural_feature_controls.json", feature_controls)


def torch_covariance(x):
    centered = x-x.mean(0)
    return (centered.T@centered/(len(x)-1)).numpy()


def theory_validation(path):
    # Independent trajectories are the sampling units; no iid-step uncertainty.
    begin = time.perf_counter()
    rows = []
    for beta in (0.0, 0.9):
        for lr in (0.03, 0.1, 0.3):
            exact = stationary_covariance(np.array([[3.]]), np.array([[2.]]), np.array([[0.7]]), lr, beta)
            rng = np.random.default_rng(9000+int(100*beta)+int(1000*lr))
            x, m, losses = np.zeros(32), np.zeros(32), np.zeros(32)
            for step in range(12000):
                noise = rng.normal(0, np.sqrt(2), 32)
                m = beta*m+(1-beta)*(3*x+noise)
                x -= lr*0.7*m
                if step >= 2000:
                    losses += 1.5*x*x/10000
            boot = losses[rng.integers(0, 32, (2000, 32))].mean(1)
            rows.append({"beta1": beta, "lr": lr, "h": 3, "noise_variance": 2, "b": 0.7,
                         "expected_loss": exact["loss"], "scalar_formula": scalar_stationary_loss(3, 2, 0.7, lr, beta),
                         "empirical_loss": float(losses.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
                         "lyapunov_relative_residual": exact["relative_residual"], "trajectory_losses": losses.tolist()})
    write_json(path/"processed/theory_validation.json", {"assumptions": "frozen scalar B; iid Gaussian noise; normalized momentum; not actual Adam",
        "steps": 12000, "burn_in": 2000, "trajectories": 32, "runtime_seconds": time.perf_counter()-begin, "results": rows})


def plots(path, summaries, geometry, records):
    folder = path/"plots"
    folder.mkdir(exist_ok=True)
    df = pd.DataFrame(summaries)
    for stage in ("stochastic", "geometry", "neural"):
        selected = df[df.stage == stage]
        regimes = selected.regime.unique()
        fig, axes = plt.subplots(1, len(regimes), figsize=(max(8, 2.4*len(regimes)), 3.6), squeeze=False)
        for axis, regime in zip(axes[0], regimes):
            data = selected[selected.regime == regime]
            axis.bar(data.method, np.maximum(data.mean_loss, 1e-12), yerr=data.sd_loss.fillna(0), color="steelblue")
            axis.set_yscale("log")
            axis.set_title(regime, fontsize=9)
            axis.tick_params(axis="x", rotation=70)
            axis.set_ylabel("mean validation loss" if stage == "neural" else "mean tail loss (floor 1e-12)")
        fig.suptitle("Five evaluation seeds; bars ± run SD; held-out controls labeled by regime")
        fig.tight_layout()
        fig.savefig(folder/f"{stage}_loss.png", dpi=150)
        fig.savefig(folder/f"{stage}_loss.pdf")
        plt.close(fig)
    gd = pd.DataFrame(geometry)
    if len(gd):
        data = gd[(gd.step == 2400)&(gd.method == "adamw")&(~gd.hessian_is_isotropic)]
        cols = ["history_curvature_capture", "centered_history_curvature_capture", "random_curvature_capture", "diagonal_curvature_capture", "oracle_curvature_capture"]
        data.groupby("regime")[cols].mean().plot.bar(figsize=(14, 4))
        plt.ylabel("rank-4 captured trace(H) / trace(H)")
        plt.tight_layout()
        plt.savefig(folder/"history_controls.png", dpi=150)
        plt.close()
    # Time-quality curves include both reached and censored trials.
    neural = [r for r in records if r["config"]["stage"] == "neural" and r["config"]["split"] == "evaluation" and r["status"] == "success"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, model in zip(axes, ("small", "deep", "fashion")):
        for method in ("adamw", "momentum", "soap"):
            runs = [pd.read_csv(trial_folder(path, r)/"metrics.csv") for r in neural if r["config"]["model"] == model and r["config"]["optimizer"]["name"] == method]
            if runs:
                axis.plot(np.mean([r.elapsed_seconds for r in runs], axis=0), np.mean([r.validation_loss for r in runs], axis=0), label=method)
        axis.set_title(model)
        axis.set_xlabel("mean loop wall time (seconds)")
        axis.set_ylabel("mean validation loss")
        axis.legend()
    fig.tight_layout()
    fig.savefig(folder/"neural_time_quality.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, regime in zip(axes, ("rising", "constant", "falling")):
        for method in ("adamw", "momentum", "shampoo"):
            trials = [r for r in records if r["config"]["stage"] == "stochastic" and r["config"]["split"] == "evaluation" and r["config"]["regime"] == regime and r["config"]["optimizer"]["name"] == method and r["status"] == "success"]
            curves = [pd.read_csv(trial_folder(path, r)/"metrics.csv") for r in trials]
            if curves:
                mean = np.mean([curve.loss.to_numpy() for curve in curves], axis=0)
                axis.plot(curves[0].step, np.maximum(mean, 1e-12), label=method)
        if regime != "constant":
            axis.axvline(1200, color="gray", linestyle=":")
        axis.set_title(regime)
        axis.set_yscale("log")
        axis.set_xlabel("gradient calls")
        axis.set_ylabel("mean true loss")
        axis.legend()
    fig.tight_layout()
    fig.savefig(folder/"stochastic_transients.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    analyze(Path(parser.parse_args().output))
