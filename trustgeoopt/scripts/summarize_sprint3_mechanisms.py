"""Exploratory whole-run subspace and direct SOAP-versus-SGD contrasts.

Post-evaluation diagnostics only: never updates frozen selections.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from analyze_sprint3 import paired_interval
from trustgeoopt.utils.logging import write_json


def summarize(path):
    protocol = json.loads((path/"campaign.json").read_text())["protocol"]
    rng = np.random.default_rng(20261014)
    g = pd.read_csv(path/"processed/geometry_snapshots.csv")
    final = g[(~g.hessian_is_isotropic)&(g.method == "adamw")&(g.step == 2400)]
    labels = {"history_curvature_capture": "History", "centered_history_curvature_capture": "Centered history",
              "random_curvature_capture": "Random", "diagonal_curvature_capture": "Coordinate", "oracle_curvature_capture": "Oracle"}
    axis = final.groupby("regime")[list(labels)].mean().rename(columns=labels).plot.bar(figsize=(14, 6), rot=45)
    axis.set_ylabel("Curvature trace fraction")
    axis.set_xlabel("")
    axis.set_title("Rank-4 subspaces: AdamW final snapshot, five evaluation seeds")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path/"plots/history_controls.png", dpi=150)
    plt.close()
    intervals = []
    for (regime, method, step), group in g[~g.hessian_is_isotropic].groupby(["regime", "method", "step"]):
        for name in ("history", "centered_history", "diagonal"):
            difference = (group[f"{name}_curvature_capture"]-group.random_curvature_capture).to_numpy()
            draws = difference[rng.integers(0, len(difference), (2000, len(difference)))].mean(1)
            intervals.append({"regime": regime, "method": method, "step": int(step), "subspace": name,
                "n_runs": len(difference), "mean_capture_minus_random": float(difference.mean()),
                "ci95": np.quantile(draws, [.025, .975]).tolist(), "paired_run_differences": difference.tolist()})
    records = [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]
    direct = []
    for regime in sorted(g.regime.unique()):
        a = {r["config"]["seed"]: r["tail_loss"] for r in records if r["config"]["stage"] == "geometry" and r["config"]["regime"] == regime and r["config"]["split"] != "tuning" and r["config"]["optimizer"]["name"] == "soap" and r["status"] == "success"}
        b = {r["config"]["seed"]: r["tail_loss"] for r in records if r["config"]["stage"] == "geometry" and r["config"]["regime"] == regime and r["config"]["split"] != "tuning" and r["config"]["optimizer"]["name"] == "momentum" and r["status"] == "success"}
        direct.append({"regime": regime, "contrast": "SOAP / Momentum", **paired_interval(a, b,
            protocol["evaluation_seeds"], rng, 2000, protocol["selection_floor"])})
    output = {"scope": "exploratory post-evaluation contrasts; no multiplicity or selection adjustment",
        "resampling_unit": "whole seed-matched run", "replicates": 2000, "seed": 20261014,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_library_sha256": hashlib.sha256((Path(__file__).parent/"analyze_sprint3.py").read_bytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256((path/name).read_bytes()).hexdigest() for name in
            ("trials.jsonl", "processed/geometry_snapshots.csv", "campaign.json")},
        "subspace_intervals": intervals, "direct_loss_comparisons": direct}
    write_json(path/"processed/mechanism_statistics.json", output)
    # Preserve every existing artifact and extend the provenance manifest.
    manifest = json.loads((path/"artifact_manifest.json").read_text())
    for target in (path/"processed/mechanism_statistics.json", path/"plots/history_controls.png"):
        relative = target.relative_to(path).as_posix()
        manifest = [entry for entry in manifest if entry["path"] != relative]
        manifest.append({"path": relative, "bytes": target.stat().st_size,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    write_json(path/"artifact_manifest.json", sorted(manifest, key=lambda entry: entry["path"]))
    print("Whole-run exploratory contrasts saved; frozen selections unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    summarize(Path(parser.parse_args().output))
