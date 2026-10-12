"""Timing/noise-phase confounding audit and additional compact measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from trustgeoopt.utils.logging import write_json


def summarize(path):
    out = path/"processed"
    if (out/"timing_phase_audit.json").exists():
        raise ValueError("already summarized; do not overwrite")
    rows = pd.read_csv(out/"baseline_runs.csv")
    phases = []
    for row in rows.to_dict("records"):
        trajectory = pd.read_csv(path/row["path"]/"trajectory.csv")
        selected = trajectory[trajectory.seconds <= row["common_time_horizon"]]
        phases.append({"regime": row["regime"], "rotation": row["rotation"], "method": row["method"],
            "seed": row["seed"], "step": int(selected.step.iloc[-1]),
            "common_time_noise_std": float(selected.noise_std.iloc[-1]),
            "transition_reached": bool(selected.step.iloc[-1] > 1200)})
    phase_frame = pd.DataFrame(phases)
    phase_frame.to_csv(out/"common_time_phase_runs.csv", index=False)
    grouped = []
    for key, frame in phase_frame.groupby(["regime", "rotation", "method"]):
        grouped.append({"regime": key[0], "rotation": key[1], "method": key[2],
            "noise_levels": sorted(frame.common_time_noise_std.unique().tolist()),
            "n_after_transition": int(frame.transition_reached.sum()), "n_runs": len(frame),
            "min_steps": int(frame.step.min()), "max_steps": int(frame.step.max())})
    write_json(out/"timing_phase_audit.json", {"groups": grouped,
        "warning": "At equal wall time, step-indexed noise changes occur at different elapsed times. Rising/falling comparisons mix noise phases and are descriptive only. Constant noise avoids this specific confound.",
        "timing_limit": "Shared process, short CPU runs, operating-system scheduling/cache/serialization; no independent repeated timing campaign.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    # All covariance-error entries within a phase refer to the SAME paired sample.
    # Select one method to avoid pseudo-replication across estimator rows.
    synthetic = pd.read_csv(out/"synthetic_estimators.csv")
    diagnostics = synthetic[synthetic.method == "paired_noise"].groupby(["regime", "phase"])[
        ["noise_overlap", "noise_covariance_relative_error", "signal_noise_norm_ratio", "subtraction_min_eigenvalue"]].agg(["mean", "std"])
    diagnostics.to_csv(out/"noise_identification_summary.csv")
    costs = synthetic.groupby("method")[["refresh_seconds", "operator_calls", "extra_gradient_calls", "retained_bytes", "actual_reconstruction_error"]].mean()
    costs.to_csv(out/"synthetic_cost_summary.csv")
    neural = pd.read_csv(out/"neural_estimators.csv")
    # Shared paired gradients are computed once for ALL layers. Their attributed
    # per-layer timing sum is an upper estimate, disclosed explicitly.
    shared = neural[neural.method == "paired_noise"].drop_duplicates(["task_seed", "seed", "epoch"])
    neural_cost = {"mean_batch32_gradient_seconds": float((shared.gradient_sampling_seconds/128).mean()),
        "mean_measured_step_seconds": float(shared.mean_step_seconds.mean()),
        "mean_paired_128_gradient_calls_refresh_seconds": float(shared.gradient_sampling_seconds.mean()),
        "paired_refresh_amortized_fraction_once_all_layers": float((shared.gradient_sampling_seconds/(100*shared.mean_step_seconds)).mean()),
        "note": "One paired collection supplies both layers; neural_gate.json sums conservative layer-attributed paired timing. HVP timing uses distinct products per layer. Step cost includes measured training/history capture and loader overhead."}
    write_json(out/"neural_cost_details.json", neural_cost)
    validation = []
    for record in [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]:
        if record["config"]["stage"] == "neural":
            validation.append({"seed": record["config"]["seed"], "task_seed": record["config"]["task_seed"],
                "validation_loss": record["validation_loss"], "train_seconds": record["train_seconds"],
                "diagnostic_seconds": record["diagnostic_seconds"], "state_bytes": record["state_bytes"]})
    pd.DataFrame(validation).to_csv(out/"neural_training_runs.csv", index=False)
    print("Timing phase audit and measured noise/cost summaries saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    summarize(Path(args.campaign))
