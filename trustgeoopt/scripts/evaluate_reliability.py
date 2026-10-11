import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from trustgeoopt.training.metrics import prediction_metrics
from trustgeoopt.utils.logging import write_json


def evaluate_runs(root, development_seeds=(0, 1), evaluation_seeds=(2, 3, 4), lead=0):
    if set(development_seeds) & set(evaluation_seeds):
        raise ValueError("development/evaluation seeds must be disjoint")
    if lead < 0:
        raise ValueError("lead must be nonnegative")
    frames = []
    for path in sorted((Path(root)/"raw").glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        if meta["status"] != "success" or meta["optimizer"]["name"] not in {"adamw", "adamw_monitor"}:
            continue
        frame = pd.read_csv(path.parent/"metrics.csv")
        if "harmful" not in frame or meta.get("resume_from"):
            continue
        frame["label"] = frame.harmful.shift(-lead)
        frame["seed"] = meta["seed"]
        frame["run_id"] = meta["experiment_id"]
        frame["regime"] = meta["config"]["objective"]["kind"]
        frames.append(frame.dropna(subset=["label"]))
    if not frames:
        raise ValueError("no qualifying synthetic AdamW runs")
    data = pd.concat(frames, ignore_index=True)
    development = data[data.seed.isin(development_seeds)]
    evaluation = data[data.seed.isin(evaluation_seeds)]
    if development.empty or evaluation.empty:
        raise ValueError("need both held-out and development runs")
    report = dict(development_seeds=list(development_seeds), evaluation_seeds=list(evaluation_seeds),
                  lead_steps=lead, source_runs=sorted(data.run_id.unique().tolist()),
                  note="Pilot pooled-step metrics are descriptive; bootstrap whole paired runs for inference.", signals={})
    signals = {"gradient_norm": 1, "scale_mismatch": 1, "momentum_alignment": -1,
               "preconditioner_drift": 1, "history_consistency": -1}
    for signal, sign in signals.items():
        negatives = sign*development.loc[development.label == 0, signal].dropna().to_numpy()
        if not len(negatives):
            report["signals"][signal] = dict(status="undefined: no development negatives")
            continue
        threshold = float(np.quantile(negatives, 0.95))
        report["signals"][signal] = prediction_metrics(evaluation.label, sign*evaluation[signal], threshold)
        report["signals"][signal]["per_run"] = {
            run_id: prediction_metrics(group.label, sign*group[signal], threshold)
            for run_id, group in evaluation.groupby("run_id")}
        report["signals"][signal]["per_regime"] = {
            regime: prediction_metrics(group.label, sign*group[signal], threshold)
            for regime, group in evaluation.groupby("regime")}
    destination = Path(root)/"processed"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/f"reliability-lead{lead}.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--development-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--lead", type=int, default=0)
    args = parser.parse_args()
    report = evaluate_runs(args.results, args.development_seeds, args.evaluation_seeds, args.lead)
    print(json.dumps({key: {k: v for k, v in value.items() if k not in {"per_run", "per_regime"}}
                      for key, value in report["signals"].items()}, indent=2))
