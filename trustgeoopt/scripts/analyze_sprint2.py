"""Rebuild derived statistics/plots from immutable campaign attempts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from trustgeoopt.training.reliability_evaluation import evaluate_campaign
from trustgeoopt.utils.logging import write_json
from trustgeoopt.utils.hardware import environment_metadata


def records(path):
    return [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]


def baseline_analysis(path, protocol, split="evaluation"):
    all_records = records(path)
    rows = []
    for record in all_records:
        c = record["config"]
        if c["split"] == split:
            rows.append(dict(seed=c["seed"], regime=c["regime"], method=c["optimizer"]["name"],
                             rate=c["optimizer"]["lr"], run_id=record["run_id"], status=record["status"],
                             **{key: record[key] for key in ("tail_loss", "tail_distance", "final_loss", "harmful_fraction", "target_step", "target_censored", "optimizer_seconds", "correction_events")}))
    frame = pd.DataFrame(rows)
    frame.to_csv(path/"processed"/f"{split}_trials.csv", index=False)
    aggregates = []
    for (regime, method), group in frame.groupby(["regime", "method"]):
        valid = group[group.status == "success"]
        aggregates.append(dict(regime=regime, method=method, rate=float(group.rate.iloc[0]),
                               successes=len(valid), attempts=len(group),
                               tail_loss_mean=float(valid.tail_loss.mean()), tail_loss_sd=float(valid.tail_loss.std()),
                               tail_distance_mean=float(valid.tail_distance.mean()),
                               harmful_fraction_mean=float(valid.harmful_fraction.mean()),
                               target_censored=int(group.target_censored.sum()),
                               optimizer_us_per_step=float(valid.optimizer_seconds.mean()/protocol["steps"]*1e6),
                               correction_events_mean=float(valid.correction_events.mean())))
    summary = pd.DataFrame(aggregates)
    summary.to_csv(path/"processed"/f"{split}_summary.csv", index=False)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    comparisons = []
    reference = frame[frame.method == "adamw"]
    if split != "evaluation":
        reference = pd.read_csv(path/"processed"/"evaluation_trials.csv")
        reference = reference[reference.method == "adamw"]
    for (regime, method), group in frame[frame.method != "adamw"].groupby(["regime", "method"]):
        paired = group.merge(reference[reference.regime == regime], on=["regime", "seed"], suffixes=("_method", "_adamw"))
        paired = paired[(paired.status_method == "success") & (paired.status_adamw == "success")]
        values = np.log10((paired.tail_loss_method.to_numpy()+1e-12)/(paired.tail_loss_adamw.to_numpy()+1e-12))
        if not len(values):
            continue
        bootstrap = values[rng.integers(len(values), size=(protocol["bootstrap_replicates"], len(values)))].mean(1)
        comparisons.append(dict(regime=regime, method=method, paired_seeds=len(values),
                                mean_log10_tail_ratio=float(values.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                                individual_log10_ratios=values.tolist(),
                                note="Five paired seeds; fixed development selection; positive favors AdamW; floor 1e-12"))
    write_json(path/"processed"/f"{split}_paired_comparisons.json", comparisons)
    figures = path/"figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(protocol["regimes"]), figsize=(18, 3.5))
    for ax, (regime, _) in zip(axes, protocol["regimes"].items()):
        for method, group in frame[frame.regime == regime].groupby("method"):
            curves = []
            for run in group[group.status == "success"].run_id:
                curve = pd.read_csv(path/"raw"/run/"metrics.csv")
                curves.append(curve.objective_after.to_numpy())
            if curves:
                values = np.array(curves)
                median = np.median(values, axis=0)
                low, high = np.quantile(values, [.1, .9], axis=0)
                line, = ax.plot(np.maximum(median, 1e-16), label=method)
                ax.fill_between(range(values.shape[1]), np.maximum(low, 1e-16), np.maximum(high, 1e-16), alpha=.12, color=line.get_color())
        ax.set(title=regime, xlabel="step", yscale="log")
    axes[0].set_ylabel("true objective; median / 10–90% seed range")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures/f"{split}_curves.png", dpi=180)
    fig.savefig(figures/f"{split}_curves.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels, values, low, high = [], [], [], []
    for comparison in comparisons:
        labels.append(f"{comparison['regime']} / {comparison['method']}")
        values.append(comparison["mean_log10_tail_ratio"])
        low.append(comparison["ci95"][0])
        high.append(comparison["ci95"][1])
    if values:
        ax.errorbar(values, range(len(values)), xerr=[np.array(values)-low, np.array(high)-values], fmt="o", markersize=3)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.axvline(0, color="grey", linewidth=1)
    ax.set_xlabel("paired log10 tail-loss ratio to AdamW; whole-seed 95% bootstrap")
    fig.tight_layout()
    fig.savefig(figures/f"{split}_paired.png", dpi=180)
    plt.close(fig)
    return summary


def reliability_analysis(path, protocol):
    report, predictions = evaluate_campaign(path, protocol)
    write_json(path/"processed"/"reliability_analysis.json", report)
    keep = ["run_id", "seed", "regime", "lead", "step", "label", "learning_rate", "noise_scale", "phase_bin",
            "norm", "alignment", "history", "preconditioned_direction", "oracle_geometry",
            "first_order", "adjusted", "adjusted_alignment", "adjusted_history", "adjusted_both"]
    predictions[keep].to_csv(path/"processed"/"reliability_predictions.csv.gz", index=False, compression="gzip")
    rows = []
    for lead, result in report.items():
        for subset, subresult in result["subsets"].items():
            for name, metrics in subresult["metrics"].items():
                rows.append(dict(lead=int(lead), subset=subset, predictor=name,
                                 **{key: value for key, value in metrics.items() if key != "per_run"}))
    pd.DataFrame(rows).to_csv(path/"processed"/"reliability_metrics.csv", index=False)
    (path/"figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, subset in zip(axes, ("seen", "unseen")):
        for predictor in ("norm", "alignment", "history", "preconditioned_direction", "adjusted", "adjusted_both"):
            ax.plot([0, 1, 3, 5], [report[str(lead)]["subsets"][subset]["metrics"][predictor]["average_precision"] for lead in (0, 1, 3, 5)], marker="o", label=predictor)
        ax.set(title=subset, xlabel="lead steps", ylabel="pooled AP", ylim=(0, 1))
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path/"figures"/"reliability_leads.png", dpi=180)
    fig.savefig(path/"figures"/"reliability_leads.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, subset in zip(axes, ("seen", "unseen")):
        for i, predictor in enumerate(("adjusted_alignment", "adjusted_history", "adjusted_both")):
            estimates = [report[str(lead)]["subsets"][subset]["increments"][f"{predictor}-minus-adjusted"] for lead in (0, 1, 3, 5)]
            v = np.array([r["estimate"] for r in estimates])
            lo, hi = np.array([r["ci95"] for r in estimates]).T
            ax.errorbar(np.array([0, 1, 3, 5])+i*.1, v, yerr=[v-lo, hi-v], fmt="o-", label=predictor)
        ax.axhline(0, color="grey")
        ax.set(title=subset, xlabel="lead steps", ylabel="incremental AP; whole-seed 95% bootstrap")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path/"figures"/"incremental_reliability.png", dpi=180)
    fig.savefig(path/"figures"/"incremental_reliability.pdf")
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--stage", choices=["baseline", "reliability", "corrections"], required=True)
    args = parser.parse_args()
    path = Path(args.results)
    state = json.loads((path/"campaign.json").read_text())
    write_json(path/"processed"/f"analysis_provenance_{args.stage}.json", dict(
        config=state["protocol"], stage=args.stage, environment=environment_metadata("cpu"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reliability_evaluation_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/"src/trustgeoopt/training/reliability_evaluation.py").read_bytes()).hexdigest(),
        source_run_manifest_sha256=hashlib.sha256((path/"trials.jsonl").read_bytes()).hexdigest(),
        command=f"python scripts/analyze_sprint2.py --results {path.as_posix()} --stage {args.stage}"))
    if args.stage == "reliability":
        reliability_analysis(path, state["protocol"])
    else:
        summary = baseline_analysis(path, state["protocol"], "evaluation" if args.stage == "baseline" else "correction_evaluation")
        print(summary.to_string(index=False))
