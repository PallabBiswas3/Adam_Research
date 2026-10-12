"""Analyze a completed frozen campaign without retraining or policy selection."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from robust_learning.analysis import bootstrap_mean, matched_point


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output):
    destination = output/"processed"/"analysis"
    destination.mkdir(exist_ok=False)
    state = json.loads((output/"campaign.json").read_text())
    if "evaluation" not in state["stages"]:
        raise ValueError("evaluation must finish before analysis")
    all_runs = [json.loads(line) for line in (output/"runs.jsonl").read_text().splitlines()]
    runs = [r for r in all_runs if r["config"]["split"] in {"evaluation", "held_out"} and r["status"] == "success"]
    protocol = state["protocol"]
    methods, regimes = protocol["methods"], [r["name"] for r in protocol["regimes"]]
    per_run, curves, matched, comparisons, summaries = [], [], [], [], []
    lookup = {}
    for r in runs:
        c = r["config"]
        labels = {"regime": c["regime"]["name"], "method": c["method"], "seed": c["seed"], "split": c["split"]}
        row = {**labels, "lr": c["lr"], **{k: v for k, v in r["test"].items() if not isinstance(v, list)},
               **r["actual_selection"], "realized_noise_rate": r["realized_noise_rate"],
               **{k: r[k] for k in ["loop_seconds", "total_seconds", "parameter_bytes", "state_bytes", "history_bytes", "updates"]}}
        per_run.append(row)
        lookup[(labels["regime"], labels["method"], labels["seed"])] = row
        curve = json.loads((output/r["path"]/"operating_curve.json").read_text())
        curves.extend({**labels, **point} for point in curve)
        for target in protocol["matched_rejected_noise_precision"]:
            point = matched_point(curve, target)
            matched.append({**labels, "target": target, "feasible": point is not None,
                **{k: point[k] if point else None for k in ["rejected_noise_precision", "noise_recall", "hard_clean_retention", "tail_clean_retention", "acceptance_fraction"]}})
    for regime in regimes:
        for method in methods:
            rows = [r for r in per_run if r["regime"] == regime and r["method"] == method]
            if not rows:
                continue
            for metric in ["macro_f1", "ece", "hard_clean_retention", "tail_clean_retention", "noise_recall", "total_seconds"]:
                values = [r[metric] for r in rows if r[metric] is not None]
                if values:
                    summaries.append({"regime": regime, "method": method, "metric": metric,
                                      **bootstrap_mean(values, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])})
            if method == "ce":
                continue
            for reference in ["ce", "global_small_loss"]:
                if method == reference:
                    continue
                for metric in ["macro_f1", "hard_clean_retention", "tail_clean_retention", "noise_recall"]:
                    differences = []
                    for r in rows:
                        other = lookup.get((regime, reference, r["seed"]))
                        if other and r[metric] is not None and other[metric] is not None:
                            differences.append(r[metric]-other[metric])
                    if differences:
                        comparisons.append({"regime": regime, "method": method, "reference": reference,
                            "metric": metric, **bootstrap_mean(differences, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])})
    matched_comparisons = []
    for regime in regimes:
        for target in protocol["matched_rejected_noise_precision"]:
            for method in methods:
                if method == "global_small_loss":
                    continue
                a = {r["seed"]: r for r in matched if r["regime"] == regime and r["method"] == method and r["target"] == target}
                b = {r["seed"]: r for r in matched if r["regime"] == regime and r["method"] == "global_small_loss" and r["target"] == target}
                for metric in ["hard_clean_retention", "tail_clean_retention"]:
                    deltas = [a[s][metric]-b[s][metric] for s in sorted(set(a)&set(b)) if a[s]["feasible"] and b[s]["feasible"] and a[s][metric] is not None and b[s][metric] is not None]
                    matched_comparisons.append({"regime": regime, "method": method, "reference": "global_small_loss", "target": target, "metric": metric,
                        "method_feasible": sum(r["feasible"] for r in a.values()), "reference_feasible": sum(r["feasible"] for r in b.values()),
                        **(bootstrap_mean(deltas, protocol["bootstrap_replicates"], protocol["bootstrap_seed"]) if deltas else {"n": 0, "mean": None, "lower": None, "upper": None})})
    # A seed is the resampling block even when aggregating all eight regimes.
    aggregate = []
    for method in methods:
        if method == "ce":
            continue
        differences = []
        for seed in protocol["evaluation_seeds"]:
            pairs = [(lookup.get((regime, method, seed)), lookup.get((regime, "ce", seed))) for regime in regimes]
            if all(a and b for a, b in pairs):
                differences.append(np.mean([a["macro_f1"]-b["macro_f1"] for a, b in pairs]))
        if differences:
            aggregate.append({"method": method, "reference": "ce", "metric": "eight_regime_mean_macro_f1",
                              **bootstrap_mean(differences, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])})
    for name, values in [("run_metrics", per_run), ("operating_curves", curves), ("matched_points", matched), ("summary", summaries),
                         ("paired_comparisons", comparisons), ("matched_comparisons", matched_comparisons), ("aggregate_comparisons", aggregate)]:
        csv_write(destination/(name+".csv"), values)
    status = {s: sum(r["status"] == s for r in all_runs) for s in sorted(set(r["status"] for r in all_runs))}
    (destination/"status.json").write_text(json.dumps({"status_counts": status, "evaluation_successes": len(runs),
        "experiment_seconds": state["experiment_seconds"], "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inference": "4000 seed-block percentile bootstrap; descriptive, five seeds; not independent steps or simultaneous intervals"}, indent=2))
    # Same eight regimes on all plots, all methods including negative results.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, regime in zip(axes.flat, regimes):
        group = [r for r in summaries if r["regime"] == regime and r["metric"] == "macro_f1"]
        for index, method in enumerate(methods):
            r = next(r for r in group if r["method"] == method)
            ax.errorbar(index, r["mean"], yerr=[[r["mean"]-r["lower"]], [r["upper"]-r["mean"]]], fmt="o", capsize=3)
        ax.set_xticks(range(5), ["CE", "Global", "Classwise", "CoTeach*", "CT*"], rotation=40)
        ax.set_title(regime.replace("_", " "), fontsize=9)
        ax.set_ylim(0, 1)
    fig.suptitle("Clean test macro-F1: five runs; descriptive seed-bootstrap 95% intervals; *toy adaptations")
    fig.tight_layout()
    fig.savefig(output/"plots"/"test_macro_f1.png", dpi=160)
    plt.close(fig)
    for ymetric, name, ylabel in [("noise_recall", "selection_precision_recall", "Noise rejection recall"),
                                   ("hard_clean_retention", "hard_clean_operating_curve", "Hard-clean retention"),
                                   ("tail_clean_retention", "tail_clean_operating_curve", "Tail-clean retention")]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
        for ax, regime in zip(axes.flat, regimes):
            for method in methods:
                # Plot individual run curves; do not fabricate a monotone envelope or interpolate.
                for i, seed in enumerate(protocol["evaluation_seeds"]):
                    points = [r for r in curves if r["regime"] == regime and r["method"] == method and r["seed"] == seed and r["rejected_noise_precision"] is not None and r[ymetric] is not None]
                    ax.plot([r["rejected_noise_precision"] for r in points], [r[ymetric] for r in points],
                        "o-", markersize=2, linewidth=.7, alpha=.4, color=f"C{methods.index(method)}", label=method if i == 0 else None)
            ax.set_title(regime.replace("_", " "), fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Rejected-noise precision", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
        fig.suptitle("Frozen diagnostic rankings; CE/peer/CT curves differ from actual training selection", y=.96, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(output/"plots"/(name+".png"), dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, method in enumerate(methods):
        rows = [r for r in per_run if r["method"] == method]
        axes[0].bar(index, np.mean([r["total_seconds"] for r in rows]))
        axes[1].bar(index, np.mean([r["parameter_bytes"]+r["state_bytes"]+r["history_bytes"] for r in rows])/1024)
    for ax in axes:
        ax.set_xticks(range(5), ["CE", "Global", "Classwise", "CoTeach*", "CT*"], rotation=30)
    axes[0].set_ylabel("Mean total run seconds (descriptive)")
    axes[1].set_ylabel("Counted parameter+optimizer+history KiB; not peak RAM")
    fig.tight_layout(); fig.savefig(output/"plots"/"compute_cost.png", dpi=160); plt.close(fig)
    print(json.dumps({"status": status, "evaluation_runs": len(runs), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(ROOT/args.output)
