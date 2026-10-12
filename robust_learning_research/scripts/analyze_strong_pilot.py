"""Summarize executed feasibility operations; never infer benchmark accuracy."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"strong-baseline-20261012"


def estimate(record, config, count, accepted_fraction=.5):
    if record["status"] != "completed_kernel_pilot":
        return None
    method = record["method"]
    section = "dividemix" if method == "dividemix" else "dss_plus" if method.startswith("dss_plus") else "dss"
    settings = config[section]
    warm = statistics.median(u["seconds"] for u in record["warmup_updates"])
    main = statistics.median(u["seconds"] for u in record["main_updates"])
    warm_steps = math.ceil(count/record["batch_size"])
    main_steps = math.ceil(count*accepted_fraction/record["batch_size"]) if method == "dividemix" else warm_steps
    scan = sum(s["forward_scan_seconds"] for s in record.get("scan", []))*count/record["pilot_train_count"]
    # Full-data GMM fitting and growing DSS histories are deliberately unestimated.
    seconds = settings["warmup"]*warm_steps*record["networks"]*warm
    seconds += (settings["epochs"]-settings["warmup"])*(main_steps*main+scan)
    return seconds/3600


if __name__ == "__main__":
    result = json.loads((OUT/"pilot"/"results.json").read_text())
    dataset = json.loads((OUT/"dataset_provenance.json").read_text())
    config = result["config"]
    rows = []
    for record in result["executed_method_records"]:
        updates = record["main_updates"]
        rows.append({"method": record["method"], "status": record["status"],
            "reason": record.get("reason", record.get("error", "")),
            "warmup_network_updates": len(record["warmup_updates"]), "main_rounds": len(updates),
            "total_seconds": record["total_seconds"], "peak_rss_mib": record["sampled_process_peak_rss_mib"],
            "main_round_median_seconds": statistics.median(u["seconds"] for u in updates) if updates else None,
            "parameter_mib": record.get("parameter_bytes", 0)/2**20,
            "sgd_state_mib": record.get("sgd_state_bytes", 0)/2**20,
            "longtail_run_hours_mid_scenario": estimate(record, config, dataset["train_count"]),
            "balanced_run_hours_mid_scenario": estimate(record, config, 45000),
            "longtail_run_hours_low_scenario": estimate(record, config, dataset["train_count"], .1),
            "longtail_run_hours_high_scenario": estimate(record, config, dataset["train_count"], .9)})
    with (OUT/"pilot_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    available = all(r["longtail_run_hours_mid_scenario"] is not None for r in rows)
    campaign = {}
    if available:
        for label, fraction in [("low", .1), ("mid", .5), ("high", .9)]:
            campaign[label+"_hours"] = sum(9*estimate(r, config, 45000, fraction)+14*estimate(r, config, dataset["train_count"], fraction)
                                            for r in result["executed_method_records"])
    summary = {"training_seconds": result["training_seconds"], "os_peak_rss_mib": result["OS_process_peak_rss_mib"],
        "rows": rows, "campaign_138_runs": campaign, "quality_metrics": None, "full_benchmark_executed": False,
        "projection_batch": 16, "projection_limitations": ["only three timed nonconverged rounds", "possible background resource contention",
        "brief pytest overlap during middle single-network timings; timings are not isolated hardware throughput",
        "full-data GMM fitting and growing DSS histories omitted", "validation/checkpoint I/O omitted", "batch128 and GPU throughput unmeasured",
        "DivideMix accepted-label fraction scenarios .1/.5/.9 are assumptions, not uncertainty intervals"]}
    (OUT/"feasibility_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    table = "| Method | Status | Warmup network updates | Main rounds | Total seconds | Peak RSS MiB |\n|---|---|---:|---:|---:|---:|\n"
    for r in rows:
        table += f"| {r['method']} | {r['status']} | {r['warmup_network_updates']} | {r['main_rounds']} | {r['total_seconds']:.2f} | {r['peak_rss_mib']:.1f} |\n"
    costs = "| Method | Long-tail hours/run | Balanced hours/run |\n|---|---:|---:|\n"
    for r in rows:
        values = [f"{r[k]:.1f}" if r[k] is not None else "not estimable: censored/failed" for k in ["longtail_run_hours_mid_scenario", "balanced_run_hours_mid_scenario"]]
        costs += f"| {r['method']} | {values[0]} | {values[1]} |\n"
    campaign_text = (f"The 138-run plan projects {campaign['mid_hours']:.0f} CPU hours in the middle scenario "
        f"({campaign['low_hours']:.0f}-{campaign['high_hours']:.0f} hours across assumed DivideMix acceptance fractions). "
        "These are workload scenarios, not statistical confidence intervals.") if available else (
        "A complete 138-run campaign estimate is unavailable because at least one method was censored or failed; do not silently omit it from the total.")
    feasibility = f"""# Benchmark feasibility: stop after the bounded pilot

One seed601 on actual CIFAR-10 images, full PreActResNet-18, CPU float32,
two PyTorch threads. Installed PyTorch {result['environment']['torch']},
Python {result['environment']['python']}; CUDA unavailable to this environment.
No dependency upgrade, paid GPU or full campaign was run. An absent nvidia-smi
command does not establish that this machine has no physical GPU.

Prepared data: {dataset['train_count']:,} long-tail training images, 2,000 separate
model-validation images, {dataset['prior_count']} separate prior-calibration labels,
10,000 clean test images. The pilot uses128 training images and batch16 only.
Archive MD5 matches torchvision's original CIFAR-10 MD5; the original host was
slow, so the verified identical archive came from a pinned mirror.
[Dataset hashes](../results/strong-baseline-20261012/dataset_provenance.json),
[mirror/setup provenance](../results/strong-baseline-20261012/mirror_provenance.json).
Original [dataset source](https://www.cs.toronto.edu/~kriz/cifar.html);
[mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python).
Dataset access is separate from MIT software licensing; no dataset images or
unlicensed DSS source are redistributed through Git.

## Actual operations and cost

{table}

Shared training elapsed {result['training_seconds']:.2f}s; OS lifetime peak
RSS {result['OS_process_peak_rss_mib']:.1f}MiB. Per-method RSS sampling is20ms;
OS peak includes imports/data preparation inside the pilot. Guards checked at
operation boundaries:180s training,1,800MiB process RSS,700MiB system available.
These are not hard OS limits. Failed/censored attempts and incomplete rounds
remain in [all records](../results/strong-baseline-20261012/pilot/results.json).

Single learners have about42.6MiB parameter storage plus about42.6MiB SGD momentum;
two learners double those tensor costs. Activations dominate variable memory,
especially DivideMix's concatenated four-view batch. Measured RSS at batch16
does **not** certify published batch128 memory feasibility. Full DSS histories
grow with examples and epochs (float32 N*10*epochs per learner); at N45K and
150epochs this is about257.5MiB per learner, beyond the pilot's short history.
Temporary history stacking and scoring add memory and CPU cost.

## Rough CPU requirements, not benchmark results

Scale the median measured network warmup update and median main round by
ceil(N/16), published horizons/warmups, and learner count. Add full-dataset
forward scans by linearly scaling the measured128-image scan. DivideMix rounds
are two learner updates and use ceil(f*N/16) iterations/epoch, f in .1,.5,.9.
The table uses f=.5; other methods process all examples. CE/classwise controls
use the base150epoch horizon. Formula is in
[analysis script](../scripts/analyze_strong_pilot.py).

{costs}

{campaign_text}
Full-data GMM fitting, growing CCS history computation, validation, checkpoints
and data-loader overhead are omitted. Few compressed microepochs do not sample
mature selection. Brief pytest overlap occurred during middle single-network
timings; background contention makes this an order-of-magnitude resource screen,
not a fair speed leaderboard. No parallel experiment campaign is assumed.
Batch128 throughput, GPU memory/speed and money cost are unmeasured; linear
rescaling from batch16 cannot establish them.

**Hard stop:** faithful long training is outside the authorized180s pilot and
the current bounded CPU sprint. No large campaign is launched. To reopen the
gate, resolve fidelity ambiguities, measure full-data epoch and batch128 memory,
and approve an explicit device/runtime/run-count budget. This does not establish
that faithful benchmarking is impossible on another provisioned machine.
"""
    (ROOT/"research"/"benchmark_feasibility.md").write_text(feasibility, encoding="utf-8")
    report = f"""# Strong baseline results: executed feasibility pilot only

**Scientific outcome: undetermined.** No converged CIFAR-10 quality comparison
has been executed. The prior synthetic hard-clean/noisy-label failure has neither
survived nor been refuted by faithful DSS or DivideMix training in this sprint.

{table}

These are completed or censored operation-level pilots, not published training
reproductions. Warmup microepochs, partial CCS history and mature-path SSL loss
weight are disclosed in the fidelity reports. Three rounds support no general
uncertainty estimate, ranking of accuracy, or claim that selection is robust.

Losses, timings, GMM posterior fractions, accepted-label counts and failure
reasons are saved in the individual JSON records. They are optimization
sanity/cost observations; GMM posterior is **not measured noise precision**.
The pilot does not open true training-label oracles, validation or test files.
Calibration labels are read only to estimate the published DSS prior.

| Requested quality measure | Status |
|---|---|
| True class-wise recall/F1 | Not executed |
| Hard-clean retention | Not executed; no independent trained reference yet |
| Rejected-noise precision/recall | Not executed |
| ECE/calibration plots | Not executed |
| Macro-F1 / balanced accuracy | Not executed |
| Total faithful training time | Not executed; pilot timings and rough requirements only |

Do not fill these fields with toy-study results, paper accuracies or extrapolated
quality. Fresh development701/702 and evaluation801-805 seeds remain unused.
Classwise small loss here is a known generic control, not a faithful RoLT/PASS
implementation. None of the methods can be declared to solve/fail the realistic
benchmark from these observations. No quality statistical comparisons or
confidence intervals are warranted at this stopping point.

Reproducible provenance: [frozen source/config](../results/strong-baseline-20261012/pilot/freeze.json),
[executed records](../results/strong-baseline-20261012/pilot/results.json),
[summary CSV](../results/strong-baseline-20261012/pilot_summary.csv),
[resource assumptions](benchmark_feasibility.md).
"""
    (ROOT/"research"/"strong_baseline_results.md").write_text(report, encoding="utf-8")
    print(json.dumps({"training_seconds": result["training_seconds"], "statuses": [r["status"] for r in rows], "campaign": campaign}))
