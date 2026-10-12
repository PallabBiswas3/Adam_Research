"""Render evidence tables from frozen results; does not fit or select models."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(output):
    analysis = output/"processed"/"analysis"
    summary = read_csv(analysis/"summary.csv")
    pairs = read_csv(analysis/"paired_comparisons.csv")
    matched = read_csv(analysis/"matched_comparisons.csv")
    aggregate = read_csv(analysis/"aggregate_comparisons.csv")
    runs = read_csv(analysis/"run_metrics.csv")
    state = json.loads((output/"campaign.json").read_text())
    methods = state["protocol"]["methods"]
    regimes = [r["name"] for r in state["protocol"]["regimes"]]
    def value(regime, method, metric):
        row = next(r for r in summary if r["regime"] == regime and r["method"] == method and r["metric"] == metric)
        return f"{float(row['mean']):.3f}"
    def table(header, rows):
        return "| "+" | ".join(header)+" |\n|"+"|".join("---" for _ in header)+"|\n"+"\n".join("| "+" | ".join(row)+" |" for row in rows)+"\n"
    f1table = table(["Regime", "CE", "Global", "Classwise", "CoTeach toy", "CT toy"],
        [[regime]+[value(regime, m, "macro_f1") for m in methods] for regime in regimes])
    retention = table(["Regime", "Global hard / tail", "Classwise hard / tail", "CT toy hard / tail"],
        [[regime]+[value(regime, m, "hard_clean_retention")+" / "+value(regime, m, "tail_clean_retention")
                    for m in ["global_small_loss", "classwise_small_loss", "confidence_tracking_toy"]] for regime in regimes])
    deltas = table(["Method minus CE", "Across-regime F1 delta", "Descriptive 95% interval", "Seed blocks"],
        [[r["method"], f"{float(r['mean']):+.4f}", f"[{float(r['lower']):+.4f}, {float(r['upper']):+.4f}]", r["n"]] for r in aggregate])
    matchtable = table(["Regime", "Minimum precision", "Classwise feasible", "Global feasible", "Paired n", "Hard retention delta", "95% interval"],
        [[r["regime"], r["target"], r["method_feasible"]+"/5", r["reference_feasible"]+"/5", r["n"],
          f"{float(r['mean']):+.3f}" if r["mean"] else "NA",
          f"[{float(r['lower']):+.3f}, {float(r['upper']):+.3f}]" if r["mean"] else "NA"]
         for r in matched if r["method"] == "classwise_small_loss" and r["metric"] == "hard_clean_retention"])
    costs = []
    for method in methods:
        group = [r for r in runs if r["method"] == method]
        mean_seconds = sum(float(r["total_seconds"]) for r in group)/len(group)
        costs.append([method, f"{mean_seconds:.3f}", group[0]["parameter_bytes"], group[0]["state_bytes"], group[0]["history_bytes"]])
    costtable = table(["Method", "Mean total seconds", "Parameter bytes", "Optimizer bytes", "History bytes"], costs)
    failure = f"""# Failure study: completed bounded CPU controls

This is a short synthetic discovery pass, not a seven-calendar-day sprint or a
published benchmark reproduction. All 245 attempts succeeded: 5 two-epoch
pilots, 40 development runs and 200 evaluation runs (five seeds, eight regimes,
five methods). Experiment wall time was {state['experiment_seconds']:.2f} seconds,
below the frozen 300-second/260-attempt ceiling. No failures or censored trials
occurred; source-fetch failures are retained separately. No GPU or dependency
upgrade was used. Literature inspection, imports, analysis, tests and file
verification are outside that experiment timer.

## Clean held-out classification

Values are mean clean TEST macro-F1 across seeds 301-305. The LR policy was
frozen on development seeds 201/202 and a different task before any test/oracle
metrics were computed. Validation and test are separate balanced 360-point
samples. Global/CE/CoTeach choose .01; classwise/CT choose .003. This two-point
grid is a bounded control study, not evidence of globally optimal tuning.

{f1table}

Equal-weight averaging across these eight particular regimes is an exploratory
summary added during analysis, not a population estimate or preregistered primary
endpoint. Resample the five seed blocks, retaining all regimes in a seed together:

{deltas}

Classwise filtering helps on this chosen suite on average, whereas the global,
CoTeach toy and CT toy averages are lower than CE. Individual regimes differ;
report all of them rather than declaring a universal ranking. Intervals use
4000 percentile bootstrap resamples of **whole runs/seeds**, not training steps
or individual samples. Five seed blocks support only descriptive uncertainty;
these are not simultaneous intervals, corrected hypothesis tests, or proof of
superiority over published algorithms. Per-regime paired contrasts and F1/ECE
intervals are in the processed tables.

## Selection tradeoff

These are actual last-epoch training acceptance rates, not post-hoc scores:

{retention}

CE retains every hard/tail-clean point and every mislabeled point. Global loss
filtering often rejects hard/tail-clean points. Classwise quotas alter that
tradeoff, while the noisy **observed** labels determine the quota groups. Hardness
is predeclared prior-aware Bayes ambiguity >= .5; rarity and hardness are not
independent in this mixture. The tail stratum is true class2 even in balanced
controls, where it is a class control rather than a minority. Empty strata would
be NA, never fabricated zeros.

The realized instance-noise rate is below the nominal .3 cap. Thus a forget rate
or .7 keep rate tied to nominal corruption can reject too much clean data. CT's
loss scaling, short trend horizon and post-epoch sampling differ from the official
benchmark. None of these causal explanations was isolated by an ablation here;
they are limitations to test, not established explanations of paper failure.

## Precision-recall operating points

Full frozen-score curves use ten predeclared requested keep fractions and retain
ties. Noise rejection is the positive event: rejected-noise precision is the
fraction of rejected examples actually flipped; noise recall is the fraction of
all flips rejected. Accepted-clean precision uses the different accepted set.
Hard/tail-clean retention denominators contain only unflipped examples.

At a **minimum** rejected-noise precision .5/.7/.9, the best hard-retention point
on each curve is selected by the evaluation oracle; achieved precision can exceed
the target and differs by method. These are minimum-precision feasibility
comparisons, not exact matched precision, deployable thresholds or retrained
models. Taking the best of ten points is optimistic/descriptive. No interpolation
or monotone envelope is imposed on nonmonotone curves. Infeasible targets remain
NA; intervals with n<5 are conditional on paired feasibility and especially weak.

{matchtable}

The classwise diagnostic curve fails to reach even .5 rejected-noise precision
in all five runs of each long-tail instance-noise setting, despite its higher
test macro-F1. For long-tail symmetric noise it has lower hard-clean retention
than global filtering at the .5 minimum precision. Better class coverage is
therefore not established as better noise separation. Degenerate intervals
when n=1 reflect one observed run, not certainty about a population.

Full paired tail-retention and all-method results, achieved precision and
acceptance fractions are retained in CSV. CE curves represent a hypothetical
loss filter; CoTeach averages peer losses; CT combines loss rank and sigmoid Z
for descriptive ranking. Those curves do not describe their actual training
selection rule. Macro-F1 is measured after the original frozen training, so a
post-hoc operating point cannot imply downstream accuracy at that point.

## Resource measurements

{costtable}

Bytes count model parameters, optimizer tensors and probability history, not
peak RAM. Total run seconds include data generation, storage and final evaluation;
loop seconds also include clean validation. CoTeach uses two learners; CT stores
20 x 820 x 3 probabilities. Selector-only timing omits minibatch peer selection
and is not a fair total-overhead metric. Timings are single-thread local CPU
measurements; no portable throughput or time-to-target claim follows.

## Evidence limits and artifacts

The study reproduces the known small-loss/imbalance tradeoff under controlled
noise; it does not identify a residual failure of a faithful strong comparator.
No new statistic, optimizer, or multi-component method was developed. H1 has
descriptive support; H0/H2/H3 remain undecided behind the fidelity gate. H4 has
limited support for a known control in this suite, with unmatched training
precision and no causal hard-retention intervention.

- [Frozen protocol](cpu_campaign_protocol.md), [fidelity](method_fidelity.md), [literature](prior_art_audit.md).
- [Run records](../results/discovery-cpu-20261012/runs.jsonl), [analysis tables](../results/discovery-cpu-20261012/processed/analysis/), [plots](../results/discovery-cpu-20261012/plots/).
- [Raw-file hashes](../results/discovery-cpu-20261012/processed/raw_manifest.json), [verification](../results/discovery-cpu-20261012/processed/verification.json).

Raw arrays/trajectories and the original source zip are preserved locally in
ignored paths; hashes are committed, while tables/curves/run records/plots are
published. Regenerate a fresh directory using the frozen config and unchanged
training/data modules; do not overwrite or resume an existing stage. Original
day-zero results are not pooled with the independent campaign.
"""
    (ROOT/"research"/"failure_study.md").write_text(failure, encoding="utf-8")
    # Expand the original source map; keep its original columns for consumers.
    audit = (ROOT/"research"/"prior_art_audit.md").read_text(encoding="utf-8")
    entries = []
    for line in audit.split("## Closest recent mechanisms")[0].splitlines():
        if not line.startswith("| ["):
            continue
        cells = [s.strip() for s in line.strip("|").split("|")]
        source, date, mechanism, benchmark, overlap = cells
        first = source.split("](")[0].lstrip("[")
        url = source.split("](")[1].split(")")[0]
        entries.append(dict(year=date[:4] if date[:4].isdigit() else "see date_venue", work=first,
            venue=date, verified_url=url, known_mechanism=mechanism, research_overlap=overlap,
            review_level=mechanism.split(",")[0], date_venue=date, benchmark_compute=benchmark,
            code_license_state="See prior_art_audit.md source table; otherwise unverified"))
    assert len(entries) == 24
    with (ROOT/"research"/"literature_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(entries[0])); writer.writeheader(); writer.writerows(entries)
    print("Rendered failure study and 24-paper matrix from evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(ROOT/args.output)
