# Robust Learning Research — Path B discovery

Status: **strong-baseline audit and bounded CIFAR-10 feasibility pilot completed;
stop before full training**. Current branch is
`research/robust-learning-baseline-validation`, based on the discovery branch.
No novel method or completed seven-calendar-day sprint is claimed.

This is a separate research workspace in Adam_Research. The trustgeoopt directory and its four completed research sprints remain untouched.

## Strong-baseline validation sprint (2026-10-12)

All six real-image operation pilots completed in178.81s, at about1.07GiB peak
process memory: CE, classwise small loss, uniform/prior-aware DSS, prior-aware
DSS+, and DivideMix. These use full PreActResNet-18 backbones with compressed
warmup and three timed main rounds; **no converged accuracy/retention comparison
or novelty claim follows**. Full faithful CPU training exceeds this sprint's
compute scope, so the hard stopping rule applies. No paid GPU campaign was run.

- [Protocol](research/strong_baseline_protocol.md), [DSS fidelity](research/dss_fidelity.md), [DivideMix fidelity](research/dividemix_fidelity.md).
- [Feasibility and resource requirements](research/benchmark_feasibility.md), [actual executed results](research/strong_baseline_results.md), [next decision and novelty overlap](research/next_research_decision.md).
- [Frozen pilot](results/strong-baseline-20261012/pilot/), [summary CSV](results/strong-baseline-20261012/pilot_summary.csv), [preservation checks](results/strong-baseline-20261012/verification.json).

Only the MIT DivideMix backbone and original license are vendored. Unlicensed
DSS source is inspected locally and never copied into the package. CIFAR images,
sealed training oracles, raw source inspections and preserved partial download
stay local/Git-ignored. Hashes, config, equations, tests and all operation records
are committed; download permissions and dataset terms remain separate from
software licensing. Existing result files are never overwritten.

Use the installed compatible CPU environment (see
[recorded versions](requirements_strong_pilot.txt)). To replay in a **fresh**
directory, from this project directory:

```powershell
python -m pytest -q tests -p no:cacheprovider
python scripts/prepare_cifar10.py --output results/strong-baseline-replay
python scripts/replay_strong_pilot.py --output results/strong-baseline-replay
```

Preparation needs the checked original CIFAR archive or network access; the
pilot has the same180s boundary guard and may censor on slower machines. This
has no full-campaign mode. Current report regeneration is
`python scripts/analyze_strong_pilot.py`; the current preservation verifier
needs the ignored local before-inventory and prepared arrays. Do not run the
old discovery report/verifier scripts to update earlier results during this sprint.

**Question to investigate:** When class imbalance, data difficulty, and label corruption interact, which known sample selectors incorrectly reject clean hard/tail examples, and does an inexpensive *additional* signal still help after recent strong baselines?

**Important prior art:** Pan et al. AAAI 2025 already track confidence trends; Pan et al. CVPR 2026 already debias class/instance selection; RoLT (Frontiers of Computer Science 2026) already addresses long-tailed noisy labels; Deng et al. (2024) already target hard clean vs noisy examples. Do NOT present these problem statements, class-wise filtering, confidence history, or similarity scoring as new.

## Starter experiment

The toy data generator exposes the observed label to training and keeps clean labels/corruption flags inside an evaluation-only oracle. The three methods in the smoke script are basic controls: CE, global small-loss, and per-class small-loss. **They are NOT faithful Co-teaching, DivideMix, AAAI 2025, or CVPR 2026 implementations.**

Install Python, numpy, torch, pytest; from this directory run:

    python -m pip install -r requirements.txt
    PYTHONPATH=src python -m pytest -q tests
    python scripts/toy_smoke.py

On Windows PowerShell, set the source path via:

    $env:PYTHONPATH = "src"; python -m pytest -q tests

See CODEX_DISCOVERY_TASK.md for the gated seven-day discovery workflow,
research/protocol.md for data leakage and evaluation controls, and
research/literature_matrix.csv for the expanded source audit.

## Completed discovery pass (2026-10-12)

The targeted literature audit covers 24 primary papers, with incomplete
extractions, version conflicts and software licenses marked explicitly. All
245 planned CPU trials succeeded in 92.30 seconds of experiment execution:
5 pilots, 40 development runs and 200 evaluations (five fresh seeds, eight noise/
prior regimes, five controls/adaptations). The cap was 300 seconds/260 attempts.
No GPU, dataset download or dependency upgrade was used.

Classwise small-loss filtering improved suite-average clean test macro-F1 over
CE by .0513; global filtering and both paper-equation toy adaptations had lower
suite averages. This is a known-control finding on a tiny synthetic problem,
with descriptive five-seed uncertainty. Co-teaching toy and confidence-tracking
toy are **not faithful published benchmark reproductions**. The two-strong-
comparator gate remains incomplete, so the decision is **no-go for method
invention or a novelty claim**. See all regimes and negative results in the reports.

- [Prior-art audit](research/prior_art_audit.md) and [24-paper matrix](research/literature_matrix.csv).
- [Method fidelity](research/method_fidelity.md), [failure study](research/failure_study.md), [decision and next sprint](research/decision.md).
- [Frozen protocol](research/cpu_campaign_protocol.md), [configuration](configs/discovery_cpu.json), [run records](results/discovery-cpu-20261012/runs.jsonl).
- [Tables](results/discovery-cpu-20261012/processed/analysis/), [plots](results/discovery-cpu-20261012/plots/), [verification](results/discovery-cpu-20261012/processed/verification.json).

19 tests passed, covering corruption, split independence, oracle isolation,
seeded training, peer selection/normalization, confidence-gap equations, metrics
and run-level inference. Frozen raw/config/source checks and one deterministic
evaluation replay passed. All 11,242 TrustGeoOpt files, including ignored raw
files and caches, were verified unchanged by SHA256 and file set.

Raw arrays/trajectories, inspected third-party source and the source snapshot zip
are preserved locally but Git-ignored. Committed hashes, complete run records,
operating curves, tables, plots and configuration provide reviewable provenance.
Third-party code is not vendored. Original smoke code and results remain intact.

## Reproduce the bounded pass

Run from this directory using the existing compatible Python environment; the
matplotlib requirement is for analysis/plots. Do not run these commands against
an existing output directory. Each stage is immutable; a failed/interrupted
campaign must be preserved and restarted in a separately named directory.

```powershell
python -m pytest -q tests -p no:cacheprovider
python scripts/run_discovery.py --stage init --output results/discovery-cpu-replay
python scripts/run_discovery.py --stage pilot --output results/discovery-cpu-replay
python scripts/run_discovery.py --stage development --output results/discovery-cpu-replay
python scripts/run_discovery.py --stage evaluation --output results/discovery-cpu-replay
python scripts/analyze_discovery.py --output results/discovery-cpu-replay
```

Do not change source/configuration between stages. Experiment timing is capped;
slower machines may have censored trials, which are retained. Numeric metrics
were deterministically replayed in the recorded environment; runtime, archive
timestamps and environment-specific binary hashes need not match elsewhere.
`build_discovery_report.py` renders the current study tables; run it only when
intentionally updating this report. `verify_discovery.py` checks the recorded
245-run campaign and requires the local before-inventory of TrustGeoOpt; it is
not a generic verification of an arbitrary replay or censored campaign.

## Research gates

1. Audit and reproduce strong relevant prior work; do not rely on toy controls alone.
2. Demonstrate a failure across independent seeds, noise families, difficulty strata, and class priors.
3. Rule out basic class-conditional thresholds, calibrated loss, and the specific improvements of the closest papers.
4. Only then test one additional low-cost statistic as a hypothesis; no new method is presumed.
5. Stop or pivot when prior work closes the gap or fair evaluations negate it.
