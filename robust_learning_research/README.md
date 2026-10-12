# Robust Learning Research — Path B discovery

Status: **bounded discovery pass completed; stop at the faithful-comparator gate**.
No novel method or completed seven-calendar-day sprint is claimed.

This is a separate workspace on the research/robust-learning-discovery branch of Adam_Research. The trustgeoopt directory and its four completed research sprints remain untouched.

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
