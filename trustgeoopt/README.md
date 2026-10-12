# TrustGeoOpt research foundation

Executable infrastructure for studying adaptive optimizer reliability in conventional
supervised networks. **No novelty, universal superiority, or publication readiness is
claimed.** TrustGeoOpt remains a design label until evidence supports a prototype.

Start with [the first-execution report](research/foundation_report.md),
[research plan](research/proposal.md), [literature review](research/literature_review.md),
[novelty matrix](research/related_work_matrix.md), and [protocol](research/experimental_protocol.md).

## Quick start

Python 3.11+, PyTorch 2.1+. Run from this directory. The inspected workstation already
had the dependencies; no packages or datasets were installed/downloaded.

```powershell
cd C:\Users\PALLAB\Desktop\Aam\trustgeoopt
python -m pytest -q
python scripts/run_synthetic.py --config configs/synthetic/smoke.yaml
python scripts/train.py --config configs/synthetic/mlp_smoke.yaml
python scripts/generate_figures.py --results results
```

In a fresh environment install `requirements.txt` or `pip install -e ".[test,vision]"`
using a PyTorch/torchvision pair appropriate for the hardware. torchvision is needed
for vision data/ResNet; YAML uses PyYAML. No experiment tracking service is required.

```powershell
python scripts/run_benchmarks.py --suite configs/synthetic/pilot_suite.yaml --output results/pilot
python scripts/generate_figures.py --results results/pilot
python scripts/run_benchmarks.py --suite configs/synthetic/reliability_suite.yaml --output results/reliability
python scripts/evaluate_reliability.py --results results/reliability
python scripts/benchmark_geometry.py
python scripts/export_literature.py
```

All attempts get unique result folders. Rerunning does not replace raw runs; derived
figures/tables are rebuilt from them. Avoid pooling repeated configurations as new
independent seeds. Suite manifests retain failures; raw run metadata includes source
hash, package/device details, config, timing, memory and status. Timing excludes setup;
CPU memory is process high-water, not an isolated experiment peak. CUDA timing is
synchronized; CUDA execution has not been verified on this CPU workstation.

## Implemented boundary

- Official PyTorch SGD, momentum, Adam, AdamW, RMSProp, AMSGrad, AdaGrad and RAdam.
- Precisely specified AdaBelief Algorithm 2 reference with decoupled decay; small,
  ungrafted matrix Shampoo with explicit unfolding, vector AdaGrad and factor cap.
- Diagonal/rotated/noisy/time-varying quadratics and Rosenbrock; true loss separated
  from noisy observed gradient. Same-time tracking labels.
- Independent causal reliability scores; optional two-gradient cosine and autograd HVP.
- Bounded rolling gradient SVD, centered/uncentered moment factors, random-basis
  controls, Cholesky/Woodbury solves. No dense neural parameter Hessian.
- MLP/deep MLP, CNN and torchvision ResNet-18; synthetic supervised data and optional
  standard vision datasets. Validation-only selection by default.
- Checkpoints preserve model, optimizer, global RNG, loader/noise RNG and observer
  state. Both experiment runners support resumption with only budget extension.

```powershell
python scripts/train.py --config configs/synthetic/mlp_smoke.yaml --resume results/raw/<run-id>/checkpoint.pt
python scripts/run_synthetic.py --config configs/synthetic/smoke.yaml --resume results/raw/<run-id>/checkpoint.pt
```

Increase `epochs` or `steps` first; same-budget resumption fails clearly. Synthetic
checkpoints are controlled by `checkpoint_interval`, trainer checkpoints every epoch.
`--evaluate-test` is for a final evaluation after validation-driven model selection.
Do not repeatedly evaluate test data while tuning. Vision configs set `download: false`;
populate data or explicitly change that setting before a planned vision run.

## Repository map

| Directory | Purpose |
| --- | --- |
| src/trustgeoopt/optimizers | reference baselines and reserved prototype boundary |
| src/trustgeoopt/diagnostics | causal state statistics, HVP and alignment |
| src/trustgeoopt/geometry | bounded sketches and low-rank algebra |
| src/trustgeoopt/models, datasets | conventional networks and fixed splits |
| src/trustgeoopt/training, utils | experiment execution, state, metrics, provenance |
| configs, experiments, scripts | reproducible phase plans and entry points |
| tests | mathematical references, checkpoint and reproducibility checks |
| research | source database, mathematical notes, evidence and decision gates |
| results | immutable raw data and reproducible figures/tables |
| paper | cautious LaTeX skeleton and generated verified-record bibliography |

TrustGeoOpt, fixed/periodic/gated corrections deliberately raise an explicit
`NotImplementedError` in the optimizer factory. Implementing them now would bypass
the user's evidence gates. Yogi/AdaBound/L-BFGS/K-FAC are reviewed, not integrated.
Sprint 3 separately evaluates a pinned authors' SOAP reference (see below).
The factor-capped Shampoo reference rejects large factors; a maintained modern
structured baseline remains required for serious CNN/ResNet comparisons.

Primary campaign grids are guarded by a maximum run count and are not launched here.
The initial pilots use untuned rates and include measurement overhead; they establish
reproducibility, not optimizer rankings. See the report for negative findings and next steps.

## Research Sprint 2

The evidence-validation sprint preserves the original pilots. See
[`research/sprint2_report.md`](research/sprint2_report.md),
[`research/sprint2_baseline_validation.md`](research/sprint2_baseline_validation.md),
and [`research/sprint2_reliability_analysis.md`](research/sprint2_reliability_analysis.md).
The new CPU campaign has independent rate tuning, fresh evaluation seeds, unseen
reliability configurations, whole-seed bootstrap intervals and an exact published
online Adam counterexample. The full TrustGeoOpt optimizer remains gated.

Reproduce in a **new** destination (the runner rejects existing campaigns/stages):

```powershell
python scripts/run_sprint2.py --stage init --output results/sprint2/my-new-campaign
python scripts/run_sprint2.py --stage baseline --output results/sprint2/my-new-campaign
python scripts/run_sprint2.py --stage reliability --output results/sprint2/my-new-campaign
python scripts/run_sprint2.py --stage counterexample --output results/sprint2/my-new-campaign
python scripts/analyze_sprint2.py --results results/sprint2/my-new-campaign --stage baseline
python scripts/analyze_sprint2.py --results results/sprint2/my-new-campaign --stage reliability
```

After reading those findings and the fixed correction protocol:

```powershell
python scripts/run_sprint2.py --stage corrections --output results/sprint2/my-new-campaign
python scripts/analyze_sprint2.py --results results/sprint2/my-new-campaign --stage corrections
python scripts/benchmark_sprint2_corrections.py --results results/sprint2/my-new-campaign
python scripts/plot_sprint2_counterexample.py --results results/sprint2/my-new-campaign
python scripts/audit_sprint2.py --results results/sprint2/my-new-campaign
python -m pytest -q
```

`adamw_step_clip`, `adamw_momentum_reset` and `adamw_lr_safeguard` are fixed
first-order research adaptations, not TrustGeoOpt or novel mechanisms. The
campaign cap is 900 seconds of campaign execution and 900 optimizer attempts;
the small deterministic scalar counterexample and analysis are separately reported.
New campaigns snapshot current code; exact historical reproduction uses the
per-stage source ZIPs and the saved configurations. Corrections were selected
after inspecting A/B, so their evaluation is explicitly exploratory.

Git keeps compact experiment summaries, plots, configurations and provenance
manifests. Generated per-run `results/**/raw/` directories, source snapshot ZIPs,
compressed per-step prediction dumps and model checkpoints are ignored and remain
on disk. Archive these separately when sharing the complete experimental evidence;
a normal Git commit does not include them. Previously tracked results remain tracked.

## Research Sprint 3

See [`research/sprint3_report.md`](research/sprint3_report.md) and
[`research/sprint4_decision.md`](research/sprint4_decision.md). The CPU-only
mechanism study includes exact Sprint 2 replay, broader stochastic tuning,
coordinate-matched geometry controls, small/deep ReLU MLPs and a Fashion-MNIST
subset. The frozen protocol specifies 1,195 planned attempts, a 1,200-attempt
cap and 1,800 seconds of experimental execution. A new optimizer is still gated.

The unmodified authors' preliminary SOAP reference and MIT license are vendored;
commit and hashes are in `src/trustgeoopt/optimizers/vendor/provenance.json`.
This is not production distributed Shampoo. Its maximum factor dimension is
128. Current Meta Shampoo requires newer Python/PyTorch than this environment.

From `trustgeoopt/`, populate Fashion-MNIST using the official dataset resources
listed in `research/sprint3_data_provenance.json` (or torchvision's download
facility), then run in a new destination:

```powershell
python scripts/run_sprint3.py --stage init --output results/sprint3/my-new-campaign
python scripts/run_sprint3.py --stage replication --output results/sprint3/my-new-campaign
python scripts/run_sprint3.py --stage stochastic --output results/sprint3/my-new-campaign
python scripts/run_sprint3.py --stage geometry --output results/sprint3/my-new-campaign
python scripts/run_sprint3.py --stage neural --output results/sprint3/my-new-campaign
python scripts/analyze_sprint3.py --output results/sprint3/my-new-campaign
python scripts/summarize_sprint3_mechanisms.py --output results/sprint3/my-new-campaign
python scripts/audit_sprint3.py --output results/sprint3/my-new-campaign
python -m pytest -q
```

Replication reads the preserved local Sprint 2 protocol/results and is excluded
from tuning. Fresh selection uses development seeds 30/31 and evaluation seeds
40–44. Re-executing a completed stage or changing source mid-campaign fails
clearly; create a new campaign. Raw trajectories and source ZIPs remain locally
ignored; committed manifests, summaries, selections, comparisons and plots are
compact. The independent audit reads earlier results without rewriting them.

## Research Sprint 4

Read [`research/sprint4_report.md`](research/sprint4_report.md), the
[`noise/curvature theory`](research/sprint4_noise_curvature_theory.md), and
[`Sprint 5 decision`](research/sprint5_decision.md). The frozen CPU campaign
executes 966 attempts, including six pilots, in 595.71 seconds. It compares
fairly selected baseline policies, known noise/curvature estimators and small
ReLU MLP layer geometry. The neural accuracy/cost gate fails; no new optimizer,
deep/Fashion expansion, GPU campaign or dependency upgrade is launched.

From `trustgeoopt/`, reproduce in a **new** directory. Keep the source unchanged
between initialization and experiment stages:

```powershell
python scripts/run_sprint4.py --stage init --output results/sprint4/my-new-campaign
python scripts/run_sprint4.py --stage pilot --output results/sprint4/my-new-campaign
python scripts/pilot_sprint4_probes.py --campaign results/sprint4/my-new-campaign
python scripts/run_sprint4.py --stage tuning --output results/sprint4/my-new-campaign
python scripts/run_sprint4.py --stage evaluation --output results/sprint4/my-new-campaign
python scripts/run_sprint4.py --stage estimators --output results/sprint4/my-new-campaign
python scripts/run_sprint4.py --stage neural --output results/sprint4/my-new-campaign
python scripts/analyze_sprint4.py --campaign results/sprint4/my-new-campaign
python scripts/summarize_sprint4.py --campaign results/sprint4/my-new-campaign
python scripts/audit_sprint4.py --campaign results/sprint4/my-new-campaign --create-manifest
python -m pytest -q
```

Select only with development seeds 50–59; evaluation seeds 60–69 never enter
selection. The campaign rejects existing destinations/started stages and records
budget censoring/failures. Read the report's timing limitations: step-indexed
noise changes confound retrospective rising/falling equal-time comparisons.
`RandomizedCurvature` is an established matrix-free range/Ritz measurement
routine with configurable rank/refresh, not TrustGeoOpt or a novel optimizer.

The historical campaign retains its original source ZIP/hash plus recorded
pilot/analysis drivers. Final source adds audit/analysis and small robustness
checks, separately snapshotted. Exact historical reproduction uses the original
snapshot; a new run captures current source. Compact run indexes, estimator
rows, intervals, plots and manifests are committed. Raw sampled gradients,
per-step trajectories, earlier-evidence inventory and source ZIPs remain local
and ignored; archive them separately to share complete raw evidence.
