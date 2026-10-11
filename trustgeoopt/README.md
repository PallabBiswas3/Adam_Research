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
the user's evidence gates. Yogi/AdaBound/L-BFGS/K-FAC/SOAP are reviewed, not integrated.
The factor-capped Shampoo reference rejects large factors; a maintained modern
structured baseline remains required for serious CNN/ResNet comparisons.

Primary campaign grids are guarded by a maximum run count and are not launched here.
The initial pilots use untuned rates and include measurement overhead; they establish
reproducibility, not optimizer rankings. See the report for negative findings and next steps.
