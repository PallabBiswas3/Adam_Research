# First-execution report

Date: 2026-10-12. Initial deliverable completed: research architecture, runnable
baseline infrastructure, numerical primitives, tests and recorded CPU experiments.
The full research program is not complete, and no novel optimizer or publication
claim is made. The starting workspace was empty; all work is under `trustgeoopt/`.

## Files and executable architecture

[README](../README.md) gives commands and directory roles. Created package/build
metadata, MIT license, configs for synthetic/MNIST/Fashion-MNIST/CIFAR-10/CIFAR-100,
optimizer/dataset/model/diagnostic/geometry/training/provenance modules, scripts,
automated tests, six phase directories, research notes, generated CSV/BibTeX and
LaTeX skeleton. `artifact_manifest.json` inventories source/config/docs and run artifacts.
Raw records are preserved; figures/tables regenerate from them.

Implemented: PyTorch SGD, momentum, Adam, AdamW/monitor-only, RMSProp, AMSGrad,
AdaGrad, RAdam; AdaBelief published residual equation with explicit decoupled decay;
factor-capped cumulative Shampoo. No Yogi, AdaBound, K-FAC, SOAP or L-BFGS integration
is claimed. TrustGeoOpt/fixed/periodic/gated correction selections are deliberately
unimplemented pending evidence. Independent score interfaces and Woodbury/SVD
geometry primitives exist for later research.

## Literature and mathematical foundation

31 paper records plus one teaching-note source are mapped in
[paper_database.csv](paper_database.csv). Primary metadata verified for Adam,
AdamW, AMSGrad, AdaBelief, RAdam, Yogi, AdaBound, AdaGrad, L-BFGS, natural gradient,
K-FAC, EKFAC, Shampoo, Sophia, SOAP, SENG, Sketchy, Oja/streaming PCA, CASPR,
LDAdam, Purifying Shampoo, ASGO, structured analysis and 2026 dynamics/staleness work.
Selected algorithms/full-text sections were inspected; most theorem/benchmark
details are explicitly unextracted. Several final venues and RMSProp archival
metadata remain unverified. OpenReview access challenges limited direct retrieval.

[Literature review](literature_review.md), [overlap matrix](related_work_matrix.md)
and [novelty assessment](novelty_assessment.md) identify substantial established
mechanisms. Error-driven refresh already appears in Purifying Shampoo (2025), with
staleness-driven damping/frequency in FOAM (2026 preprint). No research gap is verified.

Four hypotheses: causal prediction beyond norm (H1), beneficial off-diagonal
correction in selected problems (H2), lower total compute from selectivity (H3),
and robustness across tasks/seeds/settings (H4). Each has nulls, measurements,
confounds and falsification conditions. Mathematical notes distinguish covariance,
Fisher, Hessian and Gauss-Newton; derive moment lag, frozen quadratic conditioning
and limited smooth descent. PSD rank-one correction can worsen an isotropic problem.
The same-step stochastic gate is not covered by an independent-metric convergence
proof. No theorem for a full momentum/gated optimizer is claimed.

## Verification and commands actually executed

Environment: Windows, Python 3.11.4, PyTorch 2.1.2+cpu, torchvision 0.16.2+cpu.
All required packages were already installed. No CUDA device, dataset downloads,
cloud jobs or dependency installation. Git commit is unavailable; each run records
source SHA-256 and environment versions.

From project root, successful commands:

```powershell
python -m pytest -q
python scripts/run_synthetic.py --config configs/synthetic/smoke.yaml
python scripts/train.py --config configs/synthetic/mlp_smoke.yaml
python scripts/train.py --config configs/synthetic/mlp_smoke.yaml --output results/final_smoke
python scripts/run_benchmarks.py --suite configs/synthetic/pilot_suite.yaml --output results/pilot
python scripts/run_benchmarks.py --suite configs/synthetic/reliability_suite.yaml --output results/reliability
python scripts/evaluate_reliability.py --results results/reliability
python scripts/generate_figures.py --results results/pilot
python scripts/generate_figures.py --results results/reliability
python scripts/generate_figures.py --results results/final_smoke
python scripts/benchmark_geometry.py
python scripts/export_literature.py
```

Final suite: **38 passed, 1 CUDA test skipped**. Tests cover mathematical AdamW/
AMSGrad and AdaBelief references; Shampoo against a dense Kronecker reference;
low-rank solve against dense systems including dependent columns; finite-difference
gradients/HVP; tied-score AP/AUROC; all baseline progress; no-effect monitoring;
optimizer/model/RNG checkpoint continuation; same-seed trajectories; same-time
tracking labels; and CNN/ResNet forward/backward. One installed-PyTorch TypedStorage
deprecation warning is recorded; it does not affect the checks.

## Preliminary evidence and negative findings

80 baseline pilot runs and 15 diagnostic runs succeeded over seeds 0-4; each has
saved raw metrics/config/provenance. [Failure study](failure_study.md) provides
the full means/SD and tuning caveats. Untuned momentum progressed fastest in steps
at the chosen rate; AdamW/Shampoo were not shown superior. This is neither a fair
ranking nor a known Adam counterexample reproduction. No numerical failure occurred.

Reliability: thresholds fitted on seeds 0-1, evaluated on seeds 2-4. Among 1,350
held-out updates, 218 were harmful by the predefined same-time loss-increase label.
Pooled AP: gradient norm 0.1964, scale mismatch 0.2902, negative momentum alignment
0.5585, preconditioner drift 0.1403, negative history consistency 0.5368. Drift/history
exclude nine unavailable startup scores (1,341 updates), so their denominators differ.
Class prevalence is about 0.1615. Alignment's AP by regime is 0.8801 noise-free,
0.8260 curvature/optimum switch and 0.5537 noise shift. Scale mismatch's AP is
0.0846, 0.1346 and 0.4441 respectively; its switch AP is below gradient norm's
0.1393. Drift has poor pooled AUROC (0.4494), not useful support for that score.

These are descriptive, correlated-step pilot metrics with only three evaluation
seeds, not confidence-tested findings. History consistency is competitive and cheaper
than acquiring geometry. Alignment's relation to local descent is already known;
these values do not establish novelty, correction utility or causality. Bootstrap
whole runs and evaluate truly unseen regimes before passing Gate C.

Woodbury/history microbenchmark: 40 measured refresh/solve repetitions at p=128/1024,
k=2/4. Mean relative residuals approximately 2.4e-15 to 2.0e-14; retained-state and
timings are saved. Sub-millisecond timings are local exploratory measurements with
Python tracing/checks; no end-to-end speedup claim follows. On the original tiny
AdamW smoke, diagnostic time exceeded optimizer-step time, illustrating that
instrumentation is not automatically cheap at this scale.

Final synthetic MLP CPU smoke: 512 train/128 validation/128 test examples; 20
correlated input features; two hidden layers of width 32; four epochs/32 updates.
Validation loss 0.70323 to 0.17088, final validation accuracy 95.31%. **Test set
was not evaluated.** This is easy synthetic teacher data, not real-dataset efficacy.

## Blockers and limitations

There is no blocker to running the initial CPU code. Scientific gates A-G remain
open: verified novelty, tuned failure reproduction, independent diagnostic usefulness,
effective correction, total-cost advantage and real-network robustness are unresolved.
CUDA tests cannot run on this machine. Vision benchmarks require datasets and a
planned compute budget. Maintained modern structured baselines remain to integrate.
Timings start after setup and are instrumentation-heavy; concurrent pilot workloads
are unsuitable for efficiency claims. Failed tuning trials and censoring must be retained.

LaTeX source/bibliography exist; compilation was attempted with automatic package
installation disabled. Installed MiKTeX reports unfinished setup and access denied
creating its user-profile directory, so no compiled manuscript PDF is delivered.
User-wide TeX initialization is outside the initial executable research foundation.

## Ordered next iteration

1. Complete full-text/venue/2026 audit, especially Purifying Shampoo/FOAM/LDAdam
   and minibatch-consistency gates. Reassess the exact contribution before coding.
2. Reproduce the published Adam convex construction; tune all baselines with equal
   trial budgets; include isotropic, rotation, noise-shift and mean-shift controls.
3. Expand held-out seeds/regimes, obtain paired-run uncertainty and measure diagnostic
   overhead. Compare alignment with cheap history/norm and simple step safeguards.
4. If useful prediction survives, preregister a dimensionally consistent correction;
   compare random directions and maintained structured remedies. Report negative results.
5. Benchmark separate-process synchronized total compute and memory at equal target
   quality; test rank, damping and refresh ablations before implementing a full gate.
6. Use fixed data/validation splits for MLP/deep MLP/CNN, then MNIST/Fashion/CIFAR
   and ResNet-18 with >=5 evaluation seeds; evaluate test only after model selection.
7. Update the manuscript from actual evidence and proved limited statements.
   Recommend a submission only if novelty, efficacy, compute and robustness gates pass.
