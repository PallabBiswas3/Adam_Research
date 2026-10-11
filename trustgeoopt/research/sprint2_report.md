# TrustGeoOpt — Research Sprint 2

**Decision: defer geometric optimizer implementation.** The original deterministic
AdamW failure does not survive fair LR tuning. Reliability scores mostly capture
known descent dynamics. The remaining stochastic gap is a research question,
not yet a defensible geometric failure mechanism or a distinct optimizer idea.

Work is on `research/sprint2-evidence-validation` from commit `5b75629`.
No push or merge was performed. Existing code, configs, reports and 99 historical
raw attempts were inspected first; historical raw results are preserved.

## Executed evidence

**860 optimizer attempts**: 425 baseline, 180 reliability and 255 correction.
856 successful; four divergent Momentum tuning attempts retain partial traces
and errors. No budget-censored runs. Measured campaign/timing execution totals
**479.59 seconds (~8 minutes)** against a 900-second, 900-attempt CPU cap.
The deterministic scalar published construction and statistical analysis are
separately reported; the scalar run is not included in this measured campaign
timer. No GPU experiment ran. This is bounded toy evidence, not a neural benchmark.

Baselines received six independent LR candidates per regime, two development
seeds and five fresh evaluation seeds, 1,200 steps each. Reliability tested leads
0,1,3,5 at three rates, with independent seeds and five unseen configurations.
Uncertainty resamples whole paired seed clusters, retaining all runs/steps.
Every attempt has config/source/environment provenance; source ZIPs preserve the
code used per stage. Derived tables include individual results, failures,
censoring and frozen selections. Plots and per-step predictions are saved.

## Research answers

1. **Which AdamW failures persist?** None on these tuned stationary diagonal,
   rotated or isotropic quadratics; AdamW also recovers after the tested
   curvature-only switch. Noisy tail loss remains .424 ± .139; Momentum gives
   .271 ± .128 and reference Shampoo .178 ± .037. Tuning boundaries on the latter
   comparators and fixed non-LR knobs limit interpretation. The precise published
   [online Adam counterexample](https://arxiv.org/pdf/1904.09237) reproduces
   separately: average regret .669235 at 30,000 rounds, cycle iterates +1.
2. **Which diagnostic adds robust value?** Alignment has small pooled conditional
   AP lifts on unseen configs at leads 0/1 (+.0209/+.0425), but not uniform regime
   benefit or useful longer-lead lift. History fails incremental unseen validation.
   Actual update/curvature oracle geometry attains AP=1 at lead 0. A robust gate
   for purchasing geometry is unestablished.
3. **Do cheap corrections solve the residual?** The tested fixed AdamW policies
   do not robustly close the noisy gap. Schedule loss .366 has a paired interval
   including no benefit; reset loss .458; clip never activates on selected noisy
   runs. Tuned first-order Momentum already improves the gap. Fewer harmful steps
   do not necessarily mean better optimization.
4. **What would justify low-rank geometry?** A repeatable rotation-dependent
   equal-quality compute deficit surviving broader first-order tuning, with a
   recoverable curvature subspace and demonstrated benefit from off-diagonal
   information. This sprint does not show that mechanism.
5. **Is the direction distinct?** Not established. Residual scaling, directional
   restart, magnitude safeguards, structured coordinates and error-controlled
   refresh have close prior art. The [novelty assessment](novelty_assessment.md)
   and [verified targeted audit](sprint2_literature_audit.md) document it.

## Deliverables and next step

- [Baseline validation](sprint2_baseline_validation.md),
  [reliability analysis](sprint2_reliability_analysis.md),
  [cheap correction results/cost](sprint2_cheap_corrections.md).
- [Frozen CPU config](../configs/synthetic/sprint2.yaml),
  [campaign runner](../src/trustgeoopt/training/sprint2.py),
  [updated evaluator](../src/trustgeoopt/training/reliability_evaluation.py),
  [tests](../tests/test_sprint2.py), [reproduction commands](../README.md#research-sprint-2).
- [Raw results, summaries, paired comparisons and PNG/PDF plots](../results/sprint2/campaign-20261012).
- [Sprint 3 plan](sprint3_plan.md): resolve noise-floor/tuning alternatives,
  validate diagnostics on new seeds/tasks, then reconsider a geometric design.

Validation: **54 tests passed, one CUDA-only test skipped** on this CPU environment.
The [artifact audit](../results/sprint2/campaign-20261012/artifact_manifest.json)
verified all 860 configs, source snapshots, completed horizons and frozen rate
selections, and confirmed **399 tracked historical result files** retain their
original Git blobs (accounting for Windows checkout filters).
The existing PyTorch TypedStorage deprecation warning does not affect results.
Limitations include five independent evaluation seeds, two development seeds,
conditional rather than selection-aware intervals, angle-defined prediction
cohorts, exploratory correction-family selection, and an ungrafted original
Shampoo reference. No novel convergence or publication claim is made.
