# Discovery decision: stop before proposing a method

**No-go for a novelty claim or new method in this pass.** Complete bounded
controls and infrastructure are available; the prerequisite of two faithful,
strong published benchmark reproductions is not met. This is not acceptance of
the null hypothesis or proof that no useful research problem exists.

The task's seven-day workflow describes interactive research gates, not seven
days of unattended compute. Day-1 audit, day-2 data checks, day-3 controls and
day-6/7 descriptive analysis/reporting were completed in a bounded CPU pass.
Days 4-5 were assessed and stopped at fidelity/feasibility; toy adaptations are
not substitutes. The task explicitly allows stopping when faithful baselines
are infeasible. No further benchmark campaign or new method was launched.

## What the evidence establishes

The targeted audit covers 24 relevant primary papers with extraction states and
source/license conflicts. Hard-clean separation is explicit in SLRLNL (2024);
confidence trends in AAAI CT (2025); classwise geometric noisy-label selection
in RoLT (online 2025/journal 2026); class/instance debiasing and an imbalanced
extension in DSS (CVPR 2026); peer agreement in PASS. These sources close several
superficial novelty claims, not every possible residual failure.

245 short CPU trials completed, including 200 evaluations over eight regimes
and five fresh seeds. Known classwise filtering increased equal-weight
across-regime test macro-F1 over CE by .0513 (descriptive seed-block percentile
interval [.0439,.0587]). Global filtering's mean delta was -.0337; CoTeach toy
-.0317; CT toy -.0504. These are suite-specific exploratory summaries. The
published methods cannot be declared inferior from tiny-network adaptations.
Selection curves show precision/retention tradeoffs and infeasible precision
targets; high raw retention is not automatically better noise separation.
In both long-tail instance-noise conditions, classwise diagnostic rankings do
not reach .5 rejected-noise precision in any of the five runs. Higher macro-F1
does not establish a uniformly better selector or a solved selection problem.

Frozen masks, independent clean validation/test splits, development-only rate
selection, deterministic replay, source/config hashes and whole-seed resampling
support the integrity of these controls. They do not remove the limited tuning
grid, short horizon, adaptation differences, five-seed uncertainty or the
confounding between prior-aware difficulty and rarity. Existing remedies were
not shown insufficient. No additional statistic has earned a test yet.

## Falsifiable next question, conditional on benchmark feasibility

Can a faithful prior-aware DSS/CT comparator preserve minority hard-clean
examples at comparable rejected-noise precision **and** clean macro-F1 under
a held-out change of corruption statistics? Reject a residual-gap claim if the
published prior-aware extension or properly scheduled standard controls close
the tradeoff. A measured gap must survive independent development/evaluation
seeds, noise-family changes, compute accounting and the exact official mechanism.

This is a replication/failure-characterization question, not a proposed novel
algorithm. No mathematical contribution is established. Any later analytical
work should make identifiability assumptions explicit: observed label
probabilities alone generally do not uniquely identify both clean posterior
and an unknown instance-dependent transition. A confidence trend or feature
similarity cannot be treated as an oracle without additional assumptions.

## Bounded next sprint proposal (not launched)

1. Resolve source licenses and final-version mappings. Choose two faithful
   comparators, preferably CT/DSS with author clarification or an independently
   documented licensed implementation; MIT DivideMix is another candidate, but
   its full SSL recipe must be retained. Specify benchmark, backbone,
   augmentations, optimizer/schedules, split policy, metric definitions and
   tolerance before runs. Diagnose Co-teaching's paper/code normalization
   discrepancy explicitly.
2. Measure one authentic benchmark pilot's CPU time and memory. Present a
   concrete total trial/compute budget for review before dataset downloads,
   dependency changes or GPU use. If faithful execution is infeasible, stop or
   pursue a replication/infrastructure deliverable instead of a method paper.
3. Use newly registered seeds (for example development 401/402 and evaluation
   501-505), new corruption masks and a frozen held-out configuration. Current
   evaluation results are now unsealed and cannot serve as development-neutral
   evidence for a changed model. Do not retrospectively retune on them.
4. Compare published prior-aware corrections, calibrated/classwise thresholds
   and noise-rate schedules at equal tuning budgets. Use precision/recall
   frontiers with explicit achieved precision, whole-run uncertainty, clean
   macro-F1, counted/peak memory and end-to-end compute. Avoid claiming accuracy
   at a post-hoc oracle operating point without retraining under a deployable rule.
5. Only after a repeatable residual appears, register **one** mechanistic
   hypothesis and incremental signal, including conditioning variables and
   falsification criteria. Otherwise report a negative result. No new optimizer
   or multi-component method is justified by the present evidence.

Full numerical evidence: [failure study](failure_study.md).
Closest prior art: [audit](prior_art_audit.md).
Reproduction gate and deviations: [fidelity](method_fidelity.md).
