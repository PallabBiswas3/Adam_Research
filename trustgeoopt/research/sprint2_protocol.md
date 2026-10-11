# Sprint 2 protocol (frozen before new experiments)

Scope: CPU float64, one thread, 16-dimensional quadratics with a 4x4 tensor for
reference Shampoo. Existing raw results remain historical. New immutable attempts
live in a separately named campaign. Maximum experiment execution: 900 seconds,
900 attempts; no GPU. Deadline checks every 64 steps. Budget exhaustion leaves
explicit censored attempts and unstarted-trial records. Setup and file I/O count
toward the cap; literature, analysis and tests do not. See `configs/synthetic/sprint2.yaml`.

Each of five baseline optimizers gets six independently specified rates and two
development seeds, separately for each of five regimes (300 tuning attempts).
The rate minimizes mean development log10(mean final-20%-window objective + 1e-12).
Any failed/incomplete tuning seed disqualifies a candidate. Ties choose the lower
rate. Freeze selections before the five evaluation seeds (125 attempts). No
retuning on evaluation data. Report boundary selections, failures, target censoring,
paired log-loss differences and whole-seed bootstrap intervals, preserving pairing
across methods. Five independent evaluation seeds limit inferential strength.
Other knobs are fixed: momentum .9, adaptive betas (.9,.999), zero weight decay;
Shampoo cumulative matrix factors, epsilon 1e-6, refresh each step. Learning-rate
tuning is fair in trial count, not a guarantee of exhaustive tuning of all knobs.

Use matched initialization, curvature bases and additive noise draws across methods.
Stationary target: objective <=1% of initial objective; transient first crossing is
secondary. Primary metric is the final 20% window, not the best iterate. Changing
curvature keeps the optimum fixed, separating curvature from translation; the
unseen reliability configuration changes both and is labeled accordingly. Noise
change and curvature change happen halfway through baseline runs. Also report
late distance and harmful-update fraction: a rising step near the noise floor is
not automatically an optimization failure. No dense Hessians outside these toys.

Reliability: fresh AdamW runs at three predefined rates; two development seeds on
five configurations, five evaluation seeds on these plus five unseen configurations
(180 runs, 600 steps). Development transitions rescaled to step 300. Evaluation
held-out transitions stay at step 200. Exclude first-step undefined angles and
leads crossing known regime transitions. Leads 0,1,3,5 target f_{t+h} increases
under its own objective, without changing the label to f_t. Never fit on evaluation
seeds or held-out configurations. Compare norm/alignment/history/actual proposed
preconditioned direction and fitted nested logistic models. Fit normalization and
L2-regularized models using development only, no model-hyperparameter search.
Control rate, noise scale, phase, update norm and observed directional descent;
also inspect exact synthetic directional curvature, distance and true gradient
as explicitly privileged oracle controls. Exact quadratic change is a sanity
reference, not a deployable diagnostic. No post-update features enter prediction.

Bootstrap whole seed clusters, retaining every step and every paired configuration
and rate within each sampled seed. Report pooled AP (prevalence dependent),
per-regime/phase/rate metrics, macro run metrics, and paired incremental AP and
Brier-score differences. Intervals are conditional on the frozen two-seed training
and finite configuration set; they do not capture tuning/model selection uncertainty.
Exploratory multiple comparisons: no uncorrected significance/publication claims.

Cheap corrections are deferred until A/B are read. If justified, at most three
fixed mechanisms, each six learning rates, same two tuning/five evaluation seeds,
same steps and selection metric. Thresholds/schedules must be written down before
their runs; no tuning on held-out evidence. Measure optimizer plus correction
time separately from dense toy diagnostics, with a separate warmed uninstrumented
CPU timing loop. Corrections are established mechanisms, not novelty claims.

Published counterexample is a separate scalar projected online experiment with
the exact Appendix A assumptions and pre-update regret. Check all cycle-start
iterates equal +1 and positive iterates; do not equate this theorem with ordinary
fixed-rate, bias-corrected AdamW at default betas. Finite numerical evidence checks
the implementation; asymptotic conclusion belongs to the published theorem.

Execution note: the cap timer covers optimizer campaign stages and the separate
correction timing loop (479.59 seconds measured). The small deterministic scalar
counterexample was executed separately without a saved execution timer, like
analysis/test commands. This is a timing-accounting deviation from an all-inclusive
experiment cap, explicitly reported; it is not included in the 479.59-second sum.
No optimizer stage exhausted the cap. Four unstable tuning attempts were retained.
