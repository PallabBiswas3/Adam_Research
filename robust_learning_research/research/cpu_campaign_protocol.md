# Frozen bounded CPU discovery pass

Freeze before pilot/tuning/evaluation, 2026-10-12. This pass covers data/control
checks and paper-equation toy adaptations; it does not complete the two faithful
published benchmark-reproduction gate or seven calendar days of work.

Cap 300 seconds of experiment execution, 260 attempts, one CPU thread.
Planned 245 attempts: five 2-epoch pilots; 40 development runs (five methods ×
two rates × two development seeds × two regimes); 200 evaluations (five methods ×
eight regimes × five seeds). Imports, literature inspection, tests, analysis and
hashing separately reported. No GPU, cloud, dataset download or dependency upgrade.

MLP 2→32 ReLU→3; AdamW zero weight decay, default moments; batch64,20epochs,
same first-model initialization and independent seeded minibatch order across
methods. CE/global loss/classwise loss share a three-epoch warmup and .7keep
fraction. Filtering losses average over selected examples. Co-teaching uses two
independent networks and peer-exchanged minibatch mean losses; forget rate rises
linearly to the nominal .3noise parameter over three epochs. This uses generation
settings, never realized corruption flags; instance noise's actual rate is lower.
Confidence-tracking uses probability gaps, full post-epoch history, min competitor
Mann–Kendall Z>1.64485 and union with global .7small-loss samples. Its masked CE
averages over the whole minibatch as in paper Eq.12; this differs in scale from
selected-mean controls and is disclosed. No novel statistic or extra component.

Co-teaching and CT are **independent paper-equation toy adaptations**, not official
benchmark/code reproductions. Image architectures, augmentations, horizons,
validation protocols and resource costs differ. Official source/license/fidelity
inspection determines whether the stronger workflow gate can proceed; do not
silently relabel these adaptations as reproductions.

Development seeds201/202; independent evaluation301–305. Select each method's
rate from .003/.01 by mean final clean-validation macro-F1 over balanced symmetric
and long-tail instance-noise development conditions. Require all four trials per
candidate to finish. Clean validation is independently generated, balanced360
examples. Training true labels never enter selection/tuning. Final clean TEST
macro-F1/balanced accuracy/ECE use another independent balanced360examples;
evaluation validation values never select rates/checkpoints. No early stopping.

Six conditions: near-balanced (274,273,273) / long-tail (600,160,60) × symmetric,
cyclic class-conditional or instance-dependent flips. All have820train points.
Task300 rotation and std1.15; two held-out regimes use new task301/std1.45and
instance noise. Inputs, noise and order have separate RNG streams; immutable
flip masks and labels saved before training. The instance probability is
.3(.25+.75Bayes-ambiguity), so it is a cap, not an exact corruption proportion.
Known prior-aware Bayes ambiguity≥.5defines hard cases in evaluation only;
rarity and difficulty are not causally independent in this toy.

Primary diagnostics: actual final training selection, hard-clean/tail-clean
retention; full frozen-score operating curves; clean test macro-F1, balanced
accuracy and10-bin ECE; total run/step/wall cost, parameter/state bytes. CE
accepts all; loss-based post-hoc curves on CE are diagnostic hypothetical filters,
not CE training selection. Co-teaching final diagnostic uses averaged peer loss
scores; actual training acceptance is recorded separately. CT diagnostic ranks
max(base loss rank, normalized gap Z), an explicitly descriptive ranking that
does not change its fixed union training selector.

At matched rejected-noise precision targets .5/.7/.9, select a point on each
frozen diagnostic curve with best hard-clean retention, tie by tail retention.
This is oracle-assisted POSTPROCESSING only, never a deployable threshold or
selection/tuning input. Requested fractions respect ties, actual fractions may
differ. No interpolation across nonmonotone PR points; infeasible targets are
reported. Pair differences by whole independent evaluation seed, bootstrap4000
resamples; retain all phase/layer measurements within runs. Report feasibility
and conditional comparisons, not independent-step p-values or multiplicity-
corrected population claims. No robust novelty or prior-art superiority claim
is possible until faithful strong-comparator gates pass.
