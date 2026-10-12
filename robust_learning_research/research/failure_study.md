# Failure study: completed bounded CPU controls

This is a short synthetic discovery pass, not a seven-calendar-day sprint or a
published benchmark reproduction. All 245 attempts succeeded: 5 two-epoch
pilots, 40 development runs and 200 evaluation runs (five seeds, eight regimes,
five methods). Experiment wall time was 92.30 seconds,
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

| Regime | CE | Global | Classwise | CoTeach toy | CT toy |
|---|---|---|---|---|---|
| balanced_symmetric | 0.746 | 0.756 | 0.762 | 0.759 | 0.759 |
| balanced_class_conditional | 0.716 | 0.726 | 0.722 | 0.742 | 0.723 |
| balanced_instance | 0.757 | 0.760 | 0.750 | 0.761 | 0.760 |
| longtail_symmetric | 0.571 | 0.525 | 0.681 | 0.550 | 0.490 |
| longtail_class_conditional | 0.538 | 0.537 | 0.535 | 0.486 | 0.476 |
| longtail_instance | 0.625 | 0.466 | 0.751 | 0.459 | 0.468 |
| heldout_balanced_instance | 0.680 | 0.684 | 0.689 | 0.682 | 0.686 |
| heldout_longtail_instance | 0.499 | 0.411 | 0.654 | 0.440 | 0.368 |


Equal-weight averaging across these eight particular regimes is an exploratory
summary added during analysis, not a population estimate or preregistered primary
endpoint. Resample the five seed blocks, retaining all regimes in a seed together:

| Method minus CE | Across-regime F1 delta | Descriptive 95% interval | Seed blocks |
|---|---|---|---|
| global_small_loss | -0.0337 | [-0.0397, -0.0273] | 5 |
| classwise_small_loss | +0.0513 | [+0.0439, +0.0587] | 5 |
| coteaching_toy | -0.0317 | [-0.0388, -0.0249] | 5 |
| confidence_tracking_toy | -0.0504 | [-0.0628, -0.0414] | 5 |


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

| Regime | Global hard / tail | Classwise hard / tail | CT toy hard / tail |
|---|---|---|---|
| balanced_symmetric | 0.585 / 0.847 | 0.596 / 0.863 | 0.589 / 0.822 |
| balanced_class_conditional | 0.527 / 0.815 | 0.543 / 0.830 | 0.555 / 0.825 |
| balanced_instance | 0.210 / 0.703 | 0.229 / 0.764 | 0.218 / 0.707 |
| longtail_symmetric | 0.557 / 0.207 | 0.688 / 0.952 | 0.572 / 0.087 |
| longtail_class_conditional | 0.493 / 0.229 | 0.593 / 0.626 | 0.512 / 0.000 |
| longtail_instance | 0.035 / 0.000 | 0.471 / 0.871 | 0.213 / 0.000 |
| heldout_balanced_instance | 0.403 / 0.727 | 0.402 / 0.755 | 0.406 / 0.717 |
| heldout_longtail_instance | 0.082 / 0.000 | 0.506 / 0.834 | 0.223 / 0.000 |


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

| Regime | Minimum precision | Classwise feasible | Global feasible | Paired n | Hard retention delta | 95% interval |
|---|---|---|---|---|---|---|
| balanced_symmetric | 0.5 | 5/5 | 5/5 | 5 | +0.007 | [+0.001, +0.014] |
| balanced_symmetric | 0.7 | 5/5 | 5/5 | 5 | +0.007 | [+0.001, +0.014] |
| balanced_symmetric | 0.9 | 5/5 | 5/5 | 5 | +0.007 | [+0.001, +0.014] |
| balanced_class_conditional | 0.5 | 5/5 | 5/5 | 5 | +0.011 | [-0.015, +0.039] |
| balanced_class_conditional | 0.7 | 5/5 | 5/5 | 5 | +0.011 | [-0.015, +0.039] |
| balanced_class_conditional | 0.9 | 1/5 | 0/5 | 0 | NA | NA |
| balanced_instance | 0.5 | 5/5 | 5/5 | 5 | +0.007 | [-0.005, +0.020] |
| balanced_instance | 0.7 | 2/5 | 0/5 | 0 | NA | NA |
| balanced_instance | 0.9 | 0/5 | 0/5 | 0 | NA | NA |
| longtail_symmetric | 0.5 | 5/5 | 5/5 | 5 | -0.159 | [-0.203, -0.110] |
| longtail_symmetric | 0.7 | 3/5 | 5/5 | 3 | -0.158 | [-0.183, -0.111] |
| longtail_symmetric | 0.9 | 0/5 | 4/5 | 0 | NA | NA |
| longtail_class_conditional | 0.5 | 1/5 | 5/5 | 1 | -0.211 | [-0.211, -0.211] |
| longtail_class_conditional | 0.7 | 0/5 | 5/5 | 0 | NA | NA |
| longtail_class_conditional | 0.9 | 0/5 | 3/5 | 0 | NA | NA |
| longtail_instance | 0.5 | 0/5 | 5/5 | 0 | NA | NA |
| longtail_instance | 0.7 | 0/5 | 2/5 | 0 | NA | NA |
| longtail_instance | 0.9 | 0/5 | 0/5 | 0 | NA | NA |
| heldout_balanced_instance | 0.5 | 5/5 | 5/5 | 5 | +0.004 | [-0.004, +0.009] |
| heldout_balanced_instance | 0.7 | 0/5 | 0/5 | 0 | NA | NA |
| heldout_balanced_instance | 0.9 | 0/5 | 0/5 | 0 | NA | NA |
| heldout_longtail_instance | 0.5 | 0/5 | 5/5 | 0 | NA | NA |
| heldout_longtail_instance | 0.7 | 0/5 | 1/5 | 0 | NA | NA |
| heldout_longtail_instance | 0.9 | 0/5 | 0/5 | 0 | NA | NA |


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

| Method | Mean total seconds | Parameter bytes | Optimizer bytes | History bytes |
|---|---|---|---|---|
| ce | 0.330 | 780 | 1576 | 0 |
| global_small_loss | 0.317 | 780 | 1576 | 0 |
| classwise_small_loss | 0.325 | 780 | 1576 | 0 |
| coteaching_toy | 0.489 | 1560 | 3152 | 0 |
| confidence_tracking_toy | 0.386 | 780 | 1576 | 196800 |


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
