# Sprint 3 stochastic analysis

**The original noisy AdamW ranking does not survive the broader protocol.**
Its saved Sprint 2 results reproduce exactly, but cosine schedules and beta2
tuning reverse the ranking for rising noise. A falling-noise weakness is
heterogeneous and sensitive to development-seed selection; it is not evidence
that off-diagonal curvature is the missing ingredient.

Evidence: `results/sprint3/campaign-20261012/`; frozen
[protocol](sprint3_protocol.md), `processed/development_candidates.csv`,
`stochastic_selection.json`, `summary.csv`, `paired_comparisons.json`, and
`plots/stochastic_transients.png`. All losses below are true quadratic losses,
not noisy function estimates.

## Reproduction and expanded search

Fifteen replays at the **previously selected** configurations give mean tail
loss 0.4242491249 (AdamW), 0.2708251830 (Momentum SGD), 0.1775665054 (reference
Shampoo), with maximum per-run difference from saved Sprint 2 values **zero**.
Those old evaluation seeds never enter new selection.

Fresh development seeds 30/31; evaluation 40–44; 2,400 calls rather than
1,200. Each method receives 24 candidates per regime, matched streams and
horizons: rates plus beta2/momentum/refresh and constant/cosine schedules.
Momentum's lower range extends from 1e-4 down to 1e-6; Shampoo extends its
upper range from 3 to 10. AdamW beta2 is .9/.99/.999, beta1 .9. Cosine ends
at 1% of its initial rate. Weight decay is zero. 432 tuning + 45 evaluation
attempts all succeed. The remaining falling-SGD upper boundary .0003 is
flagged; selected configurations are bounded-search choices, not global optima.

| Noise regime | AdamW mean ± run SD | Momentum mean ± run SD | Shampoo mean ± run SD |
| --- | ---: | ---: | ---: |
| .1 → 10 at call 1,201 | .01634 ± .01016 | .11278 ± .06100 | .04920 ± .02220 |
| Constant 10 | .10196 ± .05299 | .15549 ± .10595 | .50584 ± .91591 |
| 10 → .1 at call 1,201 | .03535 ± .05053 | .0003702 ± .0001933 | .0001078 ± .0000144 |

Selected AdamW settings: rising (.003, beta2=.9, cosine), constant (.03,
beta2=.999, cosine), falling (.003, beta2=.9, cosine). Momentum selects .0001
with .9 momentum/cosine for rising and constant, .0003/.9/constant for falling.
Shampoo selects .3/refresh1/cosine, .3/refresh10/cosine, and 3/refresh1/constant.

## Uncertainty changes the interpretation

Whole-run paired log10(loss_method/loss_AdamW), 2,000 seed-pair resamples:

| Regime | Momentum contrast [95% interval] | Shampoo contrast [95% interval] |
| --- | ---: | ---: |
| Rising | +.8445 [.4292, 1.1998] | +.5018 [.4094, .5942] |
| Constant | +.1428 [-.2270, .4330] | +.2661 [-.1714, .8331] |
| Falling | -.6671 [-1.9415, .5856] | -1.1459 [-2.4289, .1371] |

Positive means higher loss than AdamW. The rising advantage transfers in this
small conditional study. Constant/falling comparisons have wide intervals;
arithmetic mean differences alone exaggerate certainty. Intervals condition on
two development seeds and do not include selection uncertainty or multiplicity.

For falling noise, the chosen AdamW candidate achieved 3.11e-5 and 5.04e-5
on development seeds. Evaluation losses are 5.49e-5, 3.97e-4, .10968, .06654
and 4.74e-5. Two bad trajectories dominate the mean. Development also contains
a constant-rate .03/beta2=.999 candidate with mean 1.16e-4, demonstrating that
the selected cosine schedule is not the only plausible first-order setting.
It was **not** retrospectively selected or evaluated using these evaluation
seeds. More development seeds, not a new optimizer, are the next defensible
response. Shampoo's constant-noise mean also has a large evaluation outlier.

## What explains the observed effects?

The development schedule table shows cosine improves the best development
score for all rising/constant methods, while constant rate wins for falling
SGD/Shampoo. Rate, horizon and schedule therefore materially affect the
transient/noise tradeoff. The old noisy deficit cannot be interpreted as an
intrinsic adaptive-preconditioning failure. The new study changes horizon and
schedule jointly relative to Sprint 2; it does not attribute the entire
improvement to one change.

Beta2=.9 is selected after both noise shifts, whereas .999 is selected for
constant noise. This is consistent with a responsiveness/variance tradeoff,
but only a development observation. The selected falling beta2 has a pure-noise
EMA half-life of about seven steps; its large late errors cannot simply be
blamed on beta2=.999's long memory. Insufficient remaining movement under
decay and seed-dependent slow directions are hypotheses, not isolated causes.
The raw step norms, directional curvatures, second-moment means and curves
remain available for a fresh validation study.

The [mathematical investigation](sprint3_mathematics.md) proves a discrete
fixed-B noise floor proportional to eta to first order, and an exact covariance
recurrence separating mean transient from variance. It also gives exact
pure-noise beta2 EMA moments and response. Replacing Adam's changing denominator
with its noise-scale expectation is explicitly an approximation; we have not
quantitatively decomposed actual Adam's tail loss using that approximation.

Noise covariance in this stage retains the Sprint 2 independent anisotropic
construction. The [geometry study](sprint3_geometry_analysis.md) separately
changes covariance orientation at matched trace/spectrum. A structured method
having lower loss under one cumulative normalization is not proof that it
estimated curvature better. No diagnostic or correction was selected using
the new evaluation failures.

**Decision:** reject the blanket Sprint 2 noisy ranking as a mechanism. Retain
noise-drop/decay sensitivity as an unresolved, inexpensive first-order tuning
question. No evidence here establishes a novel geometric remedy.
