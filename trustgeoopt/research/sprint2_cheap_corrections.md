# Sprint 2 cheap correction baselines

**Finding:** none of the three fixed AdamW remedies robustly closes the noisy
tail-loss gap. A tuned Momentum SGD baseline already improves that gap without
geometric information. Resetting momentum reduces local harmful updates, but
that does not imply improved sustained stochastic quality.

The [correction protocol](sprint2_correction_protocol.md) was written after
baseline/reliability analysis and before these runs. Mechanisms are established:
update-norm clipping, observed-gradient alignment restart adapted to AdamW, and
a fixed inverse-square-root LR schedule. They use no loss/Hessian/future label.
See [implementation](../src/trustgeoopt/optimizers/cheap_corrections.py).
These are research adaptations, not faithful reproductions of every cited
algorithm and not new optimizer contributions. Related-paper extraction limits
are recorded in [the literature audit](sprint2_literature_audit.md).

Each policy received the same six AdamW LR candidates, two development seeds,
five evaluation seeds, five regimes and 1,200-step horizon as ordinary AdamW:
180 tuning plus 75 selected evaluation trials, all successful. Fixed cap=.1;
reset gate is observed-gradient dot prospective displacement >0; schedule is
base_lr/sqrt(1+t/200). No policy threshold search. Policies were motivated by the
already observed A/B evaluation, so this is exploratory, not independently
confirmed family selection. No unseen-configuration transfer claim is made.

## Quality and local harmful steps

| Method | Noisy tail loss, mean ± SD | Noisy harmful fraction | Changing-curvature tail loss |
| --- | ---: | ---: | ---: |
| Tuned AdamW | .424 ± .139 | .223 | 1.58e-12 |
| Step clip | .424 ± .139 | .223 | 1.42e-4 |
| Momentum reset | .458 ± .183 | .209 | 2.70e-15 |
| LR safeguard | .366 ± .093 | .255 | 1.50e-6 |
| Tuned Momentum SGD reference | .271 ± .128 | .229 | 7.81e-43 |
| Tuned Shampoo reference | .178 ± .037 | .255 | 4.04e-4 |

Noisy selected rates: .01 for clipping/reset, .03 for the schedule; ordinary
AdamW selected .01. Clipping had **zero** events on every selected noisy run,
so its equality is expected, within roundoff. Whole paired-seed bootstrap of
log10 tail-loss ratios to AdamW gives schedule **-.057 [-.186,.084]**, reset
**+.024 [-.051,.099]**, clip effectively zero. Negative favors correction.
The schedule's modest point improvement is uncertain, not robust closure.

Reset reduced changing-curvature harmful fraction from .121 to .0208 and noisy
fraction from .223 to .209, yet noisy loss did not improve. The fixed step cap
also worsened the stationary diagonal/isotropic tails on one evaluation seed,
producing respective aggregate losses 8.83e-6/3.02e-7 instead of effectively
zero. The schedule slowed curvature-change recovery. These findings concern
the specified policies, not impossibility of all clipping/restart/schedule rules.

Full [trials](../results/sprint2/campaign-20261012/processed/correction_evaluation_trials.csv),
[summary](../results/sprint2/campaign-20261012/processed/correction_evaluation_summary.csv),
[paired comparisons](../results/sprint2/campaign-20261012/processed/correction_evaluation_paired_comparisons.json),
and [curves](../results/sprint2/campaign-20261012/figures/correction_evaluation_curves.png).
The stage's immutable source ZIP and each saved config document exact defaults.

## Measured additional cost

Nine warmed CPU blocks, rotated method order, identical precomputed gradients;
100 measured steps after 30 warmup steps. Include all observer/reset/copy work
and gradient assignment; exclude autograd, objective/Hessian evaluation, file
I/O, setup and tracing. Median of within-block ratios to ordinary AdamW:

| Policy | 16 parameters: ratio [IQR] | 16,384 parameters: ratio [IQR] |
| --- | ---: | ---: |
| Step clip | 2.00 [1.62,2.70] | 1.42 [1.27,1.47] |
| Momentum reset | 4.28 [3.18,4.76] | 1.93 [1.75,2.11] |
| LR safeguard | 2.02 [1.36,2.09] | 1.06 [.97,1.14] |

Ordinary AdamW median block times were 50.34 μs/step at 16 parameters and
282.93 μs/step at 16,384. The tiny case is noisy and dominated by Python/reduction
overheads. Ratios are paired medians, not ratios of marginal median times. These
prototype O(p) safeguards are not free; no end-to-end neural/GPU cost benefit is
established. The schedule has the lowest observed added cost on the larger case.
Raw [blocks and provenance](../results/sprint2/campaign-20261012/correction_timing).

**Decision:** first-order tuning resolves the original deterministic concern;
cheap fixed AdamW remedies do not robustly eliminate the stochastic residual.
Broader equal-budget first-order controls and a maintained structured reference
are needed before treating that residual as evidence for geometric correction.
