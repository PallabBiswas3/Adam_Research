# Proposed Sprint 3: determine whether a residual mechanism exists

**Proceed with validation; defer geometric optimizer implementation.** The original
finite-step AdamW failure disappeared under tuning. Alignment largely reflects
known descent dynamics, history did not transfer, and the remaining noisy loss
gap has a first-order alternative explanation. A full TrustGeoOpt study would
currently lack both a defensible mechanism and a verified distinct contribution.

## 1. Resolve tuning and stochastic-floor alternatives first

- Extend only flagged rate boundaries (noisy Momentum below .0001; rotated
  Shampoo above 3) and check epsilon/betas/schedules in equal predefined trial
  budgets. Compare constant, conservative and phase-independent decay schedules;
  do not tune thresholds using the five Sprint 2 evaluation seeds again.
- Reserve new development seeds (e.g. 30–33), new evaluation seeds (40–49), and
  unseen conditions/noise configurations. Freeze choices before evaluation.
- Separate curvature, optimum translation and noise shifts factorially. Use
  sustained quality, optimum distance and tracking recovery, not first target
  crossing or harmful-update count alone. Extend step horizons only after a
  cheaper run indicates slow convergence rather than a noise floor.
- Investigate the one-seed stationary degradation from the fixed .1 step cap.
  Any cap/schedule search consumes its full tuning allowance; do not silently
  compare a policy tuned on many knobs against AdamW with only an LR search.

Suggested initial budget: one predefined CPU campaign capped at 30 minutes and
1,000 attempts, reduced grids before wider exploration. This is a proposal,
not a launched campaign or an approval already granted for additional compute.

## 2. Test diagnostic robustness without buying geometry

Compare development-fitted flexible first-order controls including interactions
between direction, update magnitude, preconditioner scale and momentum state.
Keep synthetic oracle curvature/distance separate from deployable features.
Measure AP, calibration, decision utility and false-alarm costs per regime/rate,
especially at independently selected successful rates. Report undefined AP in
positive-free strata. Use whole-run/seed resampling and bootstrap training/selection
in an outer loop when affordable; current CIs condition on a two-seed fit.

Require a consistent positive incremental effect across new task families, and
a policy benefit at equal quality/cost. Small pooled AP lift alone is not a gate
criterion. Check causes of undefined near-convergence angles with alternative
predefined cohorts, retaining zero-direction counts rather than filtering silently.

## 3. Ask whether off-diagonal information is actually necessary

Only if a weakness survives step 1, compare a maintained Shampoo/SOAP reference
under a bounded small CPU task. Use rotation-matched spectra, noise-free and
matched-noise controls; report extra state, factor updates and total time. An
oracle full-Hessian quadratic direction may be a mechanistic upper bound, but
must never be presented as an affordable neural-network algorithm.

For a possible low-rank hypothesis, demonstrate a recoverable leading-curvature
subspace and a residual rotation-dependent **equal-quality compute** deficit
that tuned first-order alternatives cannot close. Gradient covariance must not
be called a Hessian without an argument. Include random-direction, stale/frozen
basis, diagonal-only and fixed-refresh controls before any gated low-rank design.

## 4. Literature and implementation decision

Audit full algorithms in SOAP, Purifying Shampoo, FOAM, Sketchy, LDAdam and SENG;
distinguish basis refresh, curvature approximation, covariance and moment
rectification. Produce a mechanism table before proposing a new update. If no
distinct residual mechanism remains, pivot to a reproducible negative result
about diagnostic confounding/tuning rather than branding a new optimizer.

Only after those conditions pass, write an explicit design and cost model for
a small geometric correction; then test one small supervised MLP/CNN with fixed
data splits. No extensive GPU or large-model campaign is justified at this point.
