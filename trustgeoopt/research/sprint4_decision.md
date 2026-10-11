# Sprint 4 decision

**Proceed only with mechanism validation; defer TrustGeoOpt implementation.**
The sprint succeeds as a diagnostic/negative study. There is evidence of basis
sensitivity in noisy quadratics, but no demonstrated residual practical
weakness that cheap first-order alternatives cannot handle and no distinct new
optimizer contribution.

## Answers grounded in Sprint 3

1. **Which AdamW weakness survives?** Matched condition-1,000 noisy rotation
   worsens loss under independently tuned constant rates. The clean mean gap
   is outlier driven and does not establish a robust rotation effect. The old
   rising-noise ranking reverses under broader tuning. Falling-noise/decay
   fragility remains heterogeneous, with wide intervals and a two-seed selection
   failure; its precise cause is unresolved.
2. **Which information is robustly useful?** Hessian geometry explains a limited
   fixed-diagonal scaling disadvantage. Gradient-history directions help in
   aligned-noise cases but fail below random controls with antialigned noise.
   Frozen effective condition omits momentum/rate dynamics and does not predict
   optimizer rankings alone. Sprint 2's small alignment lift does not become a
   validated new diagnostic here; no stronger harmful-update prediction claim
   is made without a new independent diagnostic evaluation.
3. **Can existing remedies handle the problems?** Broader AdamW schedules remove
   the original noisy ranking; Momentum/AMSGrad solve clean geometry outliers;
   Momentum competes with SOAP on rotated noise and improves deep-MLP quality.
   First-order tuning boundaries and selection uncertainty remain. We have not
   exhausted these remedies, so cannot claim they are insufficient.
4. **What would justify low-rank geometry?** A sustained-quality, equal-total-time
   deficit across fresh tasks after broader first-order safeguards, plus a
   recoverable curvature subspace that survives noise-direction controls.
   Curvature information beyond a covariance sketch, or a justified estimator
   distinguishing noise from curvature, must be demonstrated at measured cost.
   Neither gate is currently satisfied.
5. **Is the direction distinct?** Unestablished. Structured adaptive coordinates,
   gradient sketches/state transport, rotationally equivariant scaling and
   basis-dependence diagnostics already have close primary prior art. The
   [targeted review](sprint3_literature_audit.md) verifies this overlap; generic
   trust gating plus low-rank history is not a contribution claim.

## Proposed bounded Sprint 4, not launched

- Reserve development seeds 50–59 and evaluation seeds 60–69; never tune using
  Sprint 2/3 evaluations. Start with a smaller number of regimes and more
  development seeds rather than enlarging every hyperparameter axis.
- Resolve falling-noise selection fragility first. Compare phase-independent
  constant/cosine/conservative schedules, beta2 and epsilon in an equal predefined
  budget. Inspect total remaining step magnitude, low-curvature error and EMA
  response. Choose hyperparameters by a robust multi-run development criterion;
  report outer selection uncertainty if affordable. Retain failures and boundaries.
- For geometry, add schedule/beta tuning to diagonal baselines, random rotations
  and several spectra/condition numbers; compare fixed-rate matched pairs as
  well as independently tuned policies. Include known inexpensive scalar/vector
  scaling controls before any new mechanism. Broaden to an isolated compatible
  maintained Shampoo environment only if the residual is still meaningful.
- Test noise-isolating information using exact toy controls, paired gradient
  differences or Hessian-vector probes. These are measurement experiments,
  not a new optimizer. Demonstrate useful low-rank capture when noise and
  curvature disagree, with a random/diagonal control and explicit probe cost.
- Expand flagged neural learning-rate boundaries and tune stopping/schedules
  exclusively on development. Use sustained validation targets, restricted
  mean time with censored runs, paired quality/time comparisons and isolated
  memory measurements. Repeat at least two data/task seeds and a larger Fashion
  training subset within a newly specified CPU budget. No automatic GPU campaign.

Stop algorithm development if cheap baselines keep matching quality/time or
history remains dominated by noise. A careful reproducible analysis can remain
the project's outcome. Consider a prototype only after both a residual practical
gap and a distinct, demonstrably useful extra-information mechanism are present.
