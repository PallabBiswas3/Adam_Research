# Sprint 5 decision

**Pivot to measurement and baseline-selection research; defer a new optimizer.**
Sprint 4identifies what different observations recover, but its predefined
neural accuracy/cost gate fails. The standalone code is an established spectral
estimator, and no equal-total-compute optimizer gain or distinct algorithm
contribution is demonstrated.

## Seven required answers

1. **Can noise-dominated directions be separated from useful curvature?**
   Conditionally yes: independent same-point gradient pairs identify noise
   covariance, and direct HVP/common-random-number secants identify curvature
   actions under their explicit assumptions. They identify different objects.
   Estimated covariance subtraction does not reliably recover Hessian geometry,
   and exact subtraction still requires adequate, appropriately ordered excitation.
2. **Does an estimator outperform ordinary history on held-out conditions?**
   Yes for known HVP/Ritz/common-noise FD on independent random, partial and
   antialigned quadratic controls. Antialigned incremental capture over centered
   history=.8355 [.8350,.8360]. This is classical curvature information, not a
   new estimator. Paired-noise subtraction fails important controls.
3. **Does the advantage remain in real MLP training?**
   Only a small conditional improvement: final HVP gain+.0107and+.0094on two
   tasks; one interval crosses zero. Both miss the predefined .05 gain. Noise and
   gradient covariances already capture useful GGN-block directions on these tasks.
   The antialigned quadratic failure is not reproduced as an imposed neural
   counterfactual. Shared FD/subtraction are worse than centered history here.
4. **Is the added cost justified?**
   Not by this study. Both-layer HVP refresh every100 steps costs about39% of
   measured step time, above10%. No probe-guided optimizer was run, so no
   practical speedup follows from subspace accuracy. Much sparser refresh could
   reduce cost, but staleness and utility would need independent evaluation.
5. **Do established estimators already provide equivalent information?**
   Yes at the mechanism level: stochastic HVPs, periodic quasi-Newton curvature
   sampling, PSD GGN/Fisher products and randomized spectral approximation
   provide the tested curvature action. Paired noise moments and common-sample
   control variates are established identities/mechanisms. K-FAC/EKFAC/PSGD/
   Shampoo/SOAP occupy nearby structured approximation/cost tradeoffs. We did
   not benchmark every complete optimizer or prove their equivalence as updates.
6. **Does the evidence justify a new adaptive optimizer?**
   No. Broader first-order tuning attenuates the earlier rotation gap and gives
   stable falling-noise AdamW performance. SOAP's equal-call advantage costs
   more and does not establish a general equal-time advantage. Some weaknesses
   remain (AMSGrad falling rotation; selection uncertainty; constant-noise small
   AdamW basis effect), but their need for new information is unproven.
7. **Is there a defensible potential contribution or should we pivot?**
   Pivot. A useful candidate contribution is a carefully scoped reproducible
   audit of covariance/curvature identifiability, estimator failure controls,
   selection uncertainty and equal-time phase confounding. It is not currently
   a priority, new-theorem, new-estimator or publication claim. Prior-art overlap
   is substantial, and the scoping search does not certify exhaustive novelty.

## Proposed Sprint 5, not launched

- Freeze a measurement question rather than inventing an update: when does a
  noisy/history subspace retain curvature-relevant information at a given cost?
  Include excitation anisotropy, independent noise sampling, local model error,
  rank/eigengaps, stale geometry, common-sample secants and established probes.
- Reserve new development70–79and evaluation80–89; do not reuse Sprint 4evaluation
  for choosing rank, δ, refresh or stopping. Use more independent task/data seeds
  before making neural-population claims.
- Resolve fair timing: repeated warm, isolated CPU timing; noise transitions
  indexed by a shared external/sample-time definition; schedules chosen for the
  same total compute horizon; sustained quality/censoring rather than crossings.
  Current rising/falling retrospective equal-time comparisons mix noise phases.
- Compare independently tuned per-regime policies as a secondary question to
  the transferable joint policy. Separate beta2/epsilon/schedule effects with
  matched factorial controls and outer development selection resampling.
- If warranted, compare published PSGD/SQN and GGN/K-FAC/EKFAC implementations
  in compatible isolated environments without silently upgrading this workspace.
  Better HVP batching/graph reuse and rank/refresh tradeoffs are engineering
  baselines, not automatically an algorithm contribution.
- Study small neural curvature/noise counterfactuals with justified sampling and
  activation-cell controls. Label noise or changed batches do not reproduce an
  arbitrary additive covariance independently of H. Include failed mechanisms.
- Require both a residual equal-quality compute deficit against tuned existing
  methods and a useful extra-information mechanism at affordable measured cost
  before a separate tiny optimizer proposal. Continue to defer full TrustGeoOpt.

The bounded sprint succeeds with this negative decision. No automatic deep/
Fashion/GPU/cloud campaign is authorized or launched by this plan.
