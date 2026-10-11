# TrustGeoOpt Sprint 3: mechanism discovery

**Defer a new TrustGeoOpt optimizer.** This sprint identifies a conditional
geometry effect, rejects a blanket noisy AdamW deficit, and finds that existing
first-order alternatives already handle important weaknesses. It does not
establish a distinct residual mechanism requiring low-rank geometric correction.

Branch `research/sprint3-mechanism-validation` starts at the remotely verified
pushed Sprint 2 commit `75875517abe2446ed11c29fcbc6fc236f1c466e1`.
All earlier results remain unchanged. No GPU campaign or new optimizer was run.

## Executed evidence

- **Reproduction:** all 15 previously frozen noisy-quadratic runs match Sprint 2
  tail losses exactly; the maximum per-run difference is zero. Old evaluation
  seeds are excluded from selection.
- **Stochastic tuning:** 477 attempts with rates, beta2/momentum/refresh and
  constant/cosine schedules. AdamW beats SGD/Shampoo on fresh rising-noise
  evaluations (.0163 versus .1128/.0492 mean tail loss). Constant-noise and
  falling-noise contrasts are uncertain; falling AdamW errors concentrate in
  two seeds after an overly optimistic two-seed development choice.
- **Geometry:** 556 attempts with truly matched spectra, initialization and
  noise. Noisy AdamW rotated/diagonal paired log10-loss contrast .9238
  [.3485, 1.5925] survives the bounded constant-rate protocol. Momentum gives
  identical matched trajectories and competitive quality. AMSGrad/SGD resolve
  the clean AdamW outliers; broad geometry schedule tuning remains incomplete.
- **History:** under held-out antialigned noise, rank-4 history captures only
  .0439 of curvature trace on rotated objectives, versus .2707 for random and
  .8420 for oracle directions. History is a noise-dependent signal, not a
  Hessian estimate. Centering does not fix this control.
- **Neural validation:** 147 attempts on small/deep correlated-feature ReLU
  MLPs and a Fashion-MNIST subset. SOAP has no demonstrated quality/time
  advantage; cheap SGD helps the deep model while AdamW leads on the small
  model. Five fresh evaluation seeds and a held-out synthetic teacher are used.
- **Mathematics:** exact discrete fixed-B covariance and scalar stationary loss,
  exact pure-noise beta2 moments, and a limited diagonal-scaling inefficiency
  proposition. Actual Adam conclusions remain explicitly conditional.

These are whole-run comparisons, not independent-step tests. All percentile
intervals condition on a two-seed tuning stage and lack multiplicity correction;
five evaluation seeds and remaining grid boundaries limit general conclusions.
SOAP/SGD rotated-noise loss intervals overlap, while SOAP costs more on these
tiny CPU tasks. No benchmark-win, novel-theorem or publication claim is made.

## Budget, preservation and verification

Frozen cap: 1,800 experiment seconds / 1,200 attempts. Actual: **649.58 seconds**
(10.83 minutes), **1,195 attempts**, **1,181 successful / 14 failed**, no censored
or unstarted trials. Twelve failed trials are aggressive geometry SGD rates;
two are deep-MLP SGD tuning. Partial trajectories/configurations/errors survive.
Trial timers sum to 219.66 s stochastic, 338.98 s geometry and 81.25 s neural;
campaign time also includes replay, selection and recording overhead.

The separate fixed-B numerical check uses six configurations, 32 independent
trajectories each, 12,000 steps/2,000 burn-in and .61 s execution. Expected loss
lies inside all six run-resampling intervals; maximum relative empirical error
is 1.93%, and Lyapunov residuals are below 7e-16. Postprocessing, file-hash
audits, literature review, tests, imports and download lie outside the experiment
cap. Analysis provenance reports its pre-manifest time separately.

Automated suite: **69 passed, 1 CUDA test skipped**; an existing PyTorch storage
deprecation warning is retained. Independent audit verifies all attempt hashes,
complete successful horizons, tuning-only frozen selections, original source
snapshot, and **2,619 unchanged Sprint 2 artifacts**. Git also shows no modified
tracked Sprint 2 results. Vendor source/license hashes are checked in tests.

Raw data (about 276 MB), partial failures and source snapshots remain on disk
under `results/sprint3/campaign-20261012/`. They are ignored according to the
existing artifact policy. Git includes compact trial indexes, selections,
summaries, intervals, plots and a SHA256 artifact manifest; sharing the complete
raw evidence requires archiving the ignored artifacts as well. Postprocessing
has its own source provenance; optimizer-run code remains the captured version.

## Deliverables and decision

Read [stochastic analysis](sprint3_stochastic_analysis.md),
[geometry analysis](sprint3_geometry_analysis.md),
[neural validation](sprint3_neural_validation.md),
[mathematics](sprint3_mathematics.md),
[literature audit](sprint3_literature_audit.md), and
[Sprint 4 decision](sprint4_decision.md).
Configuration is `configs/synthetic/sprint3.yaml`; README supplies commands.
The novelty database and assessment are updated with verified primary sources.

The next defensible step is stronger first-order selection/generalization and
noise-versus-curvature identification, not implementing the proposed optimizer.
