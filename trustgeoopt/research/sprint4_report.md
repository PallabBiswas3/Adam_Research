# TrustGeoOpt Sprint 4: noise-aware geometry identification

**Defer a new adaptive optimizer and pivot to a controlled measurement study.**
Known curvature probes recover the Hessian space when quadratic gradient history
is noise dominated. Estimated noise subtraction is unreliable, and the small
MLP shows too little incremental geometry recovery at the measured cost to
justify extending the campaign or building an optimizer.

Branch `research/sprint4-noise-curvature` starts at the verified pushed Sprint 3
commit `0dd4bcafa989a2b7abd15829489a51a44d6064b7`. No earlier experiments are
overwritten. Frozen [protocol](sprint4_protocol.md) and configuration
`configs/synthetic/sprint4.yaml`; campaign `results/sprint4/campaign-20261012/`.
The full optimizer remains unimplemented. All scientific conclusions below
refer to actually executed CPU evidence, not to prior papers' benchmark claims.

## Executed evidence and bounded baseline tuning

**966 successful attempts, zero failed/divergent/censored optimization attempts**:
six pilots, 640 tuning, 240 held-out baseline evaluations, 60 synthetic estimator
trajectories and 20 small MLP training runs. One thread, CPU, total experiment
execution **595.71 seconds (9.93 minutes)**, within 1,800 seconds / 1,000 trials.
Trial timers sum to 586.76 seconds; campaign time also counts recording and
selection. Literature, tests, postprocessing and file hashing are outside the
experiment timer and are not represented as training time. Successful
postprocessing took 52.51 seconds. A first postprocessing attempt failed on
numpy JSON scalar serialization; its source/partial outputs were archived and
the serialization was corrected without rerunning or changing raw experiments.

Each baseline received eight predefined joint candidates × ten development
seeds50–59 × rising/falling regimes. Rotation was balanced by development-seed
parity. Select one transferable policy per method using mean log tail loss plus
0.5run SD, requiring all20 development attempts per candidate to succeed.
Freeze before240 evaluations: ten fresh seeds60–69 × three noise regimes ×
diagonal/rotated × four methods. Constant noise is a held-out regime. No earlier
evaluation data select a policy. This is a comparable bounded joint search,
not an exhaustive independently optimized policy for every regime.

Selected policies: AdamW .03/beta2=.9/epsilon=.1/conservative decay; AMSGrad
.1/.99/.001/cosine; Momentum .0003/.9/cosine; pinned authors' SOAP .03/cosine.
Conservative decay ends at10% of initial rate; cosine at1%. Zero weight decay.
Momentum search extends to .003, ten times the old .0003upper boundary; all
selected rates are interior. Beta2/epsilon/search interactions remain bounded.
SOAP retains its disclosed preliminary reference implementation and tensor
factor structure, not a maintained full-matrix distributed Shampoo claim.

Development-only seed resampling selects the original candidate in68.25% of
AdamW resamples,99.725% AMSGrad,51.475% Momentum and99.95% SOAP. AdamW/Momentum
selection uncertainty remains material; evaluation was not used to change them.
The eight joint candidates do not support isolated causal beta2/epsilon effects.

Mean tail true loss at2,400 calls (last20%), ten fresh runs per cell:

| Noise / coordinates | AdamW | AMSGrad | Momentum SGD | SOAP |
| --- | ---: | ---: | ---: | ---: |
| Rising / diagonal |.12354 |.20414 |.10669 |.10807 |
| Rising / rotated |.12557 |.13907 |.10669 |.10803 |
| Constant / diagonal |.16750 |.19061 |.12789 |.13341 |
| Constant / rotated |.19346 |.15234 |.12789 |.12987 |
| Falling / diagonal |.0006252 |.0014094 |.0164783 |.0004700 |
| Falling / rotated |.0005876 |.0263965 |.0164783 |.0004253 |

AdamW's paired rotation log10-loss contrast is .0067 [-.0038,.0170] rising,
.0509 [.0060,.0943] constant, and -.0268 [-.0321,-.0218] falling. The former
order-of-magnitude noisy rotation gap is substantially attenuated under this
different policy/noise protocol; the small constant effect remains. AMSGrad
has a falling-noise rotation weakness, contrast1.118 [.520,1.714]. Momentum
matched trajectories agree to floating-point accuracy. This does not prove
all rotation effects disappear, and differs from Sprint 3 noise scale and tuning.

SOAP has lower equal-call tail loss than AdamW in all six cells, with paired
log contrasts ranging -.069to-.153. It is near Momentum on rising/constant
noise, while falling SOAP improves on the jointly selected Momentum policy.
AdamW also performs well after falling noise, without the previous two-seed
selection's large evaluation failures. There is no need to invoke a new
geometry mechanism to explain that recovery. Transient200-call recovery loss,
tail variance, final loss, per-run wall time, sustained100-call target≤.1and
censoring are all reported in `processed/baseline_runs.csv` / `baseline_summary.csv`.
State bytes: Momentum128, AdamW264, AMSGrad392, SOAP768 on4×4 parameters.

## Equivalent compute: important limits

The primary practical check truncates every recorded trajectory at a common
per-regime wall-time horizon of about .147–.156 seconds and averages its last
100 available losses. SOAP's equal-call advantage is not a general equal-time
advantage. On **constant noise**, SOAP/Momentum paired log-loss contrast is
+.7695 [.5286,1.1026] diagonal and+.8660 [.6271,1.1469] rotated: SOAP is worse
at that common horizon. SOAP/AdamW is+.3124 [.0457,.5632] and+.6145
[.3971,.9092]. Full-run mean times vary from fractions of a second to over a
second, with cache/OS scheduling outliers; comparisons are descriptive for one
machine and this recording overhead.

Noise changes are **step indexed**. At equal elapsed time different methods
can be on different sides of the transition. The
`processed/timing_phase_audit.json` explicitly identifies this confound; rising/
falling equal-time contrasts cannot establish a causal efficiency ranking.
Schedules also remain defined over2,400 calls, not a newly selected short
wall-time horizon. No additional timing campaign or wall-time-aware tuning is
claimed. These limitations prevent claiming that every established remedy was
exhausted or that a structured method is universally unnecessary.

## Noise identification, mathematical evidence and neural gate

See [estimator comparison](sprint4_estimator_comparison.md). In the antialigned
quadratic control, centered history captures .0065 of Hessian trace versus .8420
HVP/common-noise FD and .240random. HVP's paired incremental capture is
.8355 [.8350,.8360]. Estimated covariance subtraction reaches only .0521,
below random, and worsens aligned/partial cases. Paired noise PCA identifies
Sigma's space reasonably well, but its covariance error exceeds the weak late
signal. Isotropic-H/Sigma eigenspaces are nonidentifiable controls.

The [theory](sprint4_noise_curvature_theory.md) derives conditional covariance
identities, eigenspace ordering requirements, a correctly scoped Davis–Kahan
bound, the same-point paired-noise identity, covariance-only nonidentifiability,
an SPD recovery formula when excitation covariance is also known, and the
independent-FD variance explosion. Tests verify scaling, a counterexample,
the SPD formula and numerical perturbation-bound application. No theorem about
full adaptive Adam is inferred from fixed-B or linear covariance theory.

See [neural geometry](sprint4_neural_geometry.md). Two fresh correlated-feature
tasks × ten runs, per-layer activation/gradient/noise and small GGN-block
oracles at epochs0/5/20. Actual gradients use identical parameters and
independent with-replacement minibatch pairs. No network-wide Hessian is formed.
Final HVP-minus-centered-history capture: task800+.01069 [.00176,.01972],
task801+.00942 [-.00081,.02114]. Refreshing both layers every100 steps costs
.3926 of measured step time, exceeding10%; the gain also misses the .05gate.
Shared-batch FD and estimated subtraction perform worse here. Deep MLP,
larger Fashion and optimizer proof-of-concept are **not launched because the
predefined gate fails**, not because a missing resource is hidden.

## Provenance, review and research decision

Complete automated suite: **80 passed, 2 CUDA tests skipped**, with the existing
PyTorch TypedStorage deprecation warning. New tests cover independent-pair
scaling, common-noise secants/HVPs, randomized reconstruction/counts, refresh
state, split-safe selection, layer-block derivatives and mathematical controls.

Source/configuration snapshots, immutable per-attempt configs, sampled
gradients/parameters, pair observations, true noise/H/GGN controls and
trajectories are preserved. Original experiment source hash is
`31e2bdc6b0bbc6d9c8df6a71e9b1d5354487b32a6ddc6a4964d195b11caeed50`.
Later source adds postprocessing/audits and hardens refresh monotonicity/CUDA
timing and configuration validation; it does not replace the captured run code.
The final source snapshot is recorded separately. All choices recompute from
development only, successful horizons/config hashes are audited, and **7,258
earlier Sprint 2/3 artifacts are byte-identical**. New raw artifacts and source ZIPs
remain ignored locally; Git includes run indexes, compact estimator rows,
statistics, five plots, protocol, source provenance and manifests.

All intervals use4,000whole-seed paired resamples, conditional on frozen
development selection, without multiplicity correction. Two neural tasks,
one small architecture, one refresh/rank choice, exact quadratic curvature,
layer-block oracles and short shared-process CPU timings limit generalization.
The [literature audit](sprint4_literature_audit.md) checks primary mechanisms
through2026-10-12and identifies substantial overlap. The standalone estimator
implements established randomized range/Ritz; no new update is invented.

The [Sprint 5decision](sprint5_decision.md) answers all seven research questions.
Recommended direction: a reproducible noise/curvature identifiability and
cost-audit study, with fresh wall-time-aware baseline selection before any
algorithm work. A diagnostic improvement on exact quadratics is insufficient
to justify a new neural optimizer.
