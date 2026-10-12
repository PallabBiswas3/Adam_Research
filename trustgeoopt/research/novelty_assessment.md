# Novelty assessment and decision gates

## Current decision after Sprint 4

**Pivot to controlled measurement research; defer a new optimizer.** Established
HVP/common-random-number secants recover quadratic curvature when gradient
history follows antialigned noise. Same-point paired gradients estimate noise;
estimated covariance subtraction fails important controls and cannot identify
H without excitation assumptions. These are established mechanisms, not a new
noise-aware estimator or theorem about full Adam.

Two held-out small MLP tasks show final HVP capture gains of only .0107/.0094
over centered history (one interval includes zero), with about 39% measured
both-layer refresh overhead at one refresh per 100 steps. The predefined .05
accuracy / .10 cost gate fails. Broader first-order tuning attenuates earlier
rotation gaps; SOAP's equal-call improvements do not establish a universal
equal-time benefit. Step-indexed noise transitions and short CPU timing are
explicit confounds, so existing remedies are not declared exhausted.

See [Sprint 4 report](sprint4_report.md),
[primary mechanism audit](sprint4_literature_audit.md), and
[Sprint 5 decision](sprint5_decision.md). HVP/GGN/Fisher, K-FAC/EKFAC, stochastic
quasi-Newton/PSGD, randomized spectra, noise-scale measurements, covariance
sketches, common-sample variance reduction, refresh/damping and 2026 bias
corrections substantially overlap the proposed components. No priority or
publication claim is justified by this scoping review.

| Gate | Sprint 4 evidence | Decision |
| --- | --- | --- |
| Distinct mechanism | tested information already available from established probes | unpassed |
| Noise/curvature identification | conditional identities; subtraction/excitation counterexamples | useful measurement result |
| Held-out toy recovery | direct probes beat history in antialigned/partial/random cases | passed for known probes |
| Neural incremental value | ~1 percentage point final capture gain, uncertain on one task | accuracy gate failed |
| Affordable cost | ~39% amortized HVP refresh versus 10% gate | cost gate failed |
| Residual practical need | no new equal-compute gain; tuning/timing limits remain | unestablished |
| Algorithm development | neither distinctness nor neural utility/cost gate satisfied | defer; pivot |

The earlier Sprint 3 assessment below is preserved as historical context.

## Current decision after Sprint 3

**Continue mechanism validation; defer a new optimizer.** The original noisy
ranking reproduces but reverses after broader rate/beta/schedule tuning in the
rising-noise regime. Matched noisy rotation still affects constant-rate AdamW,
yet Momentum competes with structured SOAP, and Momentum/AMSGrad handle clean
outliers. Small/deep MLP and Fashion-MNIST subset experiments establish no SOAP
quality/time advantage. History directions fail the antialigned-noise control.
See [Sprint 3 report](sprint3_report.md) and [Sprint 4 decision](sprint4_decision.md).

The [updated primary-source audit](sprint3_literature_audit.md) adds stationary
noise analysis, vector-level equivariance and parameter-basis studies, and
extracts the assumptions of existing sketch/state-transport methods. These
mechanisms already occupy the proposed components. The limited diagonal-scaling
proposition is classical preconditioning reasoning; it is not a novel Adam
theorem. No optimizer or publication novelty is claimed.

| Gate | Sprint 3 evidence | Decision |
| --- | --- | --- |
| A distinctness | strong verified overlap; targeted review not exhaustive | unpassed |
| B failure mechanism | conditional noisy rotation; seed-sensitive noise-drop errors | retain hypotheses, not a universal Adam deficit |
| C useful additional information | history fails when noise/curvature disagree | unpassed |
| D existing remedies insufficient | SGD/AMSGrad/schedules already address key cases | unpassed |
| E equal-quality efficiency | no structured advantage over cheap methods on tested tasks | unpassed |
| F neural generalization | bounded multi-seed models executed; geometric cause unisolated | partial validation only |
| G algorithm/publication | residual practical need and distinct mechanism absent | defer |

An evidence-oriented study remains a legitimate outcome. A new update rule
requires both a practical residual gap and measured, causally useful information
that known inexpensive mechanisms lack. The earlier sprint assessments below
are retained as historical context.

## Current decision after Sprint 2

**Do not proceed to a full geometric optimizer.** Independent tuning and 1,200
steps remove the original stationary AdamW deficit. The published projected
online Adam counterexample reproduces under its separate assumptions. A noisy
late-loss gap remains, but tuned Momentum SGD improves it and the tested fixed
cheap AdamW policies do not robustly close it. This does not identify geometry
as the missing ingredient. See [Sprint 2 report](sprint2_report.md),
[baseline validation](sprint2_baseline_validation.md),
[reliability analysis](sprint2_reliability_analysis.md), and
[Sprint 3 plan](sprint3_plan.md).

Alignment has a small pooled conditional lift at leads 0/1, but loses lift on
some held-out regimes and at longer leads. History fails unseen incremental
validation. Exact quadratic descent geometry explains all same-step harmful
labels with oracle information. Improvements over gradient norm alone do not
establish a new reliability mechanism or a need to purchase geometric computation.

Verified targeted prior art remains close: [AdaBelief](https://arxiv.org/html/2010.07468v5)
already uses prediction residuals; [adaptive restart](https://arxiv.org/pdf/1204.3982)
uses directional conditions; [clipping](https://arxiv.org/abs/1905.11881) controls
magnitude. [SOAP](https://arxiv.org/html/2409.11321v2) combines adaptive updates
with structured coordinates; [Purifying Shampoo](https://arxiv.org/html/2506.03595v2)
controls basis approximation error; [FOAM](https://arxiv.org/abs/2606.02365)
addresses staleness through adaptive damping (preprint). No novelty is claimed
for the Sprint 2 AdamW adaptations or a generic selective-refresh proposal.

| Gate | Sprint 2 state | Remaining evidence |
| --- | --- | --- |
| A literature/novelty | targeted audit updated; distinctness unestablished | mechanism-level review of maintained geometry/low-rank implementations |
| B reproducible failure | known online counterexample reproduced; original finite-step failure rejected | persistent quality/compute gap after broader first-order tuning |
| C incremental diagnostic | partial short-lead pooled alignment lift; general gate unpassed | stronger predictors, more seeds and positive cross-regime incremental benefit |
| D correction | fixed cheap remedies evaluated; no robust noisy closure | broader equal-budget schedules/momentum/epsilon controls and maintained structured comparator |
| E efficiency | warmed CPU correction cost measured; no end-to-end benefit demonstrated | equal-quality total cost on an actual task |
| F neural generalization | untested | small supervised models after synthetic mechanism survives |
| G publication | unpassed | defensible residual mechanism, distinct contribution and generalization |

Possible residual hypothesis: off-diagonal curvature may reduce equal-quality
time on sufficiently ill-conditioned rotated tasks after comparable first-order
controls. This sprint does not establish that hypothesis. The noisy Shampoo
advantage alone is insufficient: cumulative statistics, normalization, tensor
structure and rate dynamics differ. A useful next contribution could be a
careful negative/diagnostic study rather than another optimizer.

## Initial assessment (retained historical context)

Initial conclusion: no verified novel contribution. Updating adaptive moments,
reliability-like residual scaling, low-rank gradient sketches, structured coordinate
transformations, and adaptive preconditioner refresh all have prior art.

Purifying Shampoo (2025) explicitly develops adaptive eigenbasis refresh based on
approximation error. FOAM (2026 preprint) addresses stale Shampoo preconditioners
with adaptive damping. Thus adaptive geometric computation is already explored;
H3 alone is not a novelty claim. Inspect LDAdam, Sketchy, SENG, CASPR, ASGO and
unified structured-preconditioner theory before any new update is proposed.

Narrow candidate: causal score evaluation separating true mean/curvature shifts
from noise shifts, followed by an equal-total-compute supervised MLP/CNN study.
This is a study design, not a demonstrated research gap. A reproducible negative
result may be more useful than another overlapping optimizer.

| Gate | Current state | Evidence needed |
| --- | --- | --- |
| A literature/novelty | incomplete, strong overlap | full-text/2026 audit and mechanism comparison |
| B failure reproduction | exploratory pilots only | tuned baselines, five seeds, mechanistic controls |
| C useful diagnostics | unpassed | held-out AP lift over norm with cost accounting |
| D correction | unpassed | relevant structured remedy beaten at matched tuning |
| E efficiency | unpassed | total synchronized time, uncertainty, equal quality |
| F generalization | unpassed | multiple conventional architectures and datasets |
| G publication | unpassed | all above plus proofs/ablations/reproducibility |

Full method variants are intentionally pending. This follows the user's explicit
progression gates and initial-task definition, not an implementation blocker.
