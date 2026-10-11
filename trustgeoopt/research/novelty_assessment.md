# Novelty assessment and decision gates

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
