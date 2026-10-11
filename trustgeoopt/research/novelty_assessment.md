# Novelty assessment and decision gates

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
