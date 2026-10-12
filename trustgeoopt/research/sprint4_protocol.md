# Sprint 4 frozen protocol

Written before pilot or evaluation on 2026-10-12. Branch
`research/sprint4-noise-curvature`, parent `0dd4bcafa989a2b7abd15829489a51a44d6064b7`.
Exploratory mechanism study, not a publication benchmark. CPU, one thread,
1,800 seconds total experiment execution including pilots, 1,000 optimization
attempts maximum. Imports, tests, review and postprocessing are separately timed.
No GPU, cloud, dependency upgrade, new optimizer or evaluation-based retuning.

## Literature protocol

Scoping review of primary English papers available through 2026-10-12 (not
future October publications). Search arXiv, official proceedings and authors'
papers for Hessian-free/GGN/Fisher, K-FAC/EKFAC, Shampoo/SOAP/Sketchy,
stochastic quasi-Newton/PSGD, gradient-noise estimation, paired differences,
online PCA, randomized low-rank operators, control variates, noise-aware
preconditioning. Record exact queries, verified URLs, extraction locations,
assumptions and cost. Exclude name collisions and unverified snippets.
No claim of exhaustive coverage. Known estimators retain their established names.

## Baseline selection and evaluation

AdamW, AMSGrad, Momentum SGD, pinned authors' SOAP each receive eight predefined
joint configurations, ten development seeds 50–59, two regimes rising/falling:
640 tuning attempts. All ten seeds must succeed; choose one policy per method
by mean log10 tail loss over both regimes plus 0.5 times its run SD. Rotation
alternates by development-seed parity, balanced five/five; all methods share
the assignment. This deliberately selects a transferable policy rather than
an optimum for each evaluation condition. Joint candidates cover rates,
constant/cosine/conservative schedules, Adam beta2/epsilon, SGD boundaries and
SOAP rates. Equal candidate/seed/call budgets, not exhaustive factorial tuning.
Candidate main effects and interactions cannot be isolated from this design.

Freeze policies before 240 evaluation attempts: four methods × three noise
regimes × diagonal/rotated × seeds 60–69. Constant noise is a held-out regime.
16 dimensions, condition 1,000, 2,400 calls, transition halfway, matched
initialization/noise streams and equal spectra; isotropic gradient noise.
High/low noise SD 10/.1. Zero weight decay. SOAP first initialization call
counts. No earlier evaluation seeds enter selection. Raw trajectories retained.
Metrics: tail mean (last 20%), final loss, tail SD, post-transition 200-call
loss, sustained target (100 consecutive losses ≤ .1), total time and tensor
state memory. At a common per-regime wall-time horizon (minimum completed
loop time across methods/seeds), compare trailing 100-call loss using available
time-stamped observations. These are retrospective trajectory truncations,
not separately selected/restarted runs; timer precision and I/O disclosed.

## Synthetic estimator study

Six regimes × ten evaluation seeds = 60 fixed-policy trajectories, 1,200 calls:
aligned, orthogonal/antialigned, partial (45-degree mixing of top/bottom noise
directions), independent random noise, orientation switch at 600, isotropic H
control. None used for optimizer selection. Covariance trace 16, condition 100,
noise SD 10, Hessian condition 1,000. Frozen diagnostic AdamW .03/.99/1e-8,
conservative decay. Rank 4, window 64, snapshots 80/600/1,200.

Compare uncentered/centered history, same-parameter paired noise differences
(64 pairs, two independent calls each), covariance subtraction (indefinite
matrix retained), oracle subtraction H Cov(theta) H, random and coordinate
controls, randomized HVP range/Ritz (rank+4 probes, one power refinement),
same-noise central finite differences, independent-noise differences, oracle H.
Same-noise cancellation is exact only in this additive-noise toy; neural
same-minibatch curvature remains sampled curvature. Paired differences identify
noise, not H. HVPs are evaluated through autograd on toy loss; operation counts
and time include graph construction. δ=.01. No estimator steers training.

Primary estimator endpoints: top-H subspace overlap and trace capture relative
to random/history; noise-subspace overlap and relative covariance error for
paired noise; oracle-calibrated reconstruction error (explicitly evaluation
only), actual HVP reconstruction, refresh time, operator/gradient calls and
retained tensor bytes. Isotropic-H overlap is nonidentifiable and excluded
from inferential interpretation. Counterexamples include missing excitation,
sample subtraction error, independent small finite differences and stale
orientation. No iid covariance concentration claim for trajectory history.

## Neural geometry and gates

Ten seeds 60–69 on each of two fresh teacher/tasks 800/801: 20 small ReLU MLP
training attempts, 12 features, width 16, 512 train/256 validation, batch32,
20 epochs, fixed AdamW .003 cosine (measurement trajectory, not optimizer
benchmark). Train-only normalization. Snapshots epochs 0/5/20, per-layer input
activation moments/covariances, observed gradient moments, paired independent
with-replacement minibatch noise draws at identical parameters, sampled
GGN/Hessian-block probes, same/independent minibatch finite differences.
Full TRAIN layer GGN blocks serve only as small diagnostic oracles; no dense
network-wide Hessian is allocated. Binary softmax layer block GGN agrees with
layer Hessian inside fixed ReLU cells; no full-network equality is claimed.
At epoch 0 use frozen minibatch samples as history (no previous training history).
Compare blocks against the same oracle and same rank. Record drift over epochs,
error, independent probe RNGs, batch calls, refresh/step timing and bytes.

Run a short development-seed pilot first. Deep/Fashion expansion requires a
robust held-out neural capture gain ≥ .05 over centered history with lower
paired 95% bound >0 on both tasks AND refresh amortized every100 steps ≤10%
of step cost, AND a residual equal-cost optimizer need. If any gate fails,
record defer, do not spend on larger experiments or optimizer proof-of-concept.
Standalone range/Ritz is an implementation of classical randomized spectral
estimation, not a proposed new estimator. Rank/refresh/stabilization tested.

## Statistics, provenance and preservation

Whole-seed paired percentile bootstrap 4,000 resamples; all phases/layers from
one run stay together. Ten seed pairs per primary condition. Contrasts are
exploratory, conditional on frozen development selection; no multiplicity or
task-population claim. Missing/failed runs reported, never silently dropped.
Independent development bootstraps assess policy-selection frequency without
changing the selected policy. Frozen config, source snapshot/hash, reference
pin/license, per-attempt configs/status and compact raw metrics accompany all
results. New campaign directory cannot overwrite an existing one. Hash earlier
Sprint 2/3 evidence before/after. Commit compact evidence and plots; ignored raw
artifacts/source ZIPs remain preserved locally. Push research branch, no merge.
