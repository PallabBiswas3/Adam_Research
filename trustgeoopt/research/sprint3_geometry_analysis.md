# Sprint 3 geometry analysis

**A matched noisy rotation effect survives this learning-rate protocol, but
existing first-order baselines already address it.** Gradient history captures
useful curvature in some phases and can track entirely the wrong directions
under antialigned noise. This does not support an automatic covariance-based
geometric correction.

See the [protocol](sprint3_protocol.md), `processed/geometry_selection.json`,
`geometry_snapshots.csv`, `paired_comparisons.json`, `mechanism_statistics.json`,
and `plots/geometry_loss.png` / `history_controls.png` in
`results/sprint3/campaign-20261012/`.

## Matched construction and reference limits

Condition-1,000, dimension-16 diagonal/rotated pairs share all Hessian eigenvalues.
H_rot=Q H_diag Q^T, x_rot=Q x_diag, xi_rot=Q xi_diag. Independent learning-rate
selection uses the same two development seeds, six candidates per method and
2,400 calls. Noise-free, isotropic-noise and independently oriented anisotropic
noise pairs are tuned; condition-1 isotropic and held-out aligned/antialigned
noise are controls. Anisotropic noise condition is 100 and trace is matched.
Noise directions relative to H are controlled separately from its rotation.

AdamW, AMSGrad, Momentum SGD and pinned authors' SOAP use their default moments
and constant rates. SOAP uses a 4x4 tensor-factor basis, not a full 16x16 basis;
arbitrary Q need not preserve its tensor structure. Its float32 factor algebra
and first-call initialization remain unchanged. Current Meta Shampoo cannot
run in this environment without a Python/PyTorch upgrade; the preliminary
SOAP reference is not described as production maintained.

336 tuning attempts include 12 divergent Momentum trials at lr=.01. Their
partial trajectories/configs/errors are retained. All 140 evaluation and 80
held-out attempts succeed. Rates/schedules were not chosen after inspecting
evaluation data. Constant-rate geometry and broader stochastic tuning are
different stages; geometry has **not** exhausted beta/schedule safeguards.

## Loss and rotation

Five-seed arithmetic mean tail losses:

| Objective | AdamW | Momentum SGD | AMSGrad | SOAP |
| --- | ---: | ---: | ---: | ---: |
| Clean diagonal | 1.048e-4 | 3.84e-87 | 1.11e-85 | 4.69e-6 |
| Clean rotated | .006631 | 3.84e-87 | 2.13e-25 | 1.24e-5 |
| Isotropic noise, diagonal | .007350 | .013174 | .008067 | .012375 |
| Isotropic noise, rotated | .186595 | .013174 | .033232 | .012074 |
| Misaligned noise, diagonal | .005604 | .013259 | .006476 | .009407 |
| Misaligned noise, rotated | .182541 | .013259 | .028653 | .009615 |
| Isotropic H/noise | .011637 | .039617 | .011909 | .011439 |

Paired log10(rotated/diagonal) for AdamW is .9238 [.3485, 1.5925] with isotropic
noise, and .9796 [.4012, 1.6702] with misaligned noise. Matched momentum losses
agree to floating-point accuracy. SOAP's corresponding intervals include zero.
These comparisons include independently selected rates; they characterize the
tuned policies, not a fixed-rate marginal effect.

Clean AdamW means are outlier driven: one of five diagonal runs has tail loss
.000524, while one rotated run has .03315; the other runs are very small.
Its clean paired rotation interval spans zero broadly: .6943 [-4.8983, 6.3148].
Therefore the clean mean alone is not a reproducible rotation failure claim.
All AMSGrad/SGD clean evaluations are below the 1e-12 comparison floor. SOAP
has a smaller arithmetic mean than AdamW there, but worse paired log loss
because four AdamW trajectories already approach zero. We retain both summaries
instead of selectively choosing the favorable aggregation.

On rotated isotropic noise, SOAP/AdamW log loss is -.7022 [-1.3042, -.1667].
Yet SOAP/Momentum is +.0529 [-.2095, .3334]; no resolved structured advantage
over the cheap first-order comparator. SOAP takes about .891 s/run versus
Momentum .341 s/run, and stores 768 versus 128 tensor-state bytes. These tiny
CPU times include snapshot diagnostics and I/O, with limited timing precision;
they are not GPU throughput claims. Geometry snapshot costs are separately
recorded per run. The isotropic control favors AdamW/SOAP over selected SGD,
showing that no optimizer wins every regime.

Held-out aligned/antialigned noise retains the same frozen configurations from
isotropic-noise development. Rotated AdamW means remain about .185/.188;
Momentum .01324/.01390 and SOAP .00953/.01495. Direct SOAP/Momentum intervals
include zero in both. No evaluation-based retuning was performed.

## Scaling mismatch and useful history

At call 2,400, mean frozen effective conditions kappa(B^(1/2) H B^(1/2)):

| Regime | AdamW diagonal / rotated | SOAP diagonal / rotated |
| --- | ---: | ---: |
| Clean | 104.9 / 903.2 | 5.46 / 648.0 |
| Isotropic noise | 197.3 / 1,273.4 | 840.3 / 978.0 |
| Misaligned noise | 200.1 / 1,271.8 | 725.0 / 975.6 |

Momentum's frozen B is identity, condition 1,000, despite its excellent clean
convergence. This statistic excludes momentum, learning-rate magnitude and
the evolution of state; it is **not** a standalone performance predictor.
SOAP's condition improvement does not by itself establish equal-quality
compute benefit. Euclidean scaling-mismatch and actual step directional
curvature are saved alongside it.

For AdamW on rotated objectives, final rank-4 history captures the following
fractions of tr(H); oracle top-four capture is .8420:

| Noise control | History capture | Random capture | Coordinate-only capture | History − random [95% run interval] |
| --- | ---: | ---: | ---: | ---: |
| Isotropic | .4874 | .2707 | .4190 | .2167 [.0925, .3423] |
| Independent anisotropic | .4545 | .2707 | .4190 | .1838 [.0832, .2843] |
| Held-out aligned | .7909 | .2707 | .4190 | .5201 [.4592, .5709] |
| Held-out antialigned | .0439 | .2707 | .4190 | -.2268 [-.2719, -.1817] |

Centered history gives the same qualitative contrast (.8103 aligned versus
.0407 antialigned). On diagonal isotropic noise, history capture .2755 is below
random .3021 and far below coordinate/oracle .8420. Thus history is neither a
universal curvature estimator nor automatically more useful than diagonal
information. Snapshots at 100/1,200/2,400 permit phase comparisons; bootstrap
units are complete seed runs at each phase. Isotropic H eigenvectors are
nonidentifiable and its subspace overlaps are excluded from interpretation.
All subspace intervals are exploratory, conditional on tuning, without a
multiple-comparison correction.

The [mathematics](sprint3_mathematics.md) proves a limited constant-diagonal
scaling disadvantage on a symmetric 2D rotation and explains why
Cov(g)=H Cov(x) H+Sigma does **not** imply Cov(g)=H. This is classical geometry;
it neither proves actual Adam's stochastic behavior nor promises a new method.

**Decision:** geometry is relevant in these toys, but broad first-order tuning
is unfinished, momentum already gives competitive quality/cost, and history
fails a decisive noise-direction control. A full TrustGeoOpt optimizer remains
unjustified.
