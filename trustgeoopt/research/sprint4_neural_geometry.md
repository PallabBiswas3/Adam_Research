# Sprint 4 neural geometry validation

**The useful quadratic information distinction does not deliver the predefined
neural accuracy/cost gain.** The small MLP already has useful gradient/noise
subspaces. Known HVP probes add only about one percentage point of final GGN
trace capture and exceed the refresh-cost gate. Deep/Fashion expansion and an
optimizer proof-of-concept are deferred by the frozen decision rule.

Evidence: `results/sprint4/campaign-20261012/processed/neural_estimators.csv`,
`neural_gate.json`, `neural_cost_details.json`, `neural_subspace_drift.csv`,
`neural_training_runs.csv`, plus the phase/cost/drift plots.20 training runs:
ten fresh initialization/order seeds60–69 on each of independently generated
teacher/feature tasks800/801. No task or policy is selected using these runs.

## Model and identifiable geometry

12correlated features,512 training/256validation examples, train-only feature
normalization; one16-wide ReLU layer and binary logits. Fixed AdamW .003 cosine,
batch32,20 epochs/320 updates. This is a measurement trajectory, not a tuned
neural optimizer competition. Validation measures training progress; geometry
oracles and gradient draws use training only. No official test/Fashion data are
used in Sprint 4. No prior checkpoint or evaluation seed is reused for selection.

At epochs0/5/20, independent with-replacement minibatch pairs are drawn at
exactly identical parameters. The128 gradient calls (64 pairs) provide both
layer blocks simultaneously. Autograd gradients are validated against the
closed-form per-example gradients. Store per-layer input activation covariance
and uncentered moments, gradient history, frozen-gradient covariance, estimated
noise covariance, and actual subspaces in each raw snapshot.

The full-training **layer-block** GGN is a small analysis oracle:208 parameters
in the hidden layer and34in the output layer. No dense network-wide Hessian is
allocated. Inside fixed ReLU cells, each layer's logits are affine in that
layer's weights/biases, so its Hessian block equals its GGN. Tests check the
identity against autograd. Full-network cross-layer Hessian terms are omitted;
there is no claim that empirical Fisher/noise covariance equals the full Hessian.
Published K-FAC/EKFAC/GGN information distinctions remain relevant; those full
optimizers are not benchmarked here.

Actual sampled HVPs use one fixed128-example with-replacement curvature batch
per snapshot, independent of paired gradients. Same-minibatch FD reuses it;
independent FD changes batch on every evaluation. The fixed batch action is
compared to the full-training GGN, not to itself. For hidden-layer FD,
perturbations can cross ReLU boundaries; δ=.01was frozen and not selected after
seeing error. At epoch 0, there is no optimization history: the history control
is a frozen-point gradient sample and is explicitly labeled accordingly.

## Trace capture across phases

Means over10 runs and equally weighted layer fractions; each fraction is
relative to its own block trace, not a network-wide trace weighting.

| Task / epoch | Centered history | Frozen gradient covariance | Paired noise space | Estimated subtraction | HVP/Ritz | Shared-batch FD | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|800 /0 |.5615 |.5615 |.5624 |.2967 |.5970 |.4720 |.6226 |
|800 /5 |.5720 |.5835 |.5855 |.4405 |.6010 |.5348 |.6284 |
|800 /20 |.6052 |.6026 |.6063 |.3755 |.6159 |.5502 |.6475 |
|801 /0 |.5512 |.5512 |.5442 |.3393 |.6031 |.4842 |.6273 |
|801 /5 |.5692 |.5829 |.5823 |.4499 |.5996 |.5283 |.6322 |
|801 /20 |.6081 |.6077 |.6100 |.3455 |.6175 |.5362 |.6539 |

Final random capture=.0598/.0557; coordinate-only=.1743/.1715. Paired-noise
directions already capture useful geometry here, unlike the antialigned toy.
This is an empirical task-specific relationship, not an equality of matrices.
The neural control does not impose an independently rotatable label-noise
covariance, and therefore is not a direct replication of the quadratic mismatch.

Paired final HVP-minus-centered-history capture:

- Task800: **+.01069 [.00176,.01972]**.
- Task801: **+.00942 [-.00081,.02114]**.

The criterion required ≥.05mean improvement and a positive lower bound on
both tasks. It fails on magnitude and on one task's uncertainty. Shared-batch
FD changes capture by -.05494 [-.07575,-.03515] and -.07185
[-.09458,-.04633]; noise subtraction by -.22964 [-.27438,-.19124] and
-.26264 [-.30491,-.21566]. All intervals resample whole seeds, averaging layers
within a seed; no step/layer pseudo-replication or multiplicity claim.

HVP mean actual reconstruction error=.4506; shared FD1.0827; independent
FD10.7985, including rank truncation and minibatch/operator error. The FD
failure could involve ReLU-boundary crossing and finite-difference scale, but
those causes are not separately isolated; no universal FD failure is claimed.
Noise covariance relative error averages .3601with64 pairs. Subspace drift
across0→5and5→20is retained in CSV/plots, with independent probe draws at each
phase; it cannot be attributed entirely to a changing population space.

## Cost and gate

Measured mean training step=.001468 seconds, including loader/training and
history capture. One batch32 gradient averages .000300 seconds; collection of
128same-point independent gradients averages .03844 seconds. One128-example
HVP/Ritz refresh averages .02731 seconds per layer (24 products plus QR/Ritz).
Refreshing both layers every100 steps averages **.3926 of measured step cost**;
shared FD .3871. Both exceed the10% gate. These are measured overhead fractions
for the stated implementation/refresh, not hardware-independent lower bounds.
Larger refresh intervals might reduce cost but introduce stale geometry; they
were not tuned on evaluation data.

Paired gradients supply both layers at once: amortized gradient-collection
cost=.2744, before PCA/subtraction. The gate's sum of layer-attributed paired
timings is deliberately conservative (.5795for subtraction), double-counting
the shared collection. Even the corrected once-only collection exceeds10%,
so this accounting detail does not change the decision. It is explicitly
reported rather than passed off as an exact twice-paid cost.

Retained HVP basis/eigenvalue state:6,688 bytes hidden and1,120 bytes output in
float64 diagnostic algebra. History matrices and autograd graphs are additional
workspace. Small full-training GGN oracles and metric eigensolves are excluded
from deployable refresh timers but included in total diagnostic/campaign time.
The quadratic timings do not transfer to these networks. CPU timing is a single
shared-process measurement; no isolated peak-RSS/GPU-memory claim is made.

## Decision

There is no robust ≥5-percentage-point incremental neural capture gain at the
chosen cost, and no demonstrated equal-total-compute benefit from using the
estimated geometry in an optimizer. No estimator steers training, so accuracy
cannot be presented as an optimization gain. Established probes provide the
information being tested; the contribution is not a new estimator. Preserve
this negative gate and pivot to measurement/identifiability and baseline
selection research before proposing another optimizer.
