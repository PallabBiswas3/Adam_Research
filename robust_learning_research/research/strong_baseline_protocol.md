# Strong baseline protocol, registered before the feasibility pilot

Branch `research/robust-learning-baseline-validation`, parent `947ef97`.
Question: does the measured hard-clean/tail-selection tradeoff survive faithful
DSS (including its published prior-aware extension) and DivideMix on CIFAR-10?
No new method. Earlier synthetic evaluation seeds and results are unsealed and
cannot be used to tune this benchmark.

## Source and fidelity gate

Primary DSS main/supplement and DivideMix Algorithm 1 are the specification.
Inspect pinned official implementations, dataset loaders, launch scripts,
augmentation and hyperparameters; mark equation/code ambiguities. DSS has no
verified source-reuse license at the inspected pin: independent equation
implementation only. DivideMix is MIT: reuse its exact PreActResNet-18 with its
license and hashes; implement training equations independently. Unit-test MDA,
prior estimation, CCS normalization and zero excluded-logit gradients, trend
statistics, GMM orientation, refinement/sharpening, peer isolation and SSL loss.

## CIFAR-10 data protocol

Use the official 50,000 training/10,000 test 32x32 RGB images. Reserve 500
training examples per true class **before corruption** with split seed1601.
This is offline benchmark construction, never a selector input. Remaining
4,500/class form the balanced pool. For imbalance factor10, deterministically
sample floor(4500 * 10^(-class_index/9)) per class with independent RNG. Balanced
and long-tail regimes therefore share original image IDs where possible.

From the reserved pool, choose a separate prior-calibration sample with counts
floor(100 * 10^(-class_index/9)); estimate the prior only from its clean labels,
as DSS supplement Eq.9 specifies. Choose 200/class from disjoint remaining
reserved images for balanced model-selection validation. Prior-calibration and
model-validation IDs are disjoint from train/test. A balanced validation set
alone cannot estimate a long-tail population prior. All methods have the same
available clean-calibration information; any use must be disclosed. Oracle
training counts/clean labels are inaccessible to selectors. Uniform-prior DSS
is retained as a published sensitivity control.

Freeze corruption once: symmetric noise chooses replacement uniformly among
all ten classes for 50% sampled interventions, allowing unchanged labels (actual
flip proportion expected45%, matching DivideMix's main-paper convention).
Class-asymmetric40% uses truck->automobile, bird->airplane, deer->horse,
cat<->dog; other classes unchanged, so nominal intervention and actual global
flip rates differ. Hash observed labels, actual flip mask, subset IDs and config.
Train dataset exposes only pixels, observed labels and IDs; true labels/flip
flags are separately sealed in evaluation artifacts. Noise, split, augmentation
and model RNG streams are distinct. Unit tests cover label fidelity and isolation.

## Feasibility pilot, authorized now

One seed601, CPU-only installed environment, no upgrade or paid GPU. Load real
CIFAR-10; use a fixed128-image subset solely to exercise kernels, with batch16,
four compressed warmup microepochs and three timed update rounds. Same first
network initialization and example order per method. Full PreActResNet-18,
SGD settings and defining operations are retained: DSS MDA/CCS and prior,
DSS+ peer selection and weak/strong MixUp consistency, DivideMix two-network
GMM division, two-view refinement/co-guessing, sharpening, MixUp, labeled CE,
unlabeled MSE and prior regularization. No true training labels enter training.

This is an operation-level feasibility pilot, **not faithful completed training**:
subset, batch size, compressed warmup, early histories and nonconverged mixtures
differ. No benchmark superiority, hard-clean failure or final accuracy claim
can be inferred. Record these deviations and any failed/censored stages. Cap
180 seconds of training, process RSS1800MiB and minimum available700MiB; checks
at operation boundaries are not hard OS limits. Stop rather than replacing an
empty GMM split with an invented selection rule.

Measure per-update training/augmentation/selection wall time, process RSS,
parameter/SGD-state/history bytes, full-dataset forward-scan and GMM estimates.
Projection must specify batch16 versus published128, selected fractions,
network counts, epochs and missing I/O/checkpoint/calibration overhead. Batch128
memory and GPU throughput cannot be established by linear CPU extrapolation.
No full campaign is currently authorized. Hard stop after pilot if projected
faithful benchmarking is infeasible locally.

## Conditional full benchmark (not launched automatically)

Six methods: CE, classwise small loss, DSS uniform, DSS estimated prior,
DSS+ estimated prior and DivideMix. Run published horizons/schedules rather than
equalizing epochs by dropping SSL components. Report algorithm costs explicitly.
All share backbone/data/noise masks; one/two networks and differing horizons are
intrinsic method costs. A full result must pass warmup/selection transitions and
end-of-run completeness checks before comparison.

Development701/702, evaluation801-805. Two preregistered candidate settings per
method, initially covering three regimes: balanced symmetric50%, long-tail10
symmetric50%, and held-out long-tail10 asymmetric40%. Pick policies only on
development clean validation; confirm exact candidate settings before a campaign.
Do not reuse the pilot to choose final hyperparameters. No test checkpoint
selection; final-horizon primary results, any last-10 averaging preregistered.
The held-out asymmetric regime is excluded from development. The prospective
plan is48 development runs (6methods *2candidates *2seeds *2regimes) plus90
evaluation runs (6methods *5seeds *3regimes). This is a costed proposal, with
no implied launch approval; individual experiment runs are not independent
statistical replicates across shared seeds/regimes.

Primary metrics: clean test per-true-class recall/F1, macro-F1, balanced accuracy,
ECE/reliability diagrams and total training wall time. Selection evaluation uses
true training labels only after frozen models: actual flip rejection precision/
recall, accepted-clean precision, tail-clean retention and full operating curves.
Rejected DivideMix examples still supply unlabeled supervision; report both label
acceptance and training participation. DSS CCS modifies class supervision rather
than simply rejecting whole samples.

Use ten-class unweighted mean F1 (zero division maps to0), balanced accuracy
as unweighted mean true-class recall, and top-label ECE with15 fixed equal-width
confidence bins. Hard-clean retention is accepted-clean-and-hard divided by
all clean-and-hard training examples under the frozen independent mask. Noise
rejection precision is actual flips among rejected labels divided by all
rejected labels; report undefined empty denominators and achieved rejection
recall rather than inserting perfect precision. Separate tail/head results,
label acceptance and participation, and use matched final-horizon ensembles
for two-network methods. These definitions apply to future full evaluation;
none was computed from the timing pilot.

Hardness has no known Bayes oracle on CIFAR-10. Preregister an independent CE
reference trained on observed labels, and define hard examples by its clean-label
margin <=.1 **in evaluation only**, fixing the mask before comparator analysis.
Disclose model-dependent difficulty and use class-stratified sensitivity; do not
redefine hard cases based on each method's outcome. Pilot timing has no such
trained reference and reports no hard-clean scientific outcome.

Use whole-seed paired resampling across regimes and report five-seed limitations,
feasibility and actual achieved precision. An oracle-picked curve point is only
descriptive evaluation, never deployment calibration or accuracy at a retrained
operating point. Claim a residual only after fair tuning, fidelity checks,
independent seeds, prior-aware DSS and class-aware controls all fail to close it.
