# Sprint 3 frozen experiment protocol

Branch: `research/sprint3-mechanism-validation`, based on remotely verified
Sprint 2 commit `75875517abe2446ed11c29fcbc6fc236f1c466e1`.
Machine: Python 3.11.4, PyTorch 2.1.2 CPU, one PyTorch thread; no GPU.
Configuration: `configs/synthetic/sprint3.yaml`; immutable source/configuration
snapshot and environment metadata accompany the campaign. Code changes require
a new campaign, rather than silently changing later stages.

## Budget and independence

Cap: 1,800 seconds of experiment stages and 1,200 optimizer attempts.
Planned: 15 reproduction attempts, 477 stochastic attempts, 556 geometry
attempts, and 147 neural attempts = 1,195. Imports, download, tests, literature
work and postprocessing are outside this execution cap and are reported
separately where measured. This increases the earlier proposed 1,000-attempt
Sprint 3 plan by 195 small CPU attempts within its proposed time cap; the
current Sprint 3 request authorizes bounded CPU experiments without prescribing
an exact attempt cap. No GPU use or package upgrade.

Development seeds 30/31 and evaluation seeds 40–44 never overlap Sprint 2.
The 15 exact Sprint 2 replays use its five evaluation seeds **only to reproduce
previously frozen settings**; they cannot enter any selection function.
Selections minimize mean log10 tail loss + 1e-12 (quadratics), or final-epoch
validation loss + 1e-12 (neural), using development trials only. All required
development seeds must complete successfully. Failed/censored candidates are
disqualified and retained. Configurations are frozen on disk before evaluation.

## Stochastic experiments

Three regimes: rising, constant and falling Gaussian additive noise. Same
condition-100 Hessian, noise covariance condition 10, matched seed streams.
2,400 gradient calls; changes at call 1,201; last 20% true quadratic loss is the
primary metric. Three methods each receive 24 candidates x two development
seeds per regime. AdamW varies four rates, three beta2 values and two schedules;
SGD varies six rates, two momenta and two schedules; reference Shampoo varies
six rates, two inverse refresh intervals and two schedules. Cosine decay is
phase independent, ending at 1% of the initial rate. Zero weight decay.

Different parameterizations receive equal trial counts and horizons, not an
identical list of hyperparameters. Cost of each method is measured. Factor
refresh in Shampoo is not equivalent to beta2; no causal equivalence is claimed.
Momentum extends below the old lower boundary by 100x; Shampoo extends above
the old upper boundary from 3 to 10. Remaining boundary selections must be
reported, rather than described as globally optimal hyperparameters.

## Geometry experiments

Dimension 16, tensor shape 4x4, condition 1,000. Matched diagonal/rotated pairs
use H_rot = Q H_diag Q^T, x_rot = Q x_diag and noise_rot = Q noise_diag for the
same latent draws. This preserves initial loss, curvature spectrum and complete
noise spectrum. Three pairs: clean, isotropic noise, independent anisotropic
noise; plus a condition-1 noisy control. Anisotropic noise has condition 100 and
the same trace as isotropic noise. Each of AdamW, AMSGrad, Momentum SGD and
authors' SOAP receives six independently scaled rates x two development seeds.
Constant rates and default moments in this stage are a stated limitation; broad
schedule/beta2 tuning occurs in the stochastic stage, not a full geometry grid.

Held-out aligned and antialigned noise pairs use configurations selected on
isotropic-noise development. They consume no additional tuning. Five seeds
each; 2,400 calls. SOAP's first call initializes its basis without an update;
that call counts against its budget. Arbitrary orthogonal rotations need not
respect SOAP's Kronecker tensor structure; this is not a test of full-matrix
rotation invariance. Max factor dimension 128; its Fashion input-axis factor
is consequently omitted, with the output-axis factor retained.

Evaluation snapshots at calls 100, 1,200 and 2,400 record effective condition
of B^(1/2) H B^(1/2), frozen scaling mismatch, directional curvature and rank-4
subspace overlap/curvature capture. B excludes momentum and learning rate.
History uses the last 64 observed gradients, with both centered and uncentered
variants. Compare against random, coordinate-only and oracle Hessian subspaces.
Isotropic Hessian eigenvectors are nonidentifiable; its overlap is not evidence.
Snapshot cost is explicitly reported; all geometry trial times include it.

## Neural experiments

Correlated synthetic classification: 512 train, 256 validation, 20 features,
feature covariance condition 100, train-only feature normalization; small
32-wide ReLU MLP and six-hidden-layer 32-wide ReLU MLP, Kaiming initialization.
One teacher/task seed 700 for tuning, separately generated teacher/task 701
held out. Official Fashion-MNIST: fixed disjoint 2,048 training/512 validation
subsets drawn from its official training split, 32-wide MLP, official test
unused. Download URLs and hashes retained separately. No test-set selection.

Each architecture: AdamW, Momentum SGD and SOAP; four rates x two development
seeds, then five evaluation seeds. Twenty epochs, batch 64, identical seed
initializations and minibatch order within an architecture. Constant rate,
zero decay; no architecture, early-stop or schedule selection. Final validation
loss and measured total trial time are primary. Targets chosen before execution:
0.25 for synthetic tasks, 0.9 for Fashion. Report reach counts and censoring;
mean time among reached runs alone cannot establish a speedup. Optimizer tensor
state bytes exclude activations/allocator overhead; RSS samples occur each
epoch and are process-wide, not isolated per-trial peak allocation.

## Statistics, references and interpretation

Pair log10 loss ratios by complete evaluation run/seed; percentile bootstrap
2,000 resamples of five seed pairs. Conditional on two-seed tuning. No step-level
pseudo-replication, selection uncertainty, multiplicity correction or claims of
population-wide significance. Preserve paired raw differences. Near-zero
quadratic comparisons use a 1e-12 floor to avoid overinterpreting rounding.

Maintained Meta Shampoo currently requires Python >=3.12 and PyTorch >=2.8;
upgrading this working environment is outside this bounded sprint. Fallback is
the authors' **preliminary**, unmodified SOAP reference pinned at commit
`a1e553530fde97d0e6b307d7c82ac6d38b072340`, MIT license retained. It is not
represented as a production-maintained distributed Shampoo implementation.
Existing reference Shampoo remains labeled as the repository's small toy
implementation. TrustGeoOpt is not implemented. Exact fixed-B linear theory,
Adam approximations and conjectures are distinguished in the analysis.
