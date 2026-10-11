# Experimental protocol

Primary synthetic endpoint: time to f_t/f_0<=0.01 for fixed zero-optimum SPD
objectives (wd=0), with steps-to-target secondary. Tracking reports distance to
instantaneous optimum and integrated tracking error separately. Rosenbrock has
known optimum all ones; its finite-step progress is not a proof of convergence.
All pilots are exploratory; no chosen optimizer learning rates imply superiority.

Phase 1: matched seeds/start, spectra (min eigenvalue 1), rotation, noise streams,
steps, dtype float64. Conditions 10/100/1000/10000; diagonal/rotated/isotropic;
Rosenbrock; switched curvature/optimum; isotropic and anisotropic noise. Vary beta2,
lr, momentum. Show gradients, displacement, actual descent cosine, Rayleigh
curvature, vhat moments, spikes, finite status, time and memory. Reference Shampoo
reshapes synthetic p-vector to a configured a by b parameter matrix, so structure
is explicit. It is not an exact Hessian oracle or maintained distributed implementation.

Fair tuning: same 12 lr/wd trials per baseline on development seeds; tune model
and optimizer settings without held-out seeds/test sets. Record all trials,
including errors/censored targets; charge tuning cost separately. Same initialization,
preprocessing, schedule (constant initially), clipping (disabled initially), stopping,
evaluation frequency, device and thread count. First-order baselines use official
PyTorch; custom AdaBelief uses unrectified published Algorithm 2, including epsilon
in the residual accumulator and denominator, plus explicit decoupled weight decay.

Phase 2: scores before update, labels after update. Evaluate each scalar and simple
gradient norm on identical AdamW trajectories. Signed mismatch and negated alignment
orientations predefined. Split by complete seed/run, never randomly split time steps.
Precision/recall/FPR thresholds fitted only on development runs; report AP/AUROC,
class counts and undefined outcomes. Future HVP/two-batch scores charged at full cost.
Phase 3: Woodbury vs dense tiny reference, rank/damping/random controls; benchmark
refresh and solve time separately, include all allocations and synchronize CUDA.

Neural data: torchvision official train/test partitions; derive validation from official
training set using fixed split_seed. Fit tabular normalization only on training data;
vision uses fixed published normalization. Training noise/label noise only on train.
No test evaluation unless explicitly requested after model selection. Synthetic has
independent train/validation/test examples from one fixed teacher and feature map.
MLP depths/activations, small CNN, CIFAR-stem ResNet-18 are supported.

Primary campaigns use >=5 evaluation seeds. Save per-seed raw curves, configuration,
versions, dataset/split, source hash and git if available. Summary tables have n,
mean/std, descriptive 95% t CI, all failure counts. Paired bootstrap inference and
multiple-comparison control are subsequent work, not supplied by pilot CIs.
Default budgets do not early-stop; threshold attainment is evaluated from saved rows.
Timing includes measurement overhead; log optimizer/diagnostic/evaluation sections.
CPU memory is process high-water RSS (not isolated run peak); CUDA reset peak
allocated memory is measured. Python allocation peak is a separate limited measure.

Later ablations: no gate, no correction, constant alpha, random directions, ranks
{1,4,8}, refresh {1,10,50}, damping {1e-4,1e-2,1}, signals separately/combinations,
HVP diagnostic, and each safeguard individually. Do not execute optimizer ablations
until its design passes gates; current ablation runner examines baseline beta2 and
diagnostic thresholds, and explicitly labels these as mechanism pilots.
