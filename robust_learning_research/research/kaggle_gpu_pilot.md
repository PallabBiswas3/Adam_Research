# Kaggle GPU pilot: review before manual execution

This branch prepares a **timing and memory feasibility pilot**, not a completed
research benchmark. The prior [baseline results](strong_baseline_results.md)
measured 128-image CPU kernels at batch 16. GPU throughput and full-data memory
were unmeasured. No Kaggle execution or dataset upload is part of this branch.

## Registered paper/code decisions

| Issue | Paper | Inspected implementation | Fixed pilot choice |
|---|---|---|---|
| DSS+ pseudo-target | Supplement Eq. 6 uses raw weak-view probability raised to `1/T`; table sets `T=2` | Official `mixup_loss` uses MDA-adjusted target raised to `T` | **Paper convention:** raw probability, exponent `1/T`, `T=2`. The existing independent kernel is retained. This is not an official-code reproduction. |
| DivideMix unlabeled MSE | Eq. 10 sums squared class errors per example | Official `Train_cifar.py` uses `torch.mean((probs_u-targets_u)**2)`, averaging over examples **and classes** | **Code convention:** class mean and `lambda_u=25`. At 10 classes the paper sum is 10 times this loss for the same lambda. |
| DivideMix epoch indexing | Paper specifies 300 epochs | Inspected inclusive code loop can execute 301 | Pilot runs only one warmup and one mature-operation epoch; this difference cannot affect timing of these stages, and no full-run horizon is claimed. |
| Symmetric noise | DSS base excludes the original label | DivideMix main setup allows same-class replacement | Common registered corruption from the baseline protocol: 50% interventions with uniform replacement including the original; expected actual flips 45%. This departs from DSS base. |

These choices are fixed **before** the GPU pilot and cannot be adjusted from its
outcome. The source decisions and limitations are detailed in
[DSS fidelity](dss_fidelity.md) and [DivideMix fidelity](dividemix_fidelity.md).
Primary sources: [DSS paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.pdf),
[DSS supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Pan_Debiased_Sample_Selection_CVPR_2026_supplemental.pdf),
[DivideMix paper](https://arxiv.org/pdf/2002.07394),
[DivideMix trainer](https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py).

## What the notebook measures

The self-contained notebook embeds the reviewed scripts, independent equation
code, and the MIT-licensed DivideMix backbone. It verifies embedded SHA256
hashes at runtime. Its metadata specifies a private Python notebook, GPU on,
internet off, and no dataset sources. It does not read credentials, call the
Kaggle CLI, fetch a dataset, or install packages.

Attach an existing Kaggle dataset containing the official, **extracted**
`cifar-10-batches-py` directory under `/kaggle/input`. Torchvision verifies
the standard batch MD5s before loading it. The notebook constructs the same
long-tail split and observed labels as the baseline protocol: 18,388 training
images, 403 separate clean prior-calibration labels, seed 601, split seed 1601,
and corruption seed 2601. The runner does not open validation or test images
or pass clean training labels to the models.

For each method, two independent full PreActResNet-18 learners run one full
warmup epoch. DSS+ then computes frozen peer policies and runs one complete
operation epoch with weak and strong views, MixUp, MDA/CCS, and consistency.
DivideMix scans all training images for each learner, fits two GMMs, then runs
one complete labeled-partition epoch with unlabeled cycling, refinement,
co-guessing, MixUp, labeled CE, unlabeled MSE and prior regularization. Its
unlabeled weight is deliberately fully ramped to profile mature-path cost;
this is **not** the training trajectory after a one-epoch warmup.

The notebook gives the runner a 1,770-second internal limit and enforces a
1,800-second process timeout, including startup and data loading. The runner
checks at minibatch boundaries and records a censored result if the budget, an
empty GMM partition, or CUDA OOM prevents completion. If a minibatch crosses
the internal deadline, the notebook stops the process at 30 minutes and marks
the active method censored in the saved report. CUDA synchronization brackets stage timings. Stage records
include elapsed seconds and peak allocated/reserved VRAM; the report includes
GPU name, total VRAM, DSS host-history bytes, data counts, hashes and completion status. A stage is
**not** a full-data measurement unless its completion field is true. DSS+ has
only one epoch of CCS history before its timed operation; this profiles code
path and memory at early history length, not late-horizon CCS cost. No checkpoint,
converged quality, evaluation overhead or 150–300 epoch projection is measured.
The first CUDA stage includes cold-start overhead; stage timings are device and
session-specific and are not a method speed ranking.

## Manual Kaggle upload and run

1. Review [the notebook](../kaggle/robust_learning_gpu_pilot.ipynb),
   [runner](../scripts/run_kaggle_gpu_pilot.py), this decision record, and
   [metadata](../kaggle/kernel-metadata.json).
2. In Kaggle, create or import a **private** notebook under your account and
   upload `robust_learning_gpu_pilot.ipynb`. Set the GPU accelerator and turn
   internet access off. Confirm the notebook remains private. The JSON metadata
   is prepared for reference; a browser upload may require setting these options
   in the Kaggle UI.
3. Attach an existing Kaggle CIFAR-10 dataset with an extracted
   `cifar-10-batches-py` folder. If several attached datasets match, edit the
   final notebook cell to select the intended root explicitly. Do not upload a
   local dataset for this pilot.
4. Run the notebook yourself. Download and review
   `/kaggle/working/robust-learning-gpu-pilot/gpu_pilot_report.json`.
   A fresh Kaggle session is required for another run because the runner refuses
   to overwrite an existing report directory.

The [benchmark protocol](strong_baseline_protocol.md) still gates any full
comparison. These timings alone cannot establish method quality, reproducible
GPU cost across devices, or a scientific conclusion.
