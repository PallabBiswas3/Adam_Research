# Benchmark feasibility: stop after the bounded pilot

One seed601 on actual CIFAR-10 images, full PreActResNet-18, CPU float32,
two PyTorch threads. Installed PyTorch 2.1.2+cpu,
Python 3.11.4; CUDA unavailable to this environment.
No dependency upgrade, paid GPU or full campaign was run. An absent nvidia-smi
command does not establish that this machine has no physical GPU.

Prepared data: 18,388 long-tail training images, 2,000 separate
model-validation images, 403 separate prior-calibration labels,
10,000 clean test images. The pilot uses128 training images and batch16 only.
Archive MD5 matches torchvision's original CIFAR-10 MD5; the original host was
slow, so the verified identical archive came from a pinned mirror.
[Dataset hashes](../results/strong-baseline-20261012/dataset_provenance.json),
[mirror/setup provenance](../results/strong-baseline-20261012/mirror_provenance.json).
Original [dataset source](https://www.cs.toronto.edu/~kriz/cifar.html);
[mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python).
Dataset access is separate from MIT software licensing; no dataset images or
unlicensed DSS source are redistributed through Git.

## Actual operations and cost

| Method | Status | Warmup network updates | Main rounds | Total seconds | Peak RSS MiB |
|---|---|---:|---:|---:|---:|
| ce | completed_kernel_pilot | 32 | 3 | 13.24 | 567.1 |
| classwise_small_loss | completed_kernel_pilot | 32 | 3 | 16.25 | 569.4 |
| dss_uniform | completed_kernel_pilot | 32 | 3 | 16.50 | 570.2 |
| dss_estimated_prior | completed_kernel_pilot | 32 | 3 | 18.19 | 570.2 |
| dss_plus_estimated_prior | completed_kernel_pilot | 64 | 3 | 62.26 | 903.6 |
| dividemix | completed_kernel_pilot | 64 | 3 | 52.01 | 1097.6 |


Shared training elapsed 178.81s; OS lifetime peak
RSS 1097.6MiB. Per-method RSS sampling is20ms;
OS peak includes imports/data preparation inside the pilot. Guards checked at
operation boundaries:180s training,1,800MiB process RSS,700MiB system available.
These are not hard OS limits. Failed/censored attempts and incomplete rounds
remain in [all records](../results/strong-baseline-20261012/pilot/results.json).

Single learners have about42.6MiB parameter storage plus about42.6MiB SGD momentum;
two learners double those tensor costs. Activations dominate variable memory,
especially DivideMix's concatenated four-view batch. Measured RSS at batch16
does **not** certify published batch128 memory feasibility. Full DSS histories
grow with examples and epochs (float32 N*10*epochs per learner); at N45K and
150epochs this is about257.5MiB per learner, beyond the pilot's short history.
Temporary history stacking and scoring add memory and CPU cost.

## Rough CPU requirements, not benchmark results

Scale the median measured network warmup update and median main round by
ceil(N/16), published horizons/warmups, and learner count. Add full-dataset
forward scans by linearly scaling the measured128-image scan. DivideMix rounds
are two learner updates and use ceil(f*N/16) iterations/epoch, f in .1,.5,.9.
The table uses f=.5; other methods process all examples. CE/classwise controls
use the base150epoch horizon. Formula is in
[analysis script](../scripts/analyze_strong_pilot.py).

| Method | Long-tail hours/run | Balanced hours/run |
|---|---:|---:|
| ce | 17.4 | 42.5 |
| classwise_small_loss | 26.1 | 63.9 |
| dss_uniform | 25.3 | 61.8 |
| dss_estimated_prior | 22.8 | 55.7 |
| dss_plus_estimated_prior | 180.1 | 440.6 |
| dividemix | 298.6 | 730.7 |


The 138-run plan projects 20540 CPU hours in the middle scenario (12828-28253 hours across assumed DivideMix acceptance fractions). These are workload scenarios, not statistical confidence intervals.
Full-data GMM fitting, growing CCS history computation, validation, checkpoints
and data-loader overhead are omitted. Few compressed microepochs do not sample
mature selection. Brief pytest overlap occurred during middle single-network
timings; background contention makes this an order-of-magnitude resource screen,
not a fair speed leaderboard. No parallel experiment campaign is assumed.
Batch128 throughput, GPU memory/speed and money cost are unmeasured; linear
rescaling from batch16 cannot establish them.

**Hard stop:** faithful long training is outside the authorized180s pilot and
the current bounded CPU sprint. No large campaign is launched. To reopen the
gate, resolve fidelity ambiguities, measure full-data epoch and batch128 memory,
and approve an explicit device/runtime/run-count budget. This does not establish
that faithful benchmarking is impossible on another provisioned machine.
