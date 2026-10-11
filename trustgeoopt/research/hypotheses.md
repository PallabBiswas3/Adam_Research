# Preregistered hypotheses (initial plan)

No hypothesis is supported by the initial smoke test alone.

| Hypothesis | Null | Primary quantities | Required test | Confound/control | Falsification |
| --- | --- | --- | --- | --- | --- |
| H1 reliability | cheap state scores add no predictive value over gradient norm | held-out-seed AP, AUROC, recall at 5% FPR, lead 0/1/5 steps, scoring time | AdamW stationary/shift/noise pilots, seed split 0-4 development, 5-9 evaluation | label prevalence; lr; noise; pooled steps; compare same runs | AP lift CI includes 0 or fails across held-out regimes |
| H2 geometry | off-diagonal correction provides no useful loss progress advantage | steps/time to relative quadratic error <=0.01; validation loss | rotated versus diagonal H; correlated regression; isotropic controls; Shampoo comparison | damping, direction rescaling and clipping; random subspace | matched budget advantage absent or random directions explain it |
| H3 adaptive computation | gating costs as much as best fixed schedule at equal quality | total time to fixed target including scores, SVD and backward | fixed, periodic, gated schedules with identical ranks | unequal geometry sample counts, CPU synchronization | no time reduction under paired-seed CI or lower quality |
| H4 robustness | gains do not survive seeds/tasks/settings | mean/std/CI; failure rate; per-seed curves | >=5 seeds, MLP/deep MLP/CNN/ResNet; batch and lr grid | validation tuning; initialization and data splits | gain concentrated in one seed or stress test; unseen task regression |

Harmful update primary label: f_t(theta_next)>f_t(theta)+1e-12*max(1,abs(f_t(theta))).
Secondary spike: increase >10% of max(abs(f_t(theta)),1e-12); both are *outcomes*.
On neural minibatches, use a fixed independent probe batch before/after, never
use its post-update outcome as a score. Do not use test data for labels/tuning.
Predefine thresholds using development seeds only. AP is undefined with no positives;
AUROC is undefined with a single class. Record null, not a perfect value.
Independent runs, not correlated steps, are the uncertainty unit. CIs for comparisons
use paired seed bootstrap (10,000 draws) in primary studies; pilot tables give descriptive
Student-t intervals only. Failure is nonfinite objective/gradient/update or unmet
target at the budget. An unmet target is right-censored, not zero time.
