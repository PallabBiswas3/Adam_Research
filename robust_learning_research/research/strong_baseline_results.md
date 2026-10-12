# Strong baseline results: executed feasibility pilot only

**Scientific outcome: undetermined.** No converged CIFAR-10 quality comparison
has been executed. The prior synthetic hard-clean/noisy-label failure has neither
survived nor been refuted by faithful DSS or DivideMix training in this sprint.

| Method | Status | Warmup network updates | Main rounds | Total seconds | Peak RSS MiB |
|---|---|---:|---:|---:|---:|
| ce | completed_kernel_pilot | 32 | 3 | 13.24 | 567.1 |
| classwise_small_loss | completed_kernel_pilot | 32 | 3 | 16.25 | 569.4 |
| dss_uniform | completed_kernel_pilot | 32 | 3 | 16.50 | 570.2 |
| dss_estimated_prior | completed_kernel_pilot | 32 | 3 | 18.19 | 570.2 |
| dss_plus_estimated_prior | completed_kernel_pilot | 64 | 3 | 62.26 | 903.6 |
| dividemix | completed_kernel_pilot | 64 | 3 | 52.01 | 1097.6 |


These are completed or censored operation-level pilots, not published training
reproductions. Warmup microepochs, partial CCS history and mature-path SSL loss
weight are disclosed in the fidelity reports. Three rounds support no general
uncertainty estimate, ranking of accuracy, or claim that selection is robust.

Losses, timings, GMM posterior fractions, accepted-label counts and failure
reasons are saved in the individual JSON records. They are optimization
sanity/cost observations; GMM posterior is **not measured noise precision**.
The pilot does not open true training-label oracles, validation or test files.
Calibration labels are read only to estimate the published DSS prior.

| Requested quality measure | Status |
|---|---|
| True class-wise recall/F1 | Not executed |
| Hard-clean retention | Not executed; no independent trained reference yet |
| Rejected-noise precision/recall | Not executed |
| ECE/calibration plots | Not executed |
| Macro-F1 / balanced accuracy | Not executed |
| Total faithful training time | Not executed; pilot timings and rough requirements only |

Do not fill these fields with toy-study results, paper accuracies or extrapolated
quality. Fresh development701/702 and evaluation801-805 seeds remain unused.
Classwise small loss here is a known generic control, not a faithful RoLT/PASS
implementation. None of the methods can be declared to solve/fail the realistic
benchmark from these observations. No quality statistical comparisons or
confidence intervals are warranted at this stopping point.

Reproducible provenance: [frozen source/config](../results/strong-baseline-20261012/pilot/freeze.json),
[executed records](../results/strong-baseline-20261012/pilot/results.json),
[summary CSV](../results/strong-baseline-20261012/pilot_summary.csv),
[resource assumptions](benchmark_feasibility.md).
