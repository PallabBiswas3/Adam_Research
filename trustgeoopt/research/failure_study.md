# Phase 1 exploratory evidence

2026-10-12. Sources: `results/pilot/raw`, per-run CSVs and metadata; generated
`results/pilot/tables/individual_runs.csv`, `summary.csv` and two PNG/PDF plots.

80 successful CPU attempts: seeds 0-4, 16-dimensional SPD quadratic, kappa=100,
diagonal versus orthogonal rotation, 100 updates, float64, matrix parameter shape
4 by 4, wd=0. All eight baselines used lr=0.001; this is **untuned** and not a fair
optimizer ranking. Every method had the same start for a given seed, spectrum,
measurement pipeline and budget. Momentum used 0.9. Shampoo uses row/column structure
of the explicit 4 by 4 parameter representation, not knowledge of the true Hessian.

| Baseline | Diagonal final loss mean (SD), n=5 | Rotated final loss mean (SD), n=5 |
| --- | --- | --- |
| SGD | 6.7237 (3.0536) | 5.3614 (2.4408) |
| Momentum | 0.1355 (0.1524) | 0.0699 (0.0553) |
| Adam | 130.1372 (42.6970) | 180.3597 (124.8664) |
| AdamW | 130.1372 (42.6970) | 180.3597 (124.8664) |
| RMSProp | 107.7410 (38.8390) | 143.8316 (104.3606) |
| AMSGrad | 130.1372 (42.6970) | 180.3597 (124.8664) |
| AdaBelief reference | 87.6363 (34.2466) | 114.2836 (84.9843) |
| Shampoo reference | 153.1547 (45.9841) | 220.1665 (145.1234) |

The reference momentum reached the relative error target 0.01 in all ten runs;
SGD reached it in one rotated run. These observations mainly reveal the sensitivity
of an equal numerical lr comparison, not matched tuning. At wd=0 Adam and AdamW
coincide as expected. AMSGrad is near-identical in these short trajectories; its
known long-memory remedy was not challenged by the published counterexample here.
The structured reference did not excel at this rate; structure alone is no benefit.

Independent AdamW smoke: lr=0.03, rotated kappa=100, seed=0, 100 updates reduced
loss 319.5219 to 5.3306, no harmful updates/spikes. It did not attain 1% relative
error within this budget. This rate is not compared as a tuned winner.

All timings include diagnostics, exact toy loss/curvature probes and logging.
Pilot suites were run concurrently on CPU; do not rank methods by these small
wall-clock measurements. CPU peak memory is process-wide; the raw data is available
but does not isolate incremental optimizer memory. Use separate-process timing runs
with warmup and instrumentation controls for Gate E.

Gate B is unpassed. Next: tune lr fairly; cover kappa=10/100/1000/10000; isotropic
negative control; more budgets; no-noise/noise conditions; published online convex
Adam counterexample reproduction with its actual assumptions. Poor finite-step
progress or a loss increase is not mathematical nonconvergence.
