# Reliability study

Status: causal signal interfaces implemented; pilot scoring supported. Gate C open.

## Recorded first pilot

See [foundation_report.md](foundation_report.md) and
`results/reliability/processed/reliability-lead0.json` for actual pooled/per-run/per-regime
results. Fifteen 150-step AdamW trajectories cover seeds 0-4; seeds 0-1 fit thresholds
and 2-4 are held out. Noise-free, curvature/optimum-switch and noise-scale-switch
regimes are distinct. Negative momentum alignment and history consistency have
descriptive AP above norm; mismatch benefit varies by regime, and drift is weak overall.
Three held-out seeds and correlated updates do not establish uncertainty-tested
usefulness. No diagnostic threshold was tuned on held-out outcomes. Scoring does
not yet compare separate HVP or two-minibatch acquisition costs on neural training.

Compute gradient energy / prospective bias-corrected v sum, cosine with prospective
bias-corrected m, relative sqrt(v) drift, previous-gradient cosine and independent
gradient cosine when supplied. Prospective moments use current g and existing state,
before the parameter step. State is never changed by the monitor. Grad-norm is a
required competing score. A global scalar can miss coordinate-localized staleness.

Scores have no post-update loss or displacement inputs. Labels are f_t before/after
at the same t. Directional curvature logged after displacement is an outcome,
not a predictor. Optional pre-step HVP is available separately. AdamW+monitor-only
uses exactly torch.optim.AdamW and can be trajectory-equality tested.

Offline evaluator reports AP/AUROC and fixed development-threshold confusion metrics,
with positive/negative counts and nulls for undefined metrics. Lead times shift labels
forward *within each run*. Histories must not cross seed/regime boundaries. It does
not train a gate or certify predictive usefulness. False alarm costs include extra
geometry refreshes, not just classification error. No prediction benefit is claimed.
