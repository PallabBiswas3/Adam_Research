# Hypothesis registry — explicitly preliminary

H0 (null): after class-aware calibration and strong prior-art baselines, there is no practically material gap in selecting hard-clean, tail-clean and mislabeled examples at comparable compute and matched noise precision.

H1 (diagnostic / already known): global small-loss selection can reject hard-clean and correctly labeled minority samples disproportionately. Expected to be reproduced, not publishable as new.

H2 (open only after comparator audit): in a selected structured-noise/imbalance setting, a specific *published* selector is miscalibrated for hard-clean examples even with validation-only tuning. Identify a mechanistic failure and characterize its domain. Reject if DSS/RoLT/AAAI-2025/SLRLNL address it.

H3 (conditional): a lightweight incremental statistic provides **out-of-sample** information after conditioning on confidence trajectories, classwise loss ranks, and feature-neighborhood information. Cost and prior art must be checked before proposing anything new.

H4 (downstream): preserving hard-clean data increases held-out balanced accuracy / macro-F1, not merely apparent selection recall. Evaluate with matched noisy-label precision and fair compute.

Must report negative evidence, intervals and seeds. No theoretical innovation is claimed.

## Evidence update, 2026-10-12

H1: descriptively reproduced across the bounded toy controls, with precision/
retention tradeoffs. H4: a known classwise control improves suite-average F1,
but no matched-precision training intervention establishes causality. H0 is not
accepted or rejected; H2/H3 remain untested because two faithful strong
comparator reproductions are unavailable. Direct 2024-2026 prior art invalidates
claims that the basic problem or standard mechanisms are novel. No new statistic
was proposed or evaluated. See decision.md for the stop and conditional next step.
