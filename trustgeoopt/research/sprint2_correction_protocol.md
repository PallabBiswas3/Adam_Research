# Fixed correction protocol, written after Tasks A/B and before correction trials

Evidence: baseline evaluation finds a stochastic tail-loss gap, not persistent
deterministic AdamW failure. Among reliability development harmful updates, 31.1%
have an uphill proposed direction against the observed gradient. Geometry plus
magnitude predicts far more than norm alone; alignment has only a small and
regime-dependent conditional gain, while history fails unseen transfer. Thus test
three established first-order remedies without implementing geometric preconditioning.

Fixed definitions (no threshold/schedule tuning):

1. **Step clip**: after ordinary AdamW updates its moments, clip the total parameter
   displacement to Euclidean norm .1; keep moment state. This clips the step,
   not the gradient, and never consults loss or curvature.
2. **Momentum reset**: before updating, predict the actual AdamW displacement.
   If observed gradient dot proposed displacement >0, zero only the first-moment
   buffer, keeping second moment and global step count, then take the AdamW step.
   No noise-free gradient or future outcome enters the gate. This is an AdamW
   adaptation of established gradient-alignment restart, not a reproduction of
   accelerated-gradient restart or a new optimizer claim.
3. **LR safeguard**: learning rate = base_lr / sqrt(1+t/200), with t starting at 0.
   This is a fixed conservative schedule, not a learned reliability gate and not
   an exact AdaBound implementation. A changing optimum may need a different rule.

Same AdamW six learning rates .001,.003,.01,.03,.1,.3; each mechanism receives
60 tuning runs (five regimes × two development seeds × six rates), and 25 selected
evaluation runs (five regimes × five new seeds), all 1,200 steps. Same objective,
initialization/noise pairing and final-20% metric as Task A. Each mechanism gets
the same number of trials as ordinary AdamW, but its fixed policy is an additional
researcher's choice: this is a targeted exploratory comparison, not a claim of
equal exhaustive tuning over richer mechanism families. Already inspected
evaluation evidence motivated the family; subsequent scores are not independent
confirmatory evidence of choosing that family. No further policy tuning is allowed.

All corrections have O(p) work/state and no extra gradient/HVP. Measure a separate
warm CPU timing experiment on 16 and 16,384 parameters, alternating methods,
fixed precomputed gradients, nine repeat blocks. Include observer/reset/copy costs;
exclude synthetic true-Hessian diagnostics, imports, CSV and gradient computation.
Report observed time ratios and dispersion, not extrapolated GPU overhead.
