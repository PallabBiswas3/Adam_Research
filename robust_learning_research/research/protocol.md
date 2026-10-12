# Frozen discovery protocol (initial; no main experimental campaign launched)

Date: 2026-10-12. A discovery plan is not a preregistered final algorithm trial.

## Core question

Can any real failure of modern noisy-label training under jointly hard examples, minority groups, and structured label noise survive comparison to recent debiased sample-selection methods?

The naive proposition "small loss wrongly drops hard clean examples" is already established by prior papers. It is a **baseline behavior**, not novelty.

## Day-by-day gates (seven working days of interactive Codex work, not an unattended scheduled job)

- Day 1: inspect publications, verified algorithms and release licenses. Record at least 15-25 closely relevant papers with extraction depth and a specific overlap table. Include AAAI 2025 confidence tracking, CVPR 2026 DSS, RoLT, and SLRLNL.
- Day 2: audit experimental definitions and build independent corruption-mask and Bayes-hardness checks. Keep oracle clean labels confined to metrics and controls. Fix reproducible data/task split and noise random streams.
- Day 3: reproduce CE, global small-loss and class-wise small-loss as *controls*. Confirm the observed bias, estimate uncertainty across at least 5 seeds, and record counterexamples.
- Days 4-5: run two **faithful**, appropriately scoped recent published baselines (prefer a verified Co-teaching and either the official DSS or confidence-tracking implementation; choose based on code/license, resource feasibility and precise training protocol). Document deviations; do not call approximations reproductions.
- Day 6: test whether any residual weakness survives near-boundary, class-conditional, instance-dependent and long-tail variations; stratify by true-label ambiguity and actual corruption. Use held-out task/noise seeds not selected for method design.
- Day 7: write go/no-go decision, residual hypothesis, cost audit, novelty uncertainty, failure analysis, and next 14-day plan. Explicitly permit ending with "no sufficient research opening."

## Experimental validity

- The observed noisy label is the **only training label**. Ground-truth clean labels and flip flags may be used only for synthetic evaluation, never the selection criterion, target correction, tuning, or early stopping.
- Generate clean validation/test datasets independently from training; no synthetic oracle is allowed for hyperparameter selection beyond explicitly labeled, diagnostic-only oracle experiments.
- Difficulty in the Gaussian toy uses the known generative Bayes posterior on TRUE classes, sealed in the evaluation oracle. For CIFAR/Fashion-like images, predeclare a clean-reference or corruption-independent difficulty definition and disclose its limits. Do not define hard examples based solely on the method's own training loss.
- Vary balanced/long-tailed priors, symmetric/class-conditional/instance-dependent noise, and easy/hard strata; maintain a record of exact flip masks.
- Use matched optimizer, backbone, epochs, data augmentation and tuning budgets as practical; where official methods require distinct architectures/costs, report them and compare both total compute and accuracy.
- Primary diagnostics: **hard-clean retention at matched noise-precision**, clean tail-class retention, class-wise sample-selection precision/recall, macro-F1 and balanced accuracy on held-out clean data, wall-clock runtime and extra memory.
- Report paired run-level confidence intervals and full seeds. Selection must not optimize for oracle labels. Avoid comparing at unmatched acceptance fractions without a matched-operating-point analysis.
- Four-cell challenge matrix: (balanced vs long-tail) x (symmetric vs instance-dependent/class-conditional noise); prespecify at least one boundary-difficulty stressor.
- Predefine a practical go rule: residual shortfall of an established method on hard-clean/tail retention **at matched noise-precision and similar compute**, replicated across at least two distinct data regimes and five seeds, and not already solved by a verified recent comparator.
- Limit exploratory CPU studies initially; no cloud, paid GPU, heavyweight unapproved dependencies, or large training runs. Request authorization before scaling.

## Outputs

research/prior_art_audit.md, research/failure_study.md, research/decision.md, experiment configs, exact provenance and run-level CSV, faithful baseline checks, and an honest statement about novelty.

No claim of a new loss function, selector, optimizer, theorem, or publishable paper is authorized by this initial scaffold.
