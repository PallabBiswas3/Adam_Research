# Codex execution task — Path B research discovery

You are a careful ML researcher and PyTorch engineer. Continue an existing GitHub branch rather than starting from scratch.

Repository: PallabBiswas3/Adam_Research
Branch: research/robust-learning-discovery
Working folder: robust_learning_research

The branch inherits the historical trustgeoopt tree, which MUST remain unchanged; only modify robust_learning_research unless explicitly instructed. The project is separate, and not a rename or rewrite of TrustGeoOpt.

## Context

The prior optimizer project completed four evidence-gathering sprints and did not establish publishable optimizer novelty. Path B explores learning with corrupted labels, difficult but correct examples and class imbalance. This problem has **strong existing solutions**: SLRLNL (2024), AAAI confidence tracking (2025), RoLT (2026), CVPR DSS (2026), plus Co-teaching, DivideMix, ELR and PASS. Do not treat the basic problem or common mechanisms as novel.

Read README.md, research/protocol.md, research/literature_matrix.csv, hypotheses.md, discovery_status.md, and code/tests before writing anything.

## Seven-day iterative workflow

Day 1: Verify full-text methods and at least 15-25 closely related primary papers. Record date/venue/license, algorithm equations, benchmarks, failure modes, compute, official implementation links, and conflicts/duplicates. Explicitly compare the 2024/2025/2026 methods. Mark incomplete extraction.

Day 2: Inspect the toy data generator and tests. Build clear-data validation held out from observed-label training. Add controlled symmetric, class-conditional, and instance-dependent noise with frozen flip masks, balanced/long-tail priors, and predeclared difficulty. Ensure oracle labels NEVER leak into selectors or validation-driven tuning. Write unit tests first.

Day 3: Run CE, global small-loss and classwise small-loss controls using matched seeds/hyperparameters; record at least five seeds, precision-recall operating curves, hard-clean retention, tail-clean retention and clean held-out macro-F1. The current one-seed toy result is only a smoke check.

Days 4-5: Reproduce at least two strong relevant published baselines faithfully. Preferred: Co-teaching (with two networks, cross selection and forget rate) and one of CVPR-2026 DSS or AAAI-2025 confidence tracking (official code/mechanism and fidelity permitting). DivideMix or RoLT can be used if implementations and compute support accurate reproduction. Reuse official code only with licenses and explicit source hashes. An approximation must NOT be called a reproduction. Equalize evaluation protocol and report dissimilar training costs.

Day 6: Analyze failures on held-out noise/task seeds and minority/hardness strata. Compare at matched noisy-label precision/retention and report run-level uncertainty, calibration, failure modes and compute. Seek residual failure not already handled by the strongest available comparator. Do not train on oracle clean labels or tune using sealed evaluations.

Day 7: Write a truthful go/no-go report: reproducible gap (or none), exact closest prior art, proposed falsifiable next hypothesis, expected math/compute contribution, and a bounded next sprint. Run all tests, document errors and preserve negative results.

## Deliverables

- research/prior_art_audit.md (with verification state and at least 15-25 relevant papers)
- research/method_fidelity.md
- research/failure_study.md
- research/decision.md
- reproducible configs, source SHA records, run-level tables and plots
- tests for corruption-label fidelity, oracle isolation, data splits, baseline equivalence, seed determinism and summary metrics
- README updates reflecting only completed work

## Constraints and stop conditions

No invented novelty, fake experiment results, unverified proofs, cherry-picked seeds or test-set model selection. No new optimizer or multi-module method. No giant GPU/paid training. CPU smoke studies first; ask before significant GPU expense or dependency upgrades. Bound the initial unattended command to short runs; do not treat this prompt as authorization to work for seven days without user oversight.

Stop and state why if strong recent prior art already addresses every defensible gap, or faithful baselines are not feasible. A negative decision is a valid outcome.

## Repository workflow

Remain on research/robust-learning-discovery; work in reviewable commits or a separate child branch if needed. Preserve TrustGeoOpt completely; never force push. Report commits/tests and push new commits only after checking git diff. Do not merge to main.

Begin with the prior-art audit and the toy smoke tests, then follow the gates above.
