# Robust Learning Research — Path B discovery

Status: **research-discovery foundation, not a novel method or a completed 7-day sprint**.

This is a separate workspace on the research/robust-learning-discovery branch of Adam_Research. The trustgeoopt directory and its four completed research sprints remain untouched.

**Question to investigate:** When class imbalance, data difficulty, and label corruption interact, which known sample selectors incorrectly reject clean hard/tail examples, and does an inexpensive *additional* signal still help after recent strong baselines?

**Important prior art:** Pan et al. AAAI 2025 already track confidence trends; Pan et al. CVPR 2026 already debias class/instance selection; RoLT (Frontiers of Computer Science 2026) already addresses long-tailed noisy labels; Deng et al. (2024) already target hard clean vs noisy examples. Do NOT present these problem statements, class-wise filtering, confidence history, or similarity scoring as new.

## Starter experiment

The toy data generator exposes the observed label to training and keeps clean labels/corruption flags inside an evaluation-only oracle. The three methods in the smoke script are basic controls: CE, global small-loss, and per-class small-loss. **They are NOT faithful Co-teaching, DivideMix, AAAI 2025, or CVPR 2026 implementations.**

Install Python, numpy, torch, pytest; from this directory run:

    python -m pip install -r requirements.txt
    PYTHONPATH=src python -m pytest -q tests
    python scripts/toy_smoke.py

On Windows PowerShell, set the source path via:

    $env:PYTHONPATH = "src"; python -m pytest -q tests

See CODEX_DISCOVERY_TASK.md for the gated seven-day discovery workflow, research/protocol.md for data leakage and evaluation controls, and research/literature_matrix.csv for the initial verified source list.

## Research gates

1. Audit and reproduce strong relevant prior work; do not rely on toy controls alone.
2. Demonstrate a failure across independent seeds, noise families, difficulty strata, and class priors.
3. Rule out basic class-conditional thresholds, calibrated loss, and the specific improvements of the closest papers.
4. Only then test one additional low-cost statistic as a hypothesis; no new method is presumed.
5. Stop or pivot when prior work closes the gap or fair evaluations negate it.
