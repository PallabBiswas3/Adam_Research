# Day-zero status (2026-10-12)

- Working branch: research/robust-learning-discovery (based on Sprint 4). TrustGeoOpt is unchanged.
- Initial verified primary-source map: eight papers; this is NOT the required full systematic literature audit.
- Controlled toy generator: three Gaussian-mixture classes, long-tail counts (600,160,60), sampled corruption, known Bayes ambiguity available in sealed evaluation oracle only.
- Implemented sanity controls: CE, global small-loss, and classwise small-loss. These are intentionally simple demonstrations, **not** Co-teaching/DivideMix/DSS reproductions.
- Local CPU check: synthetic unit tests 2 passed. One seed of toy smoke was run; numbers below are illustrative, cannot support a claim.

Toy seed 0, 18 epochs, 70% keep fraction, observed-label training:
- CE: clean-hard retained 103/103; clean-tail 52/52; noisy retained 80/80 (no filtering).
- Global small loss: clean-hard retained 8/103; clean-tail 0/52; noisy retained 6/80.
- Classwise small loss: clean-hard retained 48/103; clean-tail 47/52; noisy retained 19/80.

The classwise control improves minority retention but also retains more corrupted examples. This tradeoff is a well-established research topic and no novelty conclusion follows.

**Still to do via Codex:** read full related algorithms, faithful baselines, multiple seeds, held-out tasks, class-conditional/instance-dependent noise controls, fair tuning and runtime audit. Do NOT claim the 7-day sprint is complete.

## Bounded pass update (2026-10-12; day-zero record above retained)

Completed a targeted 24-paper audit, controlled clean validation/test data,
independent corruption RNGs, three noise families and balanced/long-tail priors.
All 245 short CPU attempts succeeded, including 200 evaluations with five fresh
seeds and two held-out task/overlap configurations. Learning rates were chosen
only on independent development validation. Runtime was 92.30 experiment seconds.
19 automated tests passed; source/config/data checks and deterministic evaluation
replay passed; all 11,242 TrustGeoOpt files remained unchanged.

Known classwise filtering improves suite-average clean test macro-F1 relative to
CE; two equation-level paper adaptations do not. Negative findings, selection
precision/retention tradeoffs, calibration, compute and descriptive seed-block
intervals are retained. [Failure study](failure_study.md) has full tables.

The faithful published benchmark gate is **not completed**. Do not call the
adaptations official reproductions or infer a new method's necessity from their
results. Stop before method invention; [decision](decision.md) proposes a bounded,
reviewed faithful-comparator sprint. The audit shows direct recent prior art for
the original premise, including DSS's imbalance-aware extension. No novelty or
seven-calendar-day completion claim is made.
