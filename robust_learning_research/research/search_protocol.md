# Discovery literature and execution protocol

Date/cutoff: 2026-10-12. Mechanism-focused scoping review, not an exhaustive
systematic or priority audit. Read the eight-record foundation first. Include
20–25 primary papers on noisy labels, hard-clean selection and noisy long-tail
learning; search official proceedings, arXiv, publishers and linked official
repositories. Inspect algorithms where accessible and mark inaccessible/full
benchmark/theorem details incomplete. Distinguish posting, online and volume
years; deduplicate by DOI, arXiv identifier and normalized title. Record code
license separately from article license; absent code licensing is not inferred
from public visibility. No copying unlicensed source into the project.

Queries: Co-teaching, Co-teaching+, DivideMix, ELR, MentorNet, JoCoR, forward
loss correction, GCE, SCE, FINE, UniCon, ITEM, confidence tracking AAAI2025,
debiased sample selection CVPR2026, SLRLNL, RoLT, PASS, RCAL, long-tailed noisy
labels, confident learning, class-balanced loss, LDAM and logit adjustment.
Exclude review-only and unrelated acronym/title matches from the primary count.
Recent exact title/source conflicts and retrieval failures are logged.

Initial independent work is a short CPU campaign, not seven calendar days of
autonomous work. Hard cap: 300 experiment seconds / 260 training attempts;
one torch thread, no GPU/cloud/dependency upgrades. A pilot command runs only
five short trials before expansion. Earlier toy smoke and preservation hashes
are recorded separately. Every command works under robust_learning_research;
TrustGeoOpt is never imported, tested, edited or used as a cache/data source.

Gates: after data/tests/control validation, inspect feasibility of two faithful
published benchmark baselines. Mechanism-only toy adaptations are explicitly
not benchmark reproductions. If those faithful baselines are infeasible under
resources/source fidelity, stop method discovery, record an incomplete
comparator gate and a no-go for novelty. Do not infer that prior art solves
every possible gap just because broad problem statements overlap.
