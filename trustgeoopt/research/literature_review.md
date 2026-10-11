# Initial scoping review

Date: 2026-10-12. Status: incomplete for publication. This is a primary-source
map and selected full-text mechanism review, not a systematic review or verified
novelty claim. The database contains 31 research-paper records and one teaching-note
source. Its extraction_level fields separate metadata/abstract reading from selected
algorithm/theorem inspection; unextracted assumptions and benchmarks remain explicit.
Verified venue years are separate from first arXiv posting years where they differ.

## Question, sources and screening

Can causal gradient/state diagnostics select low-cost geometry that improves loss
progress per total compute for MLP/CNN supervised learning? Compare tuned AdamW,
momentum and structured optimizers. Follow [search_protocol.md](search_protocol.md).
Primary sources: arXiv, official PMLR/NeurIPS/JMLR pages, original course notes,
publisher records and OpenReview PDFs when accessible. Google Scholar was not
queried directly; no completeness claim follows. Query log is [search_log.json](search_log.json).
Exclude unrelated arXiv identifiers 1904.01653 (American option prices) and 1902.06720
(wide-network dynamics) from the optimizer database; they were initial retrieval
mismatches. Deduplicate canonical IDs; repeated reads do not create extra records.
OpenReview challenged direct access to SOAP/AMSGrad; arXiv supplied usable records.
CASPR's official PDF was surfaced by the search provider; full retrieval audit remains.

## A. Adaptive first-order methods

Momentum adds historical direction, not curvature. SGD is a crucial negative/control
remedy for instability; [Polyak's primary record](https://www.mathnet.ru/eng/zvmmf7713)
anchors the momentum family. [AdaGrad](https://jmlr.org/papers/v12/duchi11a.html)
already develops both diagonal and matrix geometry from past gradients. RMSProp's
EMA formula is available in [original course slides](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf);
these are teaching notes, with publication year unverified here.

[Adam](https://arxiv.org/abs/1412.6980) combines bias-corrected moments; its diagonal
denominator is a second moment, not a Hessian. The original theoretical analysis
cannot serve as an unconditional guarantee: [AMSGrad work](https://arxiv.org/abs/1904.09237)
gives an online convex counterexample and long-memory remedy. That counterexample
has not yet been reproduced by this repository's finite-step quadratic pilots.
[AdamW](https://arxiv.org/abs/1711.05101) separates decay from adaptive moments.

[AdaBelief](https://arxiv.org/html/2010.07468v5) already turns disagreement between
gradient and momentum into residual-based scaling. Its Algorithm 2 injects epsilon
into s each step as well as the update denominator; the local reference now includes
that detail, with explicitly decoupled decay and no rectification.
[RAdam](https://arxiv.org/abs/1908.03265) addresses early adaptive-rate variance;
[Yogi](https://proceedings.neurips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)
controls effective-rate changes and analyzes stochastic stationarity.
[AdaBound](https://arxiv.org/abs/1902.09843) bounds coordinate rates toward SGD.
These are competing explanations for a gate's apparent gains: scaling or clipping
may suffice without learning off-diagonal geometry.

## B. Curvature and structured metrics

Classical Newton solves H d=g, requiring positive definiteness or a trust-region/
damped modification for descent. BFGS uses secant pairs s,y; inverse-metric update
B_next=(I-rho s y^T)B(I-rho y s^T)+rho s s^T, rho=1/(y^T s), preserves SPD
when y^Ts>0. Full BFGS stores O(p^2); [L-BFGS](https://link.springer.com/article/10.1007/BF01589116)
retains a bounded number of pairs, typically O(pk), but line searches add gradient
calls and stochastic secants may be unreliable. Original Newton/BFGS source review
is a remaining literature gap; the equations here are mathematical background.

[Natural gradient](https://doi.org/10.1162/089976698300017746) uses a distribution
metric. [K-FAC](https://proceedings.mlr.press/v37/martens15.html) approximates layer
Fisher blocks via activation and output-gradient factors. Activation statistics alone
are insufficient. [EKFAC](https://arxiv.org/abs/1806.03884) corrects variance within a
Kronecker eigenbasis, explicitly separating cheaper variance updates from basis work.

[Shampoo's matrix algorithm](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf)
accumulates GG^T/G^TG and applies quarter powers on each side. The implemented
small reference follows this rule, unfolds larger tensors and uses diagonal AdaGrad
for vectors. It is ungrafted, cumulative, and capped at factor size 128; it cannot
replace a maintained modern baseline in a publication comparison.
[Sophia](https://arxiv.org/abs/2305.14342) uses periodic diagonal curvature and
clipping; its language-model experiments are outside our empirical scope.
[SOAP](https://arxiv.org/html/2409.11321v2) runs adaptive scaling in a structured
eigenbasis. Its equivalence result concerns stated modifications of Shampoo;
it does not establish that arbitrary low-rank Hessian approximations are Adam.

## C. Sketches and low-dimensional statistics

[Oja](https://link.springer.com/article/10.1007/BF00275687),
[streaming-PCA family comparison](https://arxiv.org/abs/1506.01490) and
[2026 averaged streaming PCA](https://arxiv.org/abs/2609.23332) provide existing
subspace estimators. Their distribution/eigengap assumptions must be examined before
using them on a changing gradient stream. [SENG](https://arxiv.org/abs/2006.05924)
uses sketchy empirical natural gradients and includes CNN evidence.
[Sketchy](https://arxiv.org/abs/2302.03764) establishes Frequent Directions resource/
regret trade-offs. Our rolling SVD is a numerical control, not an FD implementation.
[LDAdam](https://arxiv.org/abs/2410.16103) transports optimizer statistics between
subspaces and includes error feedback. Simple basis replacement must be checked
against this mechanism. [CASPR](https://openreview.net/pdf?id=8j9hz8DVi8) is another
axis-combination alternative; [ASGO](https://arxiv.org/abs/2503.20762) and
[unified structured analysis](https://arxiv.org/abs/2503.10537) challenge the premise
that a fuller metric must always outperform a cheaper structure.

## D. Reliability, staleness, and recent work

[Purifying Shampoo](https://arxiv.org/html/2506.03595v2) explicitly corrects eigenvalue
staleness and adaptively refreshes eigenbases using approximation error. Adaptive
refresh is therefore already investigated. [FOAM](https://arxiv.org/abs/2606.02365),
a 2026 preprint, jointly adapts damping and eigendecomposition frequency from a
staleness proxy; independent replication and theorem audit remain pending.

[Adaptive edge-of-stability work](https://arxiv.org/abs/2207.14484), revised in 2024,
[rod flow](https://arxiv.org/abs/2605.06821) and
[active-curvature feedback](https://arxiv.org/abs/2608.20638v2) caution that gradient
reversal may be part of adaptive dynamics. The last paper's title changed between
v1 and v2; the database uses the version checked on 2026-10-01.
[Heavy-tail AdamW open problem](https://arxiv.org/abs/2606.23676) is not a general
convergence theorem. Gaussian synthetic noise does not test heavy-tail claims.

## Synthesis and outstanding extraction

High confidence: adaptive moments, residual scaling, structured eigenbases, sketches,
and error-driven refresh have substantial prior art. Low confidence: this particular
joint diagnostic could support a cheaper correction in conventional networks.
No search absence establishes a gap. Matrix identity or feature covariance alone is
not a contribution. The most defensible next step is an honest diagnostic benchmark
and negative controls, followed by a correction only if it beats an existing remedy.

Remaining: original SGD/RMSProp archival attribution and Newton/BFGS originals;
full assumptions/experimental tables for abstract-only records; full 2026 coverage
through 12 October; maintained Shampoo/SOAP/K-FAC integration; papers specifically
about minibatch-consistency gating; scholarly citation chaining and backward/forward
snowballing. Peer-reviewed status is not inferred from an arXiv abstract. No numerical
performance result from the literature is reproduced or endorsed by this initial review.
