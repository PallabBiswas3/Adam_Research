# Sprint 3 targeted mechanism review

Date: 2026-10-12. Exploratory scoping update, not an exhaustive systematic
review. Searches targeted stochastic stationary error, structured coordinates,
low-rank history, and basis dependence. Primary full text was inspected at the
locations below; empirical claims from these papers are not our benchmark
results. Search log and paper database preserve earlier records.

| Primary source | Extracted mechanism and limits | Sprint 3 implication |
| --- | --- | --- |
| [Mandt, Hoffman & Blei, JMLR 2017](https://jmlr.org/papers/volume18/17-214/17-214.pdf), §§3 and 5 | Constant-rate diffusion with local quadratic and approximately constant covariance assumptions; Fisher/Hessian identification requires further statistical assumptions. | Our discrete frozen-B derivation separates covariance from curvature; it is not a theorem about nonlinear Adam. |
| [SOAP](https://arxiv.org/html/2409.11321v2), §§3–4, Algorithms 1–3 | Adaptive moments in a Shampoo-derived eigenbasis; exact equivalence to factored scaling is stated for idealized dataset moments, not every practical EMA variant. | Structured adaptive coordinates already exist. Authors' preliminary implementation is pinned and licensed; production maintenance is not claimed. |
| [Sketchy](https://arxiv.org/pdf/2302.03764), §§4.1–4.3, Theorem 3 | Frequent Directions sketches with escaped-mass compensation; online-convex regret bounds depend on gradient second moments and domain/iterate radius. | Low-rank history and diagonal remainder compensation are established; the sketch targets gradient moments, not necessarily a Hessian. |
| [LDAdam](https://arxiv.org/html/2410.16103v2), §§3–4, Assumptions 1–3 | Projection-aware state transitions and generalized error feedback. Theory assumes smooth lower-bounded objectives, unbiased bounded gradients and bounded variance. | A changing-subspace adaptive update needs state transport/error accounting; unbounded Gaussian noise here does not satisfy its bounded-gradient assumption. |
| [GaLore](https://arxiv.org/abs/2403.03507), abstract and authors' repository | Low-rank gradient projection for optimizer-state memory reduction. Full theorem conditions not audited in this sprint. | A gradient sketch alone is not a distinct contribution; no convergence guarantee is imported. |
| [Singh, 2026 preprint](https://arxiv.org/html/2608.05136v1), §4, Proposition 4.4 and Remark 4.7 | Factored-model gauge symmetry differs from ambient quadratic rotation; coordinate-wise scaling can break the symmetry. Common-scalar flow transfer is narrower than stateful Adam; equivariance alone does not guarantee low-rank recovery. | Related basis-dependence work further narrows novelty. Its solution-selection findings do not prove a speed advantage for our unfactored quadratics/MLPs. |
| [VectorAdam, NeurIPS 2022](https://www.dgp.toronto.edu/~zling/vector-adam/vectoradam.pdf), Algorithm 1 and Proposition 1 | Vector-level scalar second moment preserves rotations; proof is under a rotation-invariant objective with correspondingly transformed vector state. | Known inexpensive alternative to coordinate-wise scaling; equivariance does not infer Hessian eigenvalues. Not benchmarked here. |
| [Zhang et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/a6ffe4ec7aaa6e7e7d5bd28e14827d3a-Paper-Conference.pdf), §2 and Proposition 1 | Explicit ambient parameter-space rotation protocol; SGD momentum is equivariant, Adam is basis sensitive; broader scopes can have different practical effects. | Our matched quadratic rotation study is verification of an established distinction, not a new failure mechanism by itself. |

Implementation feasibility was checked against the current
[Meta optimizer package manifest](https://github.com/facebookresearch/optimizers/blob/main/pyproject.toml):
Python >=3.12 and torch >=2.8. Installed versions are 3.11.4 and 2.1.2 CPU.
No environment upgrade was attempted. The separately pinned
[SOAP authors' repository](https://github.com/nikhilvyas/SOAP/tree/a1e553530fde97d0e6b307d7c82ac6d38b072340)
calls its code preliminary; vendor provenance records commit, source URLs and
SHA256 hashes. The MIT license is retained. Its float32 factor eigensolves/QR,
first-call initialization and skipped oversized factors remain unmodified and
are disclosed in our protocol.

Negative novelty conclusion: covariance sketching, adaptive coordinates,
state transport, refresh scheduling and diagonal safeguards are occupied
components. A contribution would require a verified residual mechanism and
new causal information/cost tradeoff, with a comparison to appropriate existing
methods. No “first”, publication venue, theorem transfer or optimizer novelty
claim is made for this project.

Retrieval mismatches were excluded: arXiv 2309.08520 concerns sparse foundation
model scaling; 2405.16308 concerns rational approximation. Neither supports an
optimizer claim. GaLore HTML retrieval failed; its abstract is identified as
the extraction limit. Meta requirements are current software information,
not a guarantee that older compatible releases cannot be evaluated later.
