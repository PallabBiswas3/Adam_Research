# Sprint 4 mechanism-level literature audit

Scoping review searched 2026-10-12, cutoff 2026-10-12. No exhaustive or
priority claim. Primary sources below were opened; locations distinguish
mechanism extraction from abstract-only screening. Empirical results reported
by other authors are not project measurements. Existing Sprint 3 source and
license pins remain unchanged. No new dependency is needed.

## Search and screening

Protocol: [frozen protocol](sprint4_protocol.md). Queries via web search with
arXiv/official proceedings/author PDFs: `stochastic Hessian vector estimation`,
`Deep learning Hessian free optimization Martens`, `K-FAC EKFAC eigenvalue
corrected`, `gradient noise covariance two minibatches differences`,
`Preconditioned Stochastic Gradient Descent Li`, `Stochastic Quasi-Newton Byrd
Hansen Nocedal Singer`, `empirical Fisher Limitations`, `noise curvature
preconditioning 2025 2026`, `Davis Kahan statisticians`, `gradient noise scale
McCandlish`, `Oja streaming PCA`, `Stochastic Variance Reduced Gradient`,
`SENG stochastic Newton`, `Purifying Shampoo optimizer`, `FOAM optimizer`.
Direct source lookups extend earlier SOAP/Sketchy work. Date restrictions checked
against papers' version metadata rather than search-engine crawl dates.
Cosmetic shampoo results, nonprimary explanations and unrelated matrix
factorization acronym matches excluded. FOAM has two distinct optimizer papers:
folded states (2025) and Shampoo staleness/damping (2026); they are not conflated.
An initial HVP lookup
at arXiv:1409.2325 resolved to an unrelated algebraic-geometry paper and was
excluded; the verified Pearlmutter author PDF replaces it. Search yields no
license to assert there is no further related work through October 2026.

| Source and inspected location | Identified information, assumptions | Cost model and failure mechanism |
| --- | --- | --- |
| [Pearlmutter, 1994](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf), HVP construction; [verified DOI](https://doi.org/10.1162/neco.1994.6.1.147) | Directional differentiation computes Hessian-vector action without constructing H. Differentiable loss and fixed data/parameters. | Constant multiple of differentiation cost per vector, graph storage O(p+activations). Non-smooth boundaries and stochastic batch Hessians require care. |
| [Martens, ICML 2010](https://icml.cc/Conferences/2010/papers/458.pdf), §§3–4 | Hessian-free conjugate-gradient solves use curvature products, commonly PSD generalized Gauss–Newton with damping. | Several products per solve and large curvature batches; local approximation and damping affect progress. It is not noise covariance subtraction. |
| [Martens & Grosse, 2015 K-FAC](https://arxiv.org/html/1503.05671), §§2–3, 5–6, 8 | Model Fisher, not automatically observed-label second moment. Layer blocks approximate E[aaᵀ⊗ddᵀ] by E[aaᵀ]⊗E[ddᵀ]; exponential-family logits give model Fisher=GGN. | Factor storage O(a²+d²), factor solves/eigensolves O(a³+d³), refresh amortized. Factorization error, damping and stale factors; labels/model mismatch breaks empirical-Fisher identification. |
| [George et al., 2018 EKFAC](https://arxiv.org/pdf/1806.03884), §3 | Kronecker eigenbasis with empirical diagonal eigenvalue correction. Optimizes Frobenius approximation within that basis, not globally exact eigenspace recovery. | Factors plus parameter-size diagonal; corrected squared projected gradients. Wrong/stale basis remains wrong. |
| [Kunstner, Balles & Hennig, NeurIPS 2019](https://arxiv.org/pdf/1905.12558), §§3–4 | Observed-label empirical Fisher generally differs from model Fisher and GGN; agreement needs additional assumptions. | Per-example gradients can supply moments, but efficient computation does not establish curvature information. Directly rules out equating paired-noise estimates with H. |
| [Gupta et al., 2018 Shampoo](https://arxiv.org/pdf/1802.09568), algorithm and analysis | Tensor-axis gradient second moments and inverse-root scaling under online optimization assumptions. | O(sum axis sizes²) state and cubic factor roots. Tracks gradient moments; arbitrary ambient rotations need not preserve tensor structure. |
| [Vyas et al., SOAP](https://arxiv.org/html/2409.11321v2), §§3–4 | Adam moments in Shampoo-derived factor eigenbasis, plus basis refresh/state transformation. | Factored bases, squared projected gradients and QR/eigh. Curvature is not independently identified by the gradient moment. Our pinned authors' implementation is preliminary, not Meta's maintained distributed package. |
| [Eschenhagen et al., 2025 Purifying Shampoo](https://arxiv.org/pdf/2506.03595), §§3–4, Eq.11 | Decouples factor eigenvalues/eigenbasis, corrects scaling and controls relative off-diagonal error for refresh. | Two factor multiplies for error check plus scheduled spectral work. Staleness and scaling already studied; choosing refresh is not new. |
| [Feinberg et al., 2023 Sketchy](https://arxiv.org/pdf/2302.03764), §§4.1–4.3 | Frequent Directions gradient-moment sketches with escaped mass; online convex regret assumptions. | O(pr) sketch state, thin spectral work. A small sketch can estimate the wrong object for Hessian recovery. |
| [Yang et al., 2021 SENG](https://arxiv.org/pdf/2006.05924), §§2–3 | Regularized empirical-Fisher gradient Gram solves, low-rank layer structure and randomized sketches. | Extra per-sample information, sketch/least-squares costs; observed-label Fisher caveat still applies. This is prior art for low-rank gradient preconditioning. |
| [Byrd et al., SIAM 2016](https://arxiv.org/pdf/1401.7020), §2 and cost §2.1; [published DOI](https://epubs.siam.org/doi/10.1137/140954362) | Periodic subsampled Hessian-vector secants and averaged iterates for stochastic limited-memory BFGS; separates curvature sampling from noisy update differences. | O(pm) state/apply plus periodic curvature batches. Small-batch secants are noisy; sufficient sample size and positive curvature assumptions matter. |
| [Li, 2015 PSGD](https://arxiv.org/pdf/1512.04202), §IV, Eq.31 | Parameter/gradient perturbations optimize E[δgᵀPδg+δθᵀP⁻¹δθ]; balances magnitude and noise rather than merely fitting inverse H. | Additional perturbation evaluations or HVPs, factorized P options. Noise can deliberately damp scaling; noisy independent differences do not magically reveal curvature. |
| [Pooladzandi et al., 2024](https://arxiv.org/pdf/2402.04553), §2 and preconditioner appendix | General Lie-group PSGD factors using curvature products or finite differences. | Factor updates/probes beyond a gradient; parametrization, refresh and numerical stabilization constrain usefulness. Noise-aware curvature preconditioning is established territory. |
| [McCandlish et al., 2018](https://arxiv.org/pdf/1812.06162), Appendix A.1 | Different-batch-size gradient norm comparisons estimate noise trace and squared mean gradient; distinguishes simple and Hessian-weighted noise scale. | Cheap with existing parallel gradients, extra calls otherwise. Scalar noise scale is not an eigenspace or Hessian estimator. Our paired matrix moment is the elementary same-parameter difference identity. |
| [Johnson & Zhang, 2013 SVRG](https://papers.nips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf), algorithm and strong-convex theory | Same-sample gradient differences plus reference full gradient form a control variate. Smooth finite-sum assumptions for original linear rate. | Periodic full gradients, two component evaluations per inner step, O(p) state. Cancellation controls sample variability but nearby-point differences still estimate batch curvature. |
| [Jain et al., 2016 streaming PCA/Oja analysis](https://proceedings.mlr.press/v49/jain16.pdf), algorithm and assumptions | Online covariance leading vectors under statistical/eigengap assumptions. | O(pr) state and online multiplies. Optimizer histories are correlated/nonstationary and their covariance may be noise dominated. Streaming the covariance does not change what it identifies. |
| [Halko, Martinsson & Tropp, 2011](https://arxiv.org/pdf/0909.4061), §§4–5 | Randomized range finder, QR refinement and small projected spectral solve. PSD/spectral decay and adequate probes improve leading-space recovery. | O(p(r+s)) state/workspace and (q+2)(r+s) operator calls here, plus QR O(p(r+s)²). Weak eigengaps, noisy/changing operators and stale geometry defeat recovery. Our standalone implementation is this known mechanism. |
| [Yu, Wang & Samworth, Biometrika 2015](https://arxiv.org/pdf/1405.0680), Theorem2 | Population-gap Davis–Kahan variant with explicit operator/Frobenius perturbation bound. | The bound needs a reference with the desired eigenspace and positive gap. No iid error rate is available just because training steps were sampled. |
| [Vary et al., 2026 preprint](https://arxiv.org/html/2603.11989v1), introduction and theorem assumptions | Curvature/noise mismatch affects effective dimension and multipass stability under its statistical model. | Prior theoretical overlap; its generalization statements are not theorems for our finite adaptive trajectories. |
| [Nayak et al., 2026 preprint](https://arxiv.org/html/2605.20756v1), §§4–5, 8 | Cross-fits gradient/preconditioner microbatches; delta-method/Jackknife inversion correction addresses coupling and inverse bias. | Extra microbatch statistics, inversion correction and sample splitting. No Hessian recovery theorem imported; pilot mechanisms here do not establish an independent contribution. |
| [Wen et al., FOAM 2025 preprint](https://arxiv.org/abs/2512.07112), abstract-only screening | Folded optimizer moment/state compression and residual correction. | Not audited at theorem level, not benchmarked, no guarantee transferred. Distinguish acronym from curvature estimation. |
| [Nam & Ahn, FOAM 2026 preprint](https://arxiv.org/html/2606.02365v1), algorithm/assumption locations | Staleness-oriented error controls Shampoo damping and eigendecomposition frequency; distinct from the folded-state FOAM above. | Selective factor spectral refresh plus error estimates. No guarantee transferred to our estimator or adaptive trajectories; not benchmarked here. |

## Novelty conclusion before experiments

Same-parameter independent differences identify noise covariance under explicit
sampling assumptions. Common-random-number secants, stochastic HVPs, GGN,
randomized subspaces, Kronecker factors and variance-reduction controls are
established. Combining them is not by itself a distinct estimator or optimizer.
Our scope is a reproducible controlled comparison of their information and
cost when history PCA fails. Only a measured residual practical limitation
and a mechanism unavailable from these alternatives could support algorithm
development. No such fact is assumed. Search/extraction are scoped to this
question; the review cannot certify priority or exhaustive novelty.
