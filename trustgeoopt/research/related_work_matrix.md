# Explicit novelty-overlap matrix

Classification applies to mechanisms, not to papers' quality or completeness.

| Candidate component/claim | Closest checked prior art | Classification | Decision |
| --- | --- | --- | --- |
| AdamW moments/decay | Adam, AdamW | 1 Already established | reuse trusted implementation |
| distrust gradient residual | AdaBelief | 1 Already established | residual alone cannot support novelty |
| non-diagonal gradient moments | full AdaGrad, Shampoo | 1 Already established | reference comparison required |
| activation-based curvature | K-FAC, EKFAC | 1 Already established | activation moment alone not a Hessian |
| rotate adaptive coordinates | EKFAC, SOAP, LDAdam | 1 Already established | audit state transport/equivalence |
| low-memory gradient sketches | Sketchy, SENG, LDAdam | 1 Already established | no novelty from O(pk) storage |
| adaptive geometric refresh | Purifying Shampoo, FOAM | 1 Already established | H3 is empirical evaluation, not a new idea |
| stable curvature clipping | Sophia; dynamic bounds/RAdam | 2 Partially explored in this proposed combination | rule out step rescaling explanation |
| joint causal lag/alignment/noise score for CNN/MLP correction selection | AdaBelief, staleness/error work, edge-of-stability analysis | 2 Partially explored | exact overlap unresolved |
| equal-compute benchmark separating noise shift from mean/curvature shift | related dynamics studies | 3 Plausibly underexplored *study design* | no verified gap; broaden audit |
| low-rank addition improves all conditioning | isotropic analytical counterexample | 4 Unsupported, false generally | explicitly reject |
| gradient covariance equals Hessian/Fisher | geometry definitions | 4 Unsupported, false generally | distinguish matrices |
| q detects harmful updates independent of noise | non-identifiability counterexample | 4 Unsupported generally | evaluate under narrow assumptions |
| universal AdamW improvement or publication-ready novelty | no supporting evidence | 4 Unsupported | do not claim |

## Optimizer-specific overlap

| Baseline | Moment adaptation | Off-diagonal representation | Staleness/safeguard relevance | Implementation here |
| --- | --- | --- | --- | --- |
| AdamW | diagonal moments | none | base comparator | PyTorch |
| AMSGrad | maximum moment history | none | known nonconvergence remedy | PyTorch Adam AMSGrad, wd=0 pilots |
| AdaBelief | residual moments | none | gradient disagreement scaling | published unrectified reference + decoupled decay |
| Sophia | curvature diagonal | none | intermittent estimation/clipping | review only |
| Shampoo | cumulative tensor factors in reference | per-axis Gram factors | matrix-root cost; periodic refresh | small capped reference |
| SOAP | Adam in structured basis | Kronecker eigenbasis | sparse basis updates/fresh scales | review only |
| K-FAC/EKFAC | Fisher factors/basis variances | layer Kronecker blocks | damping and partial updates | review only |
| Sketchy/SENG/LDAdam | sketches/projections | low-dimensional factors | rank/transport error | review only |
| Purifying Shampoo/FOAM | eigenvalue/error correction | Shampoo factors | directly adaptive refresh/damping | review only |

All source links and extraction status are in paper_database.csv and literature_review.md.
Class 3 means a candidate direction to investigate; it is not evidence of originality.
