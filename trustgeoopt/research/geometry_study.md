# Geometry study

Status: standalone primitives, no geometric optimizer claim. Gate D open.

## Recorded primitive verification

Dense toy equality tests passed, including dependent factor columns. A 40-row CPU
microbenchmark is saved under `results/geometry/raw`; p=128/1024, k=2/4, window=2k,
ten repetitions for each setting. Mean relative solve residuals are about 2.4e-15
to 2.0e-14. Refresh/solve timings and retained storage bytes are measured per row.
These traced, instrumented sub-millisecond timings are exploratory; they do not
demonstrate an end-to-end optimizer improvement or justify a correction scale.

| Estimator | Shape/storage | Cost per refresh | Meaning/error | Adam interaction/overlap |
| --- | --- | --- | --- | --- |
| Rolling gradient history SVD (implemented) | history p by h; retained factor p by k, h=O(k) | O(p h^2+h^3) for p>=h | exact leading window singular modes; discarded squared singular tail measures approximation; moving-window bias | covariance units differ from sqrt(v); overlaps Sketchy/LDAdam families |
| Oja streaming PCA (review only) | p by k | O(pk^2) reorthogonalization | stochastic eigengap assumptions; drift tracking unproved here | established streaming method, needs rate tuning |
| Layer activation moment (review only) | a by a per layer | O(B a^2), spectral O(a^3) | input factor only; no parameter-curvature equivalence | K-FAC requires output-gradient factor too |
| Approximate Gauss-Newton HVP sketch (interface only) | p by k | k products/backwards plus O(pk^2) | loss-convex PSD GN; stochastic/rank error | potentially costlier than first-order step |
| Random directions (control) | p by k | O(pk^2) QR | no data-derived geometry | separates regularization/rescaling effects |

The rolling history is not Frequent Directions and has no FD error guarantee.
Uncentered second moment and centered covariance are separate selectable quantities.
No neural dense Hessian or p by p gradient covariance is stored by this module.
Woodbury uses Cholesky of I+F^T B^{-1}F, avoiding C^{-1} for rank-deficient PSD C.
Positive diagonal, finite inputs and nonnegative strength/damping are enforced.
Dense solve is a tiny test-only reference. Fail rather than silently changing damping.

Required next evidence: validate scaling from gradient-squared statistics to an
Adam-compatible metric; benchmark p/rank/window versus step cost; examine leading
noise directions and negative isotropic controls. A cached low-rank basis may be
cheap to apply but expensive to estimate; timing must include history copies/SVD.
