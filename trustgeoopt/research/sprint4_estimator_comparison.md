# Sprint 4 estimator comparison

**Known curvature probes recover directions that history PCA misses on these
quadratics. Paired-noise subtraction does not reliably recover curvature.**
This establishes an information distinction, not a new estimator or an
optimization speedup. Evidence is frozen under
`results/sprint4/campaign-20261012/`; [protocol](sprint4_protocol.md),
`processed/synthetic_estimators.csv`, `estimator_comparisons.json`,
`noise_identification_summary.csv`, and `synthetic_cost_summary.csv` retain
run-level measurements, uncertainty and costs. Ten fresh seed pairs per regime.

## Controlled objectives and sampling

16-dimensional rotated SPD H, condition 1,000, trace-matched covariance
condition 100, noise SD10, rank 4 and last64 gradients/parameters. Fixed AdamW
measurement policy .03, beta2=.99, conservative decay; not selected from these
evaluations. Cases: aligned, orthogonal top/noise spaces (antialigned spectrum),
45-degree top/bottom mixing, independent random covariance, orientation switch
after600 calls, and isotropic H. Snapshots80/600/1,200 capture phases. Noise
generation and probe RNGs are separate; gradients of each independent pair are
evaluated at exactly the same point, with64 pairs per refresh.

HVP range/Ritz uses8 probes, one power refinement and a final projected action:
24 autograd products. Common-noise central differences use the same additive
noise at +/-δv withδ=.01; independent differences use different noise at each
point/call. Random and coordinate controls use the same rank. H and oracle
subspaces appear only in evaluation/calibration, never in an optimizer update.

## Final recovered fraction of Hessian trace

Arithmetic means across10 runs; top-four oracle capture=.8420. Bootstrap units
are complete seeds, not steps, phases or individual eigenvectors.

| Noise orientation | Centered history | Paired noise space | Estimated subtraction | Exact signal covariance space | Random | HVP / common-noise FD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aligned | .8116 | .797 | .5034 | .815 | .240 | .8420 |
| Orthogonal/antialigned | .0065 | .007 | .0521 | .721 | .240 | .8420 |
| Partial | .4344 | .402 | .3319 | .803 | .240 | .8420 |
| Independent random | .2467 | .224 | .3212 | .762 | .240 | .8420 |
| Orientation switch, final | .0065 | .007 | .0521 | .720 | .240 | .8420 |

The exact signal-covariance column uses H Cov(theta) H, not a deployable
subtraction procedure. It isolates the target remaining after noise removal;
finite observed-history subtraction need not equal it, because of finite-sample
error, drift and trajectory dependence. The coordinate-only control captures
.429 on average here, and uncentered history has the same qualitative failures.
Isotropic H yields capture exactly .25 for every rank 4 subspace; its arbitrary
eigenvector overlaps are excluded from the inferential comparisons.

Paired HVP minus centered-history capture [95% percentile interval]: aligned
.0304 [.0239,.0372]; antialigned **.8355 [.8350,.8360]**; partial
.4076 [.4007,.4152]; independent random .5954 [.5608,.6317]; switch
.8355 [.8350,.8360]. These are exploratory intervals conditional on one fixed
measurement policy. The near-oracle recovery is expected for a clean exact
quadratic curvature operator with this spectrum, not a neural-network guarantee.

Estimated subtraction improves antialigned capture by only .0456 [.0370,.0550],
remaining below random and coordinate controls; it **worsens** aligned capture
by -.3082 [-.3415,-.2718], and partial capture by -.1025 [-.1333,-.0703].
Perfect noise identification still does not imply a Hessian eigenspace: see the
explicit weak-excitation counterexample in the [theory](sprint4_noise_curvature_theory.md).

## What noise estimates identify and why subtraction fails

Final paired-noise leading-space overlap with true Sigma averages .886–.894
for identifiable anisotropic cases. Relative covariance Frobenius error is
.319–.356 across regimes with64 pairs; run SD .043–.049 for the final primary
cases. Noise trace/eigenspace estimation is useful, but this sampling error is
large compared with the residual signal. Final signal/noise norm ratio is
.00389 in the antialigned control and .1142 aligned. Subtraction matrices are
indefinite: mean minimum eigenvalue approximately -170 antialigned and -148
aligned. Centering history cannot eliminate this uncertainty.

The noise-space overlap of isotropic Sigma is also nonidentifiable, rather
than evidence of estimator failure. The switching construction tests transfer
to a changed orientation; no parameters were chosen after seeing it. Early
history has strong transient signal and late history can have weak excitation.
The phase-resolved CSV retains this contrast, without treating correlated
window samples as independent confidence-interval observations.

## Accuracy and measured cost

Mean refresh costs across all six regimes/phases, including toy gradient/HVP
construction but excluding oracle metric evaluation:

| Mechanism | Extra information per refresh | Refresh seconds | Retained basis/state bytes |
| --- | --- | ---: | ---: |
| Centered history | 64 already observed gradients | .000658 | 512 |
| Paired noise PCA | 128 extra independent gradients | .001233 | 512 |
| Noise subtraction | same128 gradients plus small dense subtraction/eigh | .001395 | 512 |
| Randomized HVP/Ritz | 24 products | .02250 | 544 |
| Common-noise FD/Ritz | 48 gradient calls | .004846 | 544 |
| Independent FD/Ritz | 48 gradient calls | .006306 | 544 |

These tiny synthetic analytic gradients are much cheaper than neural gradients.
Toy HVP autograd overhead dominates. Timings are one-machine means with
operating-system scheduling variability; no GPU throughput claim follows.
Basis/state bytes exclude64-gradient storage (8,192 bytes), paired factors,
operator graphs and transient workspace. The standalone randomized estimator
retains O(pr) state and records an analytic workspace estimate; dense16x16
controls are analysis only. FD common-noise cancellation is an assumption of
this toy, not an inexpensive neural Hessian oracle.

Across all regimes including isotropic H, HVP/common-noise FD mean relative
Frobenius reconstruction error is .2765 (rank truncation included). Independent
FD error is **124.16**, showing that a seemingly plausible subspace can carry
wildly wrong scales. Its fixed-operator assumption is intentionally violated by
new noise each call; this is a disclosed failure control. For PCA controls,
`oracle_calibrated_error` fits curvature inside the inferred span using oracle H
and is an analysis upper bound, not an actual estimate of Hessian eigenvalues.

## Interpretation

Direct curvature action contains robust incremental information when noise and
curvature differ. Independent same-parameter gradients primarily measure
noise; their subtraction is statistically and identifiably limited. The
standalone `RandomizedCurvature` is classical randomized range/Ritz, and the
secant/HVP alternatives already occur in stochastic quasi-Newton, PSGD and
Hessian-free work. No mechanism here is labeled novel or connected to an update.
The [neural study](sprint4_neural_geometry.md) supplies the practical gate.
