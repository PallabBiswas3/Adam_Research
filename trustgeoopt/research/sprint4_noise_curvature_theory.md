# Noise versus curvature: statements with explicit assumptions

These are classical identities, counterexamples and limited deductions. They
are not new Adam convergence theorems. Fixed-preconditioner stationary theory
from [Sprint 3](sprint3_mathematics.md) remains limited to its stated model.

## 1. What covariance identifies

Let H be deterministic SPD and theta_t be measurable before current noise is
drawn. Suppose E[xi_t | theta_t]=0 and Cov(xi_t | theta_t)=Sigma, with finite
second moments. Independence of current noise and theta_t is sufficient but
stronger than these conditional assumptions. Total covariance gives exactly

\[
\mathrm{Cov}(g_t)=H C_t H+\Sigma,\qquad C_t=\mathrm{Cov}(\theta_t).
\]

For uncentered moments replace C_t by E[theta_t theta_tᵀ]. Nonzero mean adds
H mu muᵀ H. In a heterogeneous window interpreted as a mixture over times,
replace Sigma by mean conditional Sigma_t and use the mixture covariance of
theta; mean drift contributes. Actual finite trajectory sample covariance is
not an iid estimator of this ensemble mixture: noise at time i changes future
theta, and demeaning creates lag-dependent cross terms. We assert no iid
concentration rate for history windows.

For nonlinear neural objectives, total covariance instead gives
Cov(g)=Cov(grad f(theta))+E[Sigma(theta)]. There is no fixed-H identity unless
a justified local linearization with controlled remainder is introduced.

## 2. Exact eigenvector agreement needs ordering, not just commutation

If H and C commute, an orthonormal basis simultaneously diagonalizes them and
R=H C H has eigenvalues h_i² c_i. With isotropic Sigma=tau I, history covariance
has the same eigenbasis. Its leading-r space equals H's leading-r space only
if the ordering of h_i² c_i preserves the top-r set and its gap is positive.
The sufficient special case C=s I, s>0, yields R=s H², whose order agrees with
H. These statements follow by direct diagonalization.

Counterexample: H=diag(1,10), C=diag(200,.001), Sigma=I. After **perfect** noise
subtraction, R=diag(200,.1); its leading direction is orthogonal to H's.
Alternatively C=H⁻³ makes R=H⁻¹ and reverses every eigenvalue ordering.
Zero excitation in a Hessian direction prevents its identification from signal
covariance. Exact noise subtraction does not by itself fix any of these cases.

Commuting anisotropic noise can reorder the top set even without rotating
vectors. Noncommuting noise can rotate vectors. Near an optimum, excitation
may vanish while Sigma remains nonzero, so history tracks noise. Isotropic H
has no distinguished top-r eigenspace: overlap with an arbitrary chosen basis
has no curvature-identification interpretation.

## 3. A valid perturbation bound

Choose a reference A=s H²+tau I with s>0 and top-r gap
delta=s(h_r²-h_{r+1}²)>0, eigenvalues in descending order. Write a symmetric
estimated history covariance as A+E, where

\[
E=H(C-sI)H+(\Sigma-\tau I)+E_{\rm sample}.
\]

By [Yu, Wang & Samworth, Theorem 2](https://arxiv.org/pdf/1405.0680),

\[
\|\sin\Theta(\widehat U,U_H)\|_F
\le {2\min(\sqrt r\|E\|_{op},\|E\|_F)\over\delta}.
\]

Proof/application: A is symmetric and has exactly H's ordered eigenspaces;
apply the population-gap bound with the top-r cluster. Triangular inequality
can upper bound ||E||op by ||H||op²||C-sI||op + ||Sigma-tau I||op +
||E_sample||op. The bound is informative only when these perturbations are
small relative to delta. Large isotropic noise has no population eigenvector
effect, but increases sample error; anisotropic noise competes directly.
Correlated histories require an independently justified bound on E_sample.
We do not insert an iid 1/sqrt(n) rate for optimizer steps.

After unbiased noise estimation, E becomes H(C-sI)H+E_sample-E_noise (and
possible drift/heteroscedastic mismatch). Subtraction removes only one term.
The result can be indefinite, and clipping to PSD introduces another bias.
The experiments retain negative minimum eigenvalues rather than silently
declaring an SPD Hessian estimate.

## 4. Identifiability after subtraction

If R=H C H is known and C is unknown, every SPD candidate H' gives an admissible
C'=H'⁻¹ R H'⁻¹. Thus covariance alone cannot identify H, even with exact noise
knowledge. If C is also known SPD, then H is uniquely identified by

\[
H=C^{-1/2}(C^{1/2} R C^{1/2})^{1/2}C^{-1/2}.
\]

Proof: C^{1/2} R C^{1/2}=(C^{1/2} H C^{1/2})² and SPD matrices have a unique
SPD square root. Estimating and inverting C in weakly excited directions is
ill-conditioned and expensive; rank deficiency invalidates this full recovery
formula. It is an identifiability observation, not an implemented optimizer.

## 5. Paired independent gradients estimate noise

At the SAME theta, equal-size conditionally independent gradients g1/g2 with
mean grad f(theta) and covariance Sigma_B(theta) satisfy

\[
D=(g_1-g_2)/\sqrt2,\quad E[DD^T|\theta]=\Sigma_B(\theta).
\]

Proof: the common mean cancels and the two covariance terms add; conditional
cross-covariance is zero. With n independent pairs, FFᵀ where columns are
(g1-g2)/sqrt(2n) is unbiased. It need not be centered because E[D]=0. Independent
Gaussian pairs give Var of projected variance estimate=2(uᵀSigma_B u)²/n.
This estimates NOISE, not H. Stored factors need O(pn) rather than O(p²);
streaming/sketching alternatives already exist.

With correlation C12 between pair members, the expectation is Sigma_B minus
(C12+C12ᵀ)/2. Reusing the same minibatch produces zero, not a good noise estimate.
Unequal batch sizes require a different scaling. Our neural draws sample with
replacement independently from a fixed training population, so Sigma_B equals
the population-centered per-example gradient covariance divided by B. Disjoint
sampling without replacement has different finite-population/correlation terms.
Gradients at two DIFFERENT theta add signal drift and no longer isolate noise.

## 6. Curvature products and secants identify an action

For smooth f, differentiating grad f(theta) in direction v gives H(theta)v.
[Pearlmutter](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf) computes
this without constructing the Hessian. Subsampled products estimate a sampled
Hessian, whose expectation equals population H when differentiation and
expectation can be exchanged. They still have curvature sampling variance.

Central differences [g(theta+δv)-g(theta-δv)]/(2δ) give exact Hv for quadratics
with a COMMON additive noise realization. Independent additive noise instead
contributes covariance Sigma/(2δ²). Its variance diverges as δ→0. For smooth
nonquadratic gradients central bias is O(δ²) under bounded appropriate third
derivatives of the gradient; weaker Hessian-Lipschitz assumptions yield weaker
bounds. ReLU kinks invalidate a uniform smooth remainder bound.

For the same neural minibatch, additive label noise cancels partially but the
action is its batch Hessian/GGN, not exact population H. Data-dependent curvature
and crossing activation boundaries remain errors. Large probe batches reduce
sampling variance at measured cost. A changing independent batch per operator
call violates the fixed symmetric operator assumption behind range/Ritz; its
reported failure is expected, not a new discovery.

## 7. Neural curvature distinction and cost

Binary logits z(theta) with cross entropy give
H=Jᵀ[diag(p)-ppᵀ]J + sum_c (p_c-y_c) Hessian(z_c).
The first term is PSD GGN and equals the model Fisher for categorical logits;
the observed-label empirical Fisher generally does not equal it, as discussed
by [Kunstner et al.](https://arxiv.org/pdf/1905.12558).
Within one ReLU cell, a single layer's logits are affine in that layer's
weights/biases while other layers are fixed. The layer-block Hessian therefore
equals its GGN there. Cross-layer logit derivatives need not vanish; no
full-network equality is claimed. Tests verify both block curvature and mean
observed-label gradients against autograd.

The standalone estimator is classical [randomized range/Ritz](https://arxiv.org/pdf/0909.4061):
rank r, oversampling s, q extra power passes need (q+2)(r+s) products, O(p(r+s))
workspace and O(pr) retained basis. Our r=s=4,q=1 costs 24 HVPs or 48 gradient
calls per finite-difference refresh, plus QR and a small spectral solve.
Symmetrization of a noisy small matrix and PSD clipping are numerical choices,
not proof of accurate curvature. Refresh amortization must account for probe
batch size, changing/stale geometry, activation storage and total compute.
Better subspace capture alone does not prove better optimization or justify
a new adaptive method. We do not connect these probes to an optimizer.
