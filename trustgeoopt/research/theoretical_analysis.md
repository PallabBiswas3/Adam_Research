# Limited theoretical analysis

## Lemma 1: frozen preconditioned SPD quadratic

Let H,P be fixed SPD and f=theta^T H theta/2-b^T theta. With full gradient,
no momentum/no decay and theta_next=theta-eta P^{-1}grad f, transformed error
z=P^{1/2}(theta-H^{-1}b) satisfies z_next=(I-eta M)z,
M=P^{-1/2}H P^{-1/2}. Proof: substitute grad f=H(theta-H^{-1}b).
M is symmetric SPD; diagonalize it orthogonally. Stability is equivalent to
0<eta<2/lambda_max(M). Optimal fixed-rate contraction is
(kappa(M)-1)/(kappa(M)+1). This is not a theorem for Adam momentum.

If P=aI and H=aI+u c u^T, c>=0, ||u||=1, exact rank-one correction P'=H
makes M'=I. Conversely H=I, P=I and P'=I+c uu^T yields eigenvalues
1/(1+c) and 1 (p>=2), worsening kappa from 1 to 1+c. PSD correction alone
therefore guarantees no conditioning improvement. If H and P share eigenvectors,
compare max_j h_j/p_j over min_j h_j/p_j; direction-specific correction can
improve or worsen this ratio. Woodbury validation is numerical, not a convergence proof.

## Lemma 2: one-step smooth descent, conditional metric

Suppose f is L-smooth/lower bounded and deterministic grad g. Let the SPD
inverse metric A have spectrum [a,b], a>0. Smoothness yields
f(theta-eta A g)<=f(theta)-eta*a||g||^2+L*eta^2*b^2||g||^2/2.
Thus eta<=a/(L b^2) gives decrease >=eta*a||g||^2/2.
Proof: plug s=-eta A g into f(theta+s)<=f(theta)+g^Ts+L||s||^2/2,
using g^TAg>=a||g||^2 and ||Ag||<=b||g||. Summing bounds the average
squared gradient by 2(f0-f_lower)/(T eta a) for fixed eta. A may vary if
uniform bounds hold. This proves stationary-point progress only.

If A_t is measurable before an unbiased stochastic g_t with conditional
variance <=sigma^2, the expected bound adds L eta^2 b^2 sigma^2/2.
A metric or gate depending on the *same* g_t is correlated with noise, so this
conditional proof does not automatically apply. Independent samples or stronger
joint assumptions are needed. Momentum adds tracking error <grad f,A(m-g)>;
decay changes the update/objective relation. No full TrustGeoOpt theorem is proved.

## Lemma 3 and non-identifiability of moment mismatch

For stationary theta, independent gradients with second moment s, an EMA has
E[v_t]=beta2^t v0+(1-beta2^t)s. After a switch to s', expectation approaches
s' geometrically. This follows by induction of the scalar/vector recurrence.
A q ratio can indicate a distribution shift but cannot identify its cause:
E||g||^2=||E g||^2+trace Cov(g). A larger mean descent signal and larger
noise can produce equal q. For observed history, deterministic changes in mean
and stochastic fluctuations can coincide; no distribution-free reliable detector
exists from q alone. Momentum alignment is also ambiguous near an optimum.

Conjecture (unproved): under separated mean/noise shifts and sufficient independent
gradient samples, a joint lag/alignment score improves harmful-update prediction.
Counterexample program: isotropic f and high noise; aligned stale momentum;
same q under mean versus variance shifts. Existing Adam counterexamples and
edge-of-stability theory require separate assumptions; do not transfer those
results to this prototype or interpret finite-step loss oscillation as nonconvergence.
