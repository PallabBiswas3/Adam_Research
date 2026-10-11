# Sprint 3 mathematical statements and their limits

## Exact fixed-preconditioner model

Let f(x)=x^T H x/2, H symmetric positive definite, and g_t=H x_t+xi_t.
Assume xi_t is independent of all past iterates, zero mean, covariance Sigma,
with finite second moment. Gaussianity is unnecessary for the covariance result.
Take a **constant**, deterministic symmetric positive definite scaling B,
constant eta>0, normalized momentum beta in [0,1), no decay and no clipping:

\[
m_t=\beta m_{t-1}+(1-\beta)(H x_t+\xi_t),\qquad
x_{t+1}=x_t-\eta B m_t.
\]

This is a mathematical reference model, not a newly implemented optimizer.
For z_t=(x_t,m_{t-1}), z_{t+1}=A z_t+G xi_t, where

\[
A=\begin{pmatrix}
I-\eta(1-\beta)BH&-\eta\beta B\\
(1-\beta)H&\beta I
\end{pmatrix},\quad
G=\begin{pmatrix}-\eta(1-\beta)B\\(1-\beta)I\end{pmatrix}.
\]

**Proposition, proved by the recurrence:** if rho(A)<1, the limiting centered
covariance is the unique solution P=A P A^T+G Sigma G^T. Independence eliminates
the cross terms; the convergent series sum_{j>=0} A^j G Sigma G^T (A^T)^j proves
existence and uniqueness. The limiting mean is zero. Expected limiting loss is
tr(H P_xx)/2. For deterministic initial state, finite-time covariance is
P_t=sum_{j=0}^{t-1} A^j G Sigma G^T(A^T)^j, with mean A^t z_0. Therefore loss
has both transient mean-error and stochastic-variance components. Changing
noise covariance is handled exactly by P_{t+1}=A P_t A^T+G Sigma_t G^T.

`training/quadratic_theory.py` solves the **discrete** Lyapunov equation and
checks stability. This is related to, but does not adopt the continuous-time
approximation in [Mandt, Hoffman and Blei](https://jmlr.org/papers/volume18/17-214/17-214.pdf).
Their Section 3 states constant-noise, locally quadratic and diffusion
assumptions; Section 5's Fisher/Hessian identity requires an additional
well-specified statistical-model assumption. Our additive-noise constructions
impose no such identity.

When H, B and Sigma commute, diagonalize them together. In a coordinate with
curvature h, scaling b and noise variance s, eliminate momentum to obtain

\[
x_{t+1}=[1+\beta-\eta b(1-\beta)h]x_t-\beta x_{t-1}
-\eta b(1-\beta)\xi_t.
\]

Its scalar stability condition is 0<eta b h(1-beta)<2(1+beta).
Solving its two Yule–Walker covariance equations yields

\[
\mathbb E[f_i]=
\frac{\eta b s(1+\beta)}{2[2(1+\beta)-\eta b h(1-\beta)]}.
\]

At beta=0 this reduces to eta b s/[2(2-eta b h)]. At small eta it is eta b s/4
to first order, independent of normalized momentum. Thus decreasing eta can
reduce a stochastic floor without adding geometric information, while making
transient contraction slower. Curvature enters the finite-step denominator and
stability limit. This independence from beta is **not** a statement about
PyTorch SGD at the same nominal rate: its unnormalized momentum corresponds
to eta_eff=lr/(1-beta). Learning-rate comparisons must account for that scale.

Validation: scalar formula versus matrix Lyapunov solution in automated tests;
six independent simulations (32 whole trajectories each), burn-in 2,000,
10,000 measured steps. Results, resampling intervals and numerical residuals
are in `processed/theory_validation.json`. They validate the stated fixed-B
model; agreement does not prove the approximation below for actual Adam.

## Approximate Adam interpretation, explicitly conditional

Near a quadratic optimum, **if** Hx is negligible relative to stationary noise,
coordinate noise variances are positive, the second moment fluctuates little,
bias correction has settled, epsilon is accounted for, and dependencies between
m_t and v_t can be neglected, replace the denominator by
B_ii ≈ 1/(sqrt(Sigma_ii)+epsilon). The exact fixed-B statements then provide a
local approximation. Adam's actual denominator depends on the current and
past gradients; generally E[m/sqrt(v)] is not E[m]/sqrt(E[v]). No equality or
unconditional Adam stationary-error theorem is claimed.

For iid Gaussian **pure-noise** scalar gradients with variance s, the uncorrected
EMA v_t=beta2 v_{t-1}+(1-beta2)g_t^2 has stationary

\[
\mathbb E[v]=s,\qquad
\operatorname{Var}(v)=2s^2\frac{1-\beta_2}{1+\beta_2}.
\]

Proof: E[g^2]=s, Var(g^2)=2s^2, independence and the sum of squared EMA weights.
Relative standard deviation is sqrt(2(1-beta2)/(1+beta2)); it is about 0.324,
0.100 and 0.0316 for beta2=.9,.99,.999. Following a sudden pure-noise variance
change, E[v_n]=s_new+(E[v_0]-s_new) beta2^n. Half-lives are about 6.58, 68.97
and 692.80 steps. Bias correction adds its known factor 1/(1-beta2^t) at finite
times; these formulas describe the uncorrected EMA or settled limit. Correlated
noise adds covariance terms; nonzero Hx adds signal and iterate dependencies.

**Conjecture to test, not a result:** after a noise decrease, a large beta2 may
retain an oversized denominator and prolong transient error. After an increase,
it may initially underscale the denominator. Faster beta2 responds sooner but
has noisier scaling. Schedules, cumulative Shampoo normalization, momentum
normalization and covariance orientation can all alter the comparison. A tail
loss gap alone cannot distinguish them.

## A limited geometry inefficiency condition, proved

For k>1 define the 45-degree-rotated Hessian

\[
H=\frac12\begin{pmatrix}k+1&k-1\\k-1&k+1\end{pmatrix}.
\]

**Proposition:** any positive diagonal constant B=diag(d1,d2) satisfies
kappa(B^(1/2) H B^(1/2))>=k; equality holds when d1=d2. To prove it, put
a=(k+1)/2, c=(k-1)/2. The preconditioned matrix has trace a(d1+d2) and determinant
d1 d2(a^2-c^2). Its trace/sqrt(determinant) is minimized at d1=d2 by AM–GM.
For a positive 2x2 spectrum, this ratio equals sqrt(kappa)+1/sqrt(kappa), an
increasing function of kappa>=1. Equal scaling has condition k, proving the
bound. Full B=H^-1 has condition 1. For **fixed-B, momentum-free, deterministic**
gradient descent, the best constant-rate worst-direction contraction is
(kappa-1)/(kappa+1). Hence this family establishes a strict limitation of
diagonal constant scaling relative to an oracle structured inverse.

The corresponding diagonal H=diag(1,k) admits a diagonal inverse with condition
1. This is a geometry-dependent limitation with identical spectra. It is
classical preconditioning reasoning, not a new algorithm or proof of Adam's
failure. Time-varying Adam, momentum acceleration, limited budgets, unknown H,
cost of estimating H, and stochastic floors are outside the proposition.
The oracle inverse is not an affordable neural comparator.

## Why covariance need not recover curvature

For independent additive zero-mean noise,

\[
\operatorname{Cov}(g_t)=H\operatorname{Cov}(x_t)H+\Sigma,
\quad
\mathbb E[g_t g_t^T]=H\mathbb E[x_t x_t^T]H+\Sigma.
\]

Noise is independent of x_t **before** observing the current gradient, so the
cross terms vanish. Neither equation says Cov(g)=H. During a nonstationary
history window, changing means also contribute; centering the window does not
isolate curvature. Near an optimum, Sigma may dominate, and a noise subspace
may be unrelated or antialigned to H's leading directions. A gradient-history
sketch must demonstrate useful curvature capture beyond random/diagonal
controls, phase stability and a practical benefit before it justifies a
geometric correction. No such implication follows from a large covariance
eigenvalue by itself.
