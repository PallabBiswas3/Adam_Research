# Mathematical formulation

Let L(theta)=N^{-1} sum_i ell(f_theta(x_i), y_i), and g_t=grad L_{B_t}(theta_t).
Assume a fixed training objective unless an experiment is explicitly tracking.
Adam: m_t=beta1 m_{t-1}+(1-beta1)g_t;
v_t=beta2 v_{t-1}+(1-beta2)g_t elementwise squared.
Use mhat=m_t/(1-beta1^t), vhat=v_t/(1-beta2^t).
AdamW: theta_next=(1-eta*wd)theta-eta*mhat/(sqrt(vhat)+eps).
AMSGrad uses the running maximum of *uncorrected* v before bias correction
in the integrated PyTorch implementation. Adam/RMSProp/SGD weight decay is
coupled L2 in the baseline factory; AdamW and our specified AdaBelief reference
use decoupled decay. Compare wd=0 for mechanism studies; label this distinction.

## Linear geometry

For L(w)=||Xw-y||^2/(2N), grad L=X^T(Xw-y)/N and H=X^TX/N.
H is the uncentered input second moment; it equals covariance only for centered
inputs. Singular vectors of X are Hessian eigenvectors; eigenvalues are squared
singular values/N. If H is SPD, the optimum is unique. Along eigenvector j,
GD error contracts by 1-eta*lambda_j. Stability requires 0<eta<2/lambda_max.
The optimal constant GD rate is 2/(lambda_max+lambda_min); worst contraction
is (kappa-1)/(kappa+1). Rank deficient X needs a pseudoinverse and null-space care.
Rotating X preserves its spectrum but changes whether diagonal scaling can
represent principal directions. It does not prove Adam fails.

## Nonlinear geometry

For squared loss, stacked residual r and output Jacobian J (N outputs by p
parameters for scalar regression) give H=J^T J/N + sum_i r_i Hessian f_i/N.
Gauss-Newton discards the residual term. General loss gives G=J^T H_ell J/N;
G is PSD for convex output losses, while the actual network Hessian can be
indefinite. Deep ReLU is only piecewise smooth; results requiring smoothness
apply away from activation boundaries or to smooth activations.

| Matrix | Dimension | What it measures | Caveat |
| --- | --- | --- | --- |
| Input covariance | d by d | variation in x | curvature only in special linear losses |
| Layer activation moment | a by a | variation in layer inputs | one factor of a layer approximation |
| Gradient covariance | p by p | centered stochastic gradient variation | not Hessian; mean outer product is distinct |
| Gradient second moment | p by p | E[gg^T] | mixes signal and noise |
| Hessian | p by p | second derivative of loss | may be indefinite |
| Fisher | p by p | expected score outer product under model labels | empirical-label Fisher is generally different |

At fixed theta for a noisy quadratic g=H(theta-theta*)+xi, E[gg^T]
=H e e^T H+Sigma. Noise can dominate its principal directions. Thus gradient
history is not an unbiased curvature estimate. History at changing theta adds
trajectory drift. No dense parameter Hessian is constructed for neural models.

## Controlled mechanisms and controls

1. Diagonal anisotropy (kappa=10,100,1000,10000): compare tuned GD/momentum,
AdamW, AMSGrad and Shampoo; include H=I as a no-benefit control.
2. Orthogonally rotated H with same spectrum/start norm: test coordinate effects,
never compare different objective scale accidentally.
3. Switched H and optimum: evaluate f_t before and after the update at the same
t; report tracking distance and exclude switch jumps from loss-spike labels.
4. Additive noise with anisotropic covariance and scheduled scale: log true versus
observed gradients; matched random streams across methods, stationary and shift controls.
5. Moment lag: after a variance switch, E[v_{tau+s}]=b+(a-b)beta2^s when prior
uncorrected moment expectation was a and new gradient second moment is b.
Half-life is log(1/2)/log(beta2). Bias correction changes startup behavior.
6. Alignment: a noisy minibatch can oppose true gradient; compare independent
minibatches and history reuse, include their gradient costs.
7. Isotropic/noise-free case: extra geometric work may be unnecessary or harmful.

## Candidate primitive, not an enabled optimizer

P=B+F F^T, B=diag(D+lambda)>0; F=U C^{1/2}, C PSD.
P^{-1}m=B^{-1}m-B^{-1}F(I+F^T B^{-1}F)^{-1}F^T B^{-1}m.
Dimensions: B p-vector, F p by k, inner system k by k. Storage O(pk+k^2),
assembly O(pk^2), factorization O(k^3), application O(pk).
For Adam equivalence D=sqrt(vhat)+eps, lambda=0, F empty. D is a gradient-scale
matrix; covariance has squared gradient units and Hessian has loss/parameter^2.
Any future mapping requires normalization and a separately tuned strength with
matching units. Adding raw covariance to D is not justified dimensional arithmetic.
