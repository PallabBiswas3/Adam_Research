"""Exact fixed-preconditioner linear theory, not an exact theorem about Adam."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def stationary_covariance(hessian, noise_covariance, preconditioner, lr, beta1=0.0):
    """State is (x_t, m_{t-1}); m_t is a normalized EMA of noisy gradients."""
    h, s, b = map(np.asarray, (hessian, noise_covariance, preconditioner))
    d = len(h)
    if lr <= 0 or not 0 <= beta1 < 1:
        raise ValueError("positive learning rate and beta1 in [0,1) required")
    for matrix in (h, b):
        if matrix.shape != (d, d) or not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() <= 0:
            raise ValueError("H and B must be symmetric positive definite")
    if s.shape != (d, d) or not np.allclose(s, s.T) or np.linalg.eigvalsh(s).min() < -1e-12:
        raise ValueError("noise covariance must be positive semidefinite")
    a = np.block([[np.eye(d)-lr*(1-beta1)*b@h, -lr*beta1*b],
                  [(1-beta1)*h, beta1*np.eye(d)]])
    g = np.vstack([-lr*(1-beta1)*b, (1-beta1)*np.eye(d)])
    radius = np.max(np.abs(np.linalg.eigvals(a)))
    if radius >= 1:
        raise ValueError("no stable stationary covariance")
    q = g@s@g.T
    p = solve_discrete_lyapunov(a, q)
    return {"covariance": p, "transition": a, "noise_map": g,
            "spectral_radius": float(radius), "loss": float(np.trace(h@p[:d, :d])/2),
            "relative_residual": float(np.linalg.norm(p-a@p@a.T-q)/max(np.linalg.norm(q), 1e-30))}


def scalar_stationary_loss(h, variance, b, lr, beta1=0.0):
    denominator = 2*(1+beta1)-lr*b*h*(1-beta1)
    if denominator <= 0:
        raise ValueError("unstable scalar recurrence")
    return lr*b*variance*(1+beta1)/(2*denominator)


def second_moment_statistics(variance, beta2):
    """Stationary v for iid zero-mean Gaussian gradients with fixed variance."""
    if variance < 0 or not 0 <= beta2 < 1:
        raise ValueError("invalid variance or decay")
    return {"mean": variance, "variance": 2*variance**2*(1-beta2)/(1+beta2)}


def effective_condition(h, b):
    values, vectors = np.linalg.eigh(b)
    root = (vectors*np.sqrt(values))@vectors.T
    spectrum = np.linalg.eigvalsh(root@h@root)
    return float(spectrum[-1]/spectrum[0])
