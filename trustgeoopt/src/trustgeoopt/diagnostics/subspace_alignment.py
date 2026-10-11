import torch


def projected_energy_fraction(vector, orthonormal_basis, eps=1e-12):
    """Fraction of squared energy in an orthonormal p by k basis."""
    if vector.square().sum() <= eps:
        return float("nan")
    return float((orthonormal_basis.T @ vector).square().sum()/vector.square().sum())
