"""Whole-run descriptive inference; never used by a trainer or selector."""
import numpy as np


def bootstrap_mean(values, replicates=4000, seed=20261015):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("nonempty finite run-level values required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (replicates, len(values)))].mean(1)
    lower, upper = np.quantile(means, [.025, .975])
    return {"n": len(values), "mean": float(values.mean()), "lower": float(lower), "upper": float(upper)}


def matched_point(curve, target):
    """Best hard retention at a minimum rejection precision; oracle audit only.

    Do not interpret this as an implementable operating-point selection rule,
    or as exact equality of achieved precision between methods.
    """
    candidates = [r for r in curve if r["rejected_noise_precision"] is not None
                  and r["rejected_noise_precision"] >= target
                  and r["hard_clean_retention"] is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["hard_clean_retention"],
        r["tail_clean_retention"] if r["tail_clean_retention"] is not None else -1,
        r["acceptance_fraction"]))
