"""Independent equations from DSS and DivideMix; no unlicensed source copied."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def adjusted_probability(probability, marginal, prior):
    if not torch.isfinite(prior).all() or (prior <= 0).any() or (marginal <= 0).any():
        raise ValueError("strictly positive finite prior/marginal required")
    adjusted = probability*prior/marginal
    return adjusted/adjusted.sum(-1, keepdim=True)


def mann_kendall_z(history):
    """History [epochs, examples, classes]; paper's no-tie-adjustment variance."""
    n = len(history)
    if n < 2:
        return torch.zeros_like(history[0])
    score = torch.zeros_like(history[0])
    for t in range(1, n):
        score += torch.sign(history[t]-history[:t]).sum(0)
    variance = n*(n-1)*(2*n+5)/18
    return (score-torch.sign(score))/variance**.5


def candidate_classes(history, observed, threshold):
    exclude = mann_kendall_z(history) > threshold
    exclude[torch.arange(len(observed)), observed] = False
    return exclude


def ccs_loss(logits, observed, exclude, keep):
    if exclude[torch.arange(len(observed)), observed].any():
        raise ValueError("annotated class must remain in denominator")
    modified = logits.masked_fill(exclude, -torch.inf)
    per_example = F.cross_entropy(modified, observed, reduction="none")
    return (per_example*keep).mean()


def sharpen(probability, temperature):
    if temperature <= 0:
        raise ValueError("positive temperature required")
    powered = probability**(1/temperature)
    return powered/powered.sum(-1, keepdim=True)


def refine_targets(observed, probability, clean_weight, temperature):
    label = F.one_hot(observed, probability.shape[-1]).to(probability.dtype)
    refined = clean_weight[:, None]*label+(1-clean_weight[:, None])*probability
    return sharpen(refined, temperature).detach()


def semisupervised_loss(logits_x, target_x, logits_u, target_u, mse_class_reduction="mean"):
    """Official DivideMix class-mean MSE; paper Eq.10 class-sum selectable.

    The difference changes effective lambda_u by the class count. Preserve and
    disclose the implementation convention instead of silently conflating them.
    """
    lx = -(target_x*F.log_softmax(logits_x, dim=1)).sum(1).mean()
    difference = (logits_u.softmax(1)-target_u)**2
    if mse_class_reduction not in {"mean", "sum"}:
        raise ValueError("MSE class reduction must be mean or sum")
    lu = difference.mean() if mse_class_reduction == "mean" else difference.sum(1).mean()
    prediction = torch.cat([logits_x, logits_u]).softmax(1).mean(0)
    prior = torch.ones_like(prediction)/len(prediction)
    reg = (prior*(prior.log()-prediction.clamp_min(1e-12).log())).sum()
    return lx, lu, reg


def fit_clean_probability(loss, seed):
    loss = np.asarray(loss, dtype=np.float64)
    if len(loss) < 4 or not np.isfinite(loss).all() or np.ptp(loss) <= 1e-12:
        raise ValueError("nondegenerate finite observed-loss scan required")
    scaled = ((loss-loss.min())/np.ptp(loss)).reshape(-1, 1)
    mixture = GaussianMixture(n_components=2, max_iter=10, tol=.01, reg_covar=.0005, random_state=seed)
    mixture.fit(scaled)
    return mixture.predict_proba(scaled)[:, mixture.means_.argmin()]


def classwise_keep(loss, observed, fraction):
    keep = torch.zeros(len(loss), dtype=torch.bool)
    for c in observed.unique():
        indices = torch.where(observed == c)[0]
        selected = indices[torch.argsort(loss[indices], stable=True)[:max(1, int(fraction*len(indices)))]]
        keep[selected] = True
    return keep


def peer_policies(first, second):
    """Exchange frozen policies, never self-select in a two-network update."""
    return second, first
