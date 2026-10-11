import numpy as np
from scipy.stats import rankdata


def prediction_metrics(labels, scores, threshold):
    """No sklearn dependency. AP/AUROC correctly group tied scores."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("binary labels required")
    p, n = int(labels.sum()), int((1-labels).sum())
    predicted = scores >= threshold
    tp, fp = int((predicted & (labels == 1)).sum()), int((predicted & (labels == 0)).sum())
    ap = auc = None
    if p:
        order = np.argsort(-scores, kind="stable")
        sorted_y, sorted_s = labels[order], scores[order]
        boundaries = np.r_[np.flatnonzero(np.diff(sorted_s)), len(sorted_s)-1]
        cumulative = sorted_y.cumsum()[boundaries]
        precision = cumulative/(boundaries+1)
        ap = float(np.sum(precision*np.diff(np.r_[0, cumulative]))/p)
    if p and n:
        ranks = rankdata(scores)
        auc = float((ranks[labels == 1].sum()-p*(p+1)/2)/(p*n))
    return dict(samples=int(len(labels)), positive=p, negative=n, threshold=float(threshold),
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/p if p else None,
                false_positive_rate=fp/n if n else None, average_precision=ap, roc_auc=auc,
                prevalence=p/(p+n) if p+n else None)
