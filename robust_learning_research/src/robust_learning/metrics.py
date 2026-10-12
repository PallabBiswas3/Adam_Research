"""Evaluation-only labels/oracles; no metrics are available inside selectors."""
import numpy as np


def classification_metrics(true, prediction, probability, classes=3):
    matrix = np.zeros((classes, classes), dtype=int)
    for a, b in zip(true, prediction):
        matrix[a, b] += 1
    tp = matrix.diagonal()
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(classes), where=matrix.sum(1) > 0)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(classes), where=matrix.sum(0) > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(classes), where=precision+recall > 0)
    confidence = probability.max(1)
    correct = np.asarray(true) == np.asarray(prediction)
    ece = 0.
    for lower in np.linspace(0, .9, 10):
        selected = (confidence >= lower)&(confidence < lower+.1 if lower < .89 else confidence <= 1)
        if selected.any():
            ece += selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return {"macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "accuracy": float(correct.mean()), "ece": float(ece), "per_class_recall": recall.tolist(),
            "confusion_matrix": matrix.tolist()}


def selection_metrics(keep, oracle, hard_threshold=.5):
    clean = ~oracle.flipped
    hard = oracle.ambiguity >= hard_threshold
    tail = oracle.true_y == 2
    accepted, rejected = int(keep.sum()), int((~keep).sum())
    result = {"accepted": accepted, "acceptance_fraction": accepted/len(keep),
        "accepted_clean_precision": float(clean[keep].mean()) if accepted else None,
        "rejected_noise_precision": float(oracle.flipped[~keep].mean()) if rejected else None,
        "noise_recall": float((~keep)[oracle.flipped].mean()) if oracle.flipped.any() else None,
        "clean_recall": float(keep[clean].mean()) if clean.any() else None}
    for name, mask in (("hard_clean", clean&hard), ("easy_clean", clean&~hard), ("tail_clean", clean&tail)):
        result[name+"_count"] = int(mask.sum())
        result[name+"_retention"] = float(keep[mask].mean()) if mask.any() else None
    for c in range(3):
        mask = clean&(oracle.true_y == c)
        result[f"clean_class{c}_retention"] = float(keep[mask].mean()) if mask.any() else None
    return result


def operating_curve(scores, oracle, fractions):
    """Oracle used only to score frozen rankings, never train/choose a policy.

    Accept all scores tied at the cutoff; actual fractions can exceed requests.
    """
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or len(scores) != len(oracle.true_y):
        raise ValueError("finite sample scores required")
    rows = []
    ordered = np.sort(scores)[::-1]
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("fraction in (0,1] required")
        count = max(1, int(fraction*len(scores)))
        keep = scores >= ordered[count-1]
        rows.append({"requested_fraction": fraction, "threshold": float(ordered[count-1]),
                     **selection_metrics(keep, oracle)})
    return rows
