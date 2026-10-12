"""Observed-label-only toy training. Evaluation oracle is never imported here."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import small_loss_mask, coteaching_losses, confidence_gap_z, classwise_scores
from .metrics import classification_metrics


def tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel()*obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def evaluate(model, clean_set):
    with torch.no_grad():
        probability = model(torch.tensor(clean_set.x)).softmax(1).numpy()
    return classification_metrics(clean_set.y, probability.argmax(1), probability)


def run_training(train, validation, config, deadline):
    """No true training labels/flip flags/ambiguity can cross this interface."""
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    def network():
        return torch.nn.Sequential(torch.nn.Linear(2, config["hidden"]), torch.nn.ReLU(), torch.nn.Linear(config["hidden"], 3))
    model = network()
    method = config["method"]
    if method not in {"ce", "global_small_loss", "classwise_small_loss", "coteaching_toy", "confidence_tracking_toy"}:
        raise ValueError("unknown baseline")
    second = None
    if method == "coteaching_toy":
        torch.manual_seed(config["seed"]+10000)
        second = network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0)
    optimizer2 = torch.optim.AdamW(second.parameters(), lr=config["lr"], weight_decay=0) if second is not None else None
    x, y = torch.tensor(train.x), torch.tensor(train.observed_y)
    batch_rng = torch.Generator().manual_seed(config["seed"]+6000)
    history, rows = [], []
    mask = torch.ones(len(y), dtype=torch.bool)
    loop_begin = time.perf_counter()
    selector_seconds = 0.
    updates = 0
    for epoch in range(config["epochs"]):
        if time.perf_counter() >= deadline:
            return {"status": "budget_censored", "epochs_completed": epoch, "rows": rows}, None
        selection_begin = time.perf_counter()
        with torch.no_grad():
            losses = F.cross_entropy(model(x), y, reduction="none")
            if epoch >= config["warmup"] and method in {"global_small_loss", "classwise_small_loss", "confidence_tracking_toy"}:
                mask = small_loss_mask(losses, y, config["keep_fraction"], method == "classwise_small_loss")
                if method == "confidence_tracking_toy" and history:
                    mask |= confidence_gap_z(torch.stack(history), y) > config["confidence_z_threshold"]
        selector_seconds += time.perf_counter()-selection_begin
        order = torch.randperm(len(y), generator=batch_rng)
        actual_mask = torch.zeros(len(y), dtype=torch.bool)
        for indices in order.split(config["batch_size"]):
            if time.perf_counter() >= deadline:
                return {"status": "budget_censored", "epochs_completed": epoch, "rows": rows}, None
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices])
            if second is not None:
                optimizer2.zero_grad(set_to_none=True)
                forget = config["nominal_noise_rate"]*min(epoch/max(config["warmup"], 1), 1)
                loss, loss2, own1, own2 = coteaching_losses(logits, second(x[indices]), y[indices], forget)
                actual_mask[indices[own2]] = True  # peer selections actually training model1
                loss.backward()
                loss2.backward()
                optimizer.step()
                optimizer2.step()
            else:
                selected = mask[indices]
                actual_mask[indices[selected]] = True
                if not selected.any():
                    continue
                per_sample = F.cross_entropy(logits, y[indices], reduction="none")
                loss = (per_sample*selected).mean() if method == "confidence_tracking_toy" else per_sample[selected].mean()
                loss.backward()
                optimizer.step()
            updates += 1
        with torch.no_grad():
            if method == "confidence_tracking_toy":
                history.append(model(x).softmax(1))
        validation_metrics = evaluate(model, validation)
        rows.append({"epoch": epoch+1, "validation_macro_f1": validation_metrics["macro_f1"],
            "accepted_fraction": float(actual_mask.float().mean()), "seconds": time.perf_counter()-loop_begin})
        if not np.isfinite(rows[-1]["validation_macro_f1"]):
            return {"status": "failed", "error": "nonfinite validation metric", "rows": rows}, None
    with torch.no_grad():
        loss = F.cross_entropy(model(x), y, reduction="none")
        if second is not None:
            loss = (loss+F.cross_entropy(second(x), y, reduction="none"))/2
        scores = -loss
        if method == "classwise_small_loss":
            scores = classwise_scores(loss, y)
        elif method == "confidence_tracking_toy":
            z = confidence_gap_z(torch.stack(history), y)
            # Diagnostic ranking only; the training rule above is exact union.
            normalized_z = torch.sigmoid(z)
            loss_ranks = classwise_scores(loss, torch.zeros_like(y))
            scores = torch.maximum(normalized_z, loss_ranks)
    result = {"status": "success", "epochs_completed": config["epochs"],
        "validation_macro_f1": rows[-1]["validation_macro_f1"], "rows": rows,
        "loop_seconds": time.perf_counter()-loop_begin, "selector_seconds": selector_seconds,
        "updates": updates, "state_bytes": tensor_bytes(optimizer.state)+(tensor_bytes(optimizer2.state) if optimizer2 else 0),
        "parameter_bytes": tensor_bytes(list(model.parameters()))+(tensor_bytes(list(second.parameters())) if second else 0),
        "history_bytes": tensor_bytes(history), "actual_keep": actual_mask.numpy(), "scores": scores.numpy()}
    return result, model
