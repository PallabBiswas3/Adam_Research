"""Development-only fitting and paired whole-seed resampling for Sprint 2."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .metrics import prediction_metrics


def lead_labels(frame, lead, transitions=()):
    if lead < 0:
        raise ValueError("nonnegative lead required")
    result = frame.copy()
    result["label"] = result.harmful.shift(-lead)
    mask = result.label.notna()
    for transition in transitions:
        mask &= ~((result.step < transition) & (result.step+lead >= transition))
    return result.loc[mask].copy()


def load_campaign_frames(path, lead):
    frames = []
    for filename in sorted((Path(path)/"raw").glob("*/metadata.json")):
        meta = json.loads(filename.read_text())
        config = meta["config"]
        if not config["split"].startswith("reliability_") or meta["status"] != "success":
            continue
        objective = config["objective"]
        transitions = [objective[key] for key in ("change_step", "noise_change_step") if key in objective]
        frame = lead_labels(pd.read_csv(filename.parent/"metrics.csv"), lead, transitions)
        frame["seed"] = config["seed"]
        frame["run_id"] = meta["experiment_id"]
        frame["regime"] = config["regime"]
        frame["split"] = config["split"]
        frame["phase_bin"] = pd.cut(frame.phase, [0, 0.2, 0.6, 1], labels=["early", "middle", "late"], include_lowest=True).astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError("no complete reliability runs")
    return pd.concat(frames, ignore_index=True)


def fit_logistic(x, y, weights=None, regularization=0.01):
    """Frozen L2 penalty; intercept unpenalized, scaling learned on development."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ValueError("finite features and both development classes required")
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-8)
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = w/w.sum()

    def objective(beta):
        z = design @ beta
        loss = np.dot(w, np.logaddexp(0, z)-y*z)+0.5*regularization*np.dot(beta[1:], beta[1:])
        gradient = design.T @ (w*(expit(z)-y))
        gradient[1:] += regularization*beta[1:]
        return loss, gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return dict(mean=mean.tolist(), scale=scale.tolist(), beta=result.x.tolist(),
                penalty=regularization, converged=bool(result.success), iterations=int(result.nit))


def predict_logistic(model, x):
    x = (np.asarray(x, float)-model["mean"])/model["scale"]
    return expit(model["beta"][0]+x @ np.asarray(model["beta"])[1:])


def average_precision(y, score):
    return prediction_metrics(y, score, 0.5)["average_precision"]


def paired_cluster_interval(frame, first, second, replicates=1000, seed=20261012, metric="ap"):
    """Resample seeds: each duplicate carries ALL its runs and ordered steps.

    Model fits remain fixed. No iid step bootstrap and no lost method pairing.
    """
    if metric not in {"ap", "brier"}:
        raise ValueError("unknown metric")
    seeds = sorted(frame.seed.unique())
    if len(seeds) < 2:
        raise ValueError("need at least two seed clusters")
    codes = pd.Categorical(frame.seed, categories=seeds).codes
    y = frame.label.to_numpy()
    ranks = []
    for column in (first, second):
        score = frame[column].to_numpy()
        order = np.argsort(-score, kind="stable")
        boundaries = np.r_[np.flatnonzero(np.diff(score[order])), len(order)-1]
        ranks.append((y[order], codes[order], boundaries))
    losses = (frame[second].to_numpy()-y)**2-(frame[first].to_numpy()-y)**2

    def difference(counts):
        if metric == "brier":
            weights = counts[codes]
            return float(np.dot(weights, losses)/weights.sum())
        ap = []
        for ordered_y, ordered_codes, boundaries in ranks:
            weights = counts[ordered_codes]
            cumulative_positive = np.cumsum(weights*ordered_y)[boundaries]
            if cumulative_positive[-1] == 0:
                return None
            cumulative_count = np.cumsum(weights)[boundaries]
            precision = np.divide(cumulative_positive, cumulative_count,
                                  out=np.zeros_like(cumulative_positive, dtype=float), where=cumulative_count > 0)
            ap.append(float(np.dot(precision, np.diff(np.r_[0, cumulative_positive]))/cumulative_positive[-1]))
        return ap[0]-ap[1]

    rng = np.random.default_rng(seed)
    estimates, cache = [], {}
    for _ in range(replicates):
        counts = np.bincount(rng.integers(len(seeds), size=len(seeds)), minlength=len(seeds))
        key = tuple(counts.tolist())
        if key not in cache:
            cache[key] = difference(counts)
        value = cache[key]
        if value is not None:
            estimates.append(value)
    return dict(estimate=difference(np.ones(len(seeds), dtype=int)), ci95=np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else None,
                seed_clusters=len(seeds), valid_resamples=len(estimates), metric=metric,
                note="Whole-seed percentile bootstrap, conditional on frozen development fit/configurations")


def feature_frame(data):
    output = data.copy()
    for original, new in (("gradient_norm", "log_gradient"), ("proposed_step_norm", "log_step"),
                          ("learning_rate", "log_lr"), ("noise_scale", "log_noise"),
                          ("oracle_directional_curvature", "log_curvature"), ("distance_before", "log_distance")):
        output[new] = np.log1p(output[original].clip(lower=0))
    output["observed_linear_change"] = -output.gradient_norm*output.proposed_step_norm*output.observed_descent_alignment
    output["curvature_change"] = 0.5*output.oracle_directional_curvature*output.proposed_step_norm**2
    output["observed_geometry_margin"] = (output.observed_linear_change+output.curvature_change)/output.objective_before.clip(lower=1e-8)
    output["observed_geometry_margin"] = np.arcsinh(output.observed_geometry_margin)
    output["phase_squared"] = output.phase**2
    return output


def evaluate_campaign(path, protocol):
    reports, prediction_frames = {}, []
    common = ["log_gradient", "log_step", "observed_descent_alignment", "log_lr", "log_noise", "phase", "phase_squared"]
    adjusted = common+["log_curvature", "log_distance", "observed_geometry_margin"]
    feature_sets = {"first_order": common, "adjusted": adjusted,
                    "adjusted_alignment": adjusted+["momentum_alignment"],
                    "adjusted_history": adjusted+["history_consistency"],
                    "adjusted_both": adjusted+["momentum_alignment", "history_consistency"]}
    raw = {"norm": ("gradient_norm", 1), "alignment": ("momentum_alignment", -1),
           "history": ("history_consistency", -1), "preconditioned_direction": ("observed_descent_alignment", -1),
           "oracle_geometry": ("oracle_quadratic_change", 1)}
    for lead in (0, 1, 3, 5):
        data = feature_frame(load_campaign_frames(path, lead))
        original_counts = data.groupby(["split", "regime"]).size().to_dict()
        required = sorted(set(sum(feature_sets.values(), [])+[column for column, _ in raw.values()]))
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required+["label"])
        dev = data[data.split == "reliability_development"].copy()
        evaluation = data[data.split == "reliability_evaluation"].copy()
        if set(dev.seed) & set(evaluation.seed):
            raise ValueError("development and evaluation seeds overlap")
        if dev.empty or evaluation.empty:
            raise ValueError("missing independent split")
        models, thresholds = {}, {}
        weights = 1/dev.groupby("run_id").run_id.transform("size").to_numpy()
        for name, features in feature_sets.items():
            model = fit_logistic(dev[features], dev.label, weights)
            model["features"] = features
            models[name] = model
            dev[name] = predict_logistic(model, dev[features])
            evaluation[name] = predict_logistic(model, evaluation[features])
        for name, (column, sign) in raw.items():
            dev[name] = sign*dev[column]
            evaluation[name] = sign*evaluation[column]
        for name in list(models)+list(raw):
            thresholds[name] = float(np.quantile(dev.loc[dev.label == 0, name], 0.95))
        by_subset = {}
        for subset, mask in (("seen", evaluation.regime.isin(protocol["regimes"])),
                             ("unseen", evaluation.regime.isin(protocol["held_out_regimes"]))):
            subset_data = evaluation.loc[mask].copy()
            metrics = {name: prediction_metrics(subset_data.label, subset_data[name], thresholds[name])
                       for name in thresholds}
            for name in thresholds:
                per_run = {run: prediction_metrics(g.label, g[name], thresholds[name]) for run, g in subset_data.groupby("run_id")}
                defined = [m["average_precision"] for m in per_run.values() if m["average_precision"] is not None]
                metrics[name]["macro_run_ap"] = float(np.mean(defined)) if defined else None
                metrics[name]["runs_with_positive_labels"] = len(defined)
                metrics[name]["total_runs"] = len(per_run)
                metrics[name]["per_run"] = per_run
            increments = {}
            for first, second in (("alignment", "norm"), ("history", "norm"),
                                  ("preconditioned_direction", "alignment"), ("adjusted_alignment", "adjusted"),
                                  ("adjusted_history", "adjusted"), ("adjusted_both", "adjusted")):
                key = f"{first}-minus-{second}"
                increments[key] = paired_cluster_interval(subset_data, first, second, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
                if first in models:
                    increments[key]["brier_improvement"] = paired_cluster_interval(subset_data, first, second,
                        protocol["bootstrap_replicates"], protocol["bootstrap_seed"], "brier")
            strata = {}
            for column in ("regime", "phase_bin", "learning_rate", "noise_scale"):
                strata[column] = {str(key): {name: prediction_metrics(g.label, g[name], thresholds[name])
                                           for name in thresholds} for key, g in subset_data.groupby(column)}
            by_subset[subset] = dict(metrics=metrics, increments=increments, strata=strata,
                                    runs=int(subset_data.run_id.nunique()), seed_clusters=int(subset_data.seed.nunique()))
        reports[str(lead)] = dict(subsets=by_subset, models=models, thresholds=thresholds,
                                 development_runs=sorted(dev.run_id.unique()), evaluation_runs=sorted(evaluation.run_id.unique()),
                                 evaluation_proposal_max_error=float(evaluation.proposal_error.max()),
                                 retained_samples=len(evaluation), label="same-objective actual future loss increase",
                                 common_complete_rows=True,
                                 rows_before_angle_complete_mask={f"{split}/{regime}": int(count) for (split, regime), count in original_counts.items()},
                                 rows_after_angle_complete_mask={f"{split}/{regime}": int(count) for (split, regime), count in data.groupby(["split", "regime"]).size().items()})
        evaluation["lead"] = lead
        prediction_frames.append(evaluation)
    return reports, pd.concat(prediction_frames, ignore_index=True)
