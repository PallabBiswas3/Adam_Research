import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

import _bootstrap  # noqa: F401


def generate(root):
    root = Path(root)
    for folder in ("figures", "tables", "processed"):
        (root/folder).mkdir(parents=True, exist_ok=True)
    summaries = []
    panels = {}
    for path in sorted((root/"raw").glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        config, summary = meta["config"], meta.get("summary", {})
        objective = config.get("objective", {}).get("kind", config.get("dataset", {}).get("name"))
        condition = config.get("objective", {}).get("condition")
        rotated = config.get("objective", {}).get("rotated", objective != "quadratic")
        summaries.append(dict(run_id=meta["experiment_id"], seed=meta["seed"], status=meta["status"],
            objective=objective, condition=condition, rotated=rotated,
            optimizer=meta["optimizer"]["name"], lr=meta["optimizer"].get("lr"),
            config_group=json.dumps({k: v for k, v in config.items() if k not in {"seed", "name"}}, sort_keys=True),
            final_loss=summary.get("final_loss", summary.get("best_validation_loss")),
            runtime_seconds=meta["runtime_seconds"], target_seconds=summary.get("target_seconds"),
            target_step=summary.get("target_step"), target_censored=summary.get("target_censored"),
            harmful_updates=summary.get("harmful_updates"), loss_spikes=summary.get("loss_spikes"),
            cpu_process_peak_rss_bytes=meta.get("cpu_process_peak_rss_bytes")))
        if meta["status"] != "success":
            continue
        frame = pd.read_csv(path.parent/"metrics.csv")
        x_col, y_col = ("step", "objective_after") if "step" in frame else ("epoch", "validation_loss")
        key = (objective, condition, rotated)
        group = summaries[-1]["config_group"]
        panels.setdefault(key, {}).setdefault(group, []).append(
            dict(frame=frame, x=x_col, y=y_col, optimizer=meta["optimizer"]["name"], lr=meta["optimizer"].get("lr")))
    if not summaries:
        raise ValueError("no saved runs found")
    individual = pd.DataFrame(summaries)
    individual.to_csv(root/"tables"/"individual_runs.csv", index=False)
    aggregates = []
    for group, rows in individual.groupby("config_group"):
        good = rows[(rows.status == "success") & rows.final_loss.notna()]
        n = len(good)
        mean = float(good.final_loss.mean()) if n else None
        std = float(good.final_loss.std(ddof=1)) if n > 1 else None
        half = float(t.ppf(0.975, n-1)*std/np.sqrt(n)) if n > 1 else None
        aggregates.append(dict(config_group=group, optimizer=rows.optimizer.iloc[0], objective=rows.objective.iloc[0],
            condition=rows.condition.iloc[0], rotated=rows.rotated.iloc[0], attempted=len(rows), n=n,
            failures=int((rows.status != "success").sum()), mean_final_loss=mean, std_final_loss=std,
            ci95_low=mean-half if half is not None else None, ci95_high=mean+half if half is not None else None,
            mean_runtime_seconds=float(good.runtime_seconds.mean()) if n else None,
            targets_reached=int(good.target_step.notna().sum()),
            note="Descriptive t CI over successful runs; report failures separately. No timing superiority claim."))
    pd.DataFrame(aggregates).to_csv(root/"tables"/"summary.csv", index=False)
    for (objective, condition, rotated), groups in panels.items():
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for index, (group, runs) in enumerate(sorted(groups.items())):
            color = plt.get_cmap("tab10")(index % 10)
            first = runs[0]
            # Align by actual step/epoch; keep failed runs in tables, never invent missing tails.
            series = [r["frame"].set_index(r["x"])[r["y"]] for r in runs]
            joined = pd.concat(series, axis=1)
            x = joined.index.to_numpy()+1
            mean, std = joined.mean(axis=1).to_numpy(), joined.std(axis=1, ddof=1).to_numpy()
            for run in runs:
                axis.plot(run["frame"][run["x"]]+1, run["frame"][run["y"]], color=color, alpha=0.15, linewidth=0.6)
            axis.plot(x, mean, color=color, linewidth=1.8,
                      label=f"{first['optimizer']} lr={first['lr']} (n={len(runs)})")
            if len(runs) > 1:
                axis.fill_between(x, np.maximum(mean-std, 0), mean+std, color=color, alpha=0.08)
        axis.set(xlabel=first["x"]+" (1-based)", ylabel=first["y"],
                 title=f"{objective}, condition={condition}: mean and SD; faint individual runs")
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        name = f"{objective}-k{condition}-rot{rotated}"
        fig.savefig(root/"figures"/f"{name}.png", dpi=160)
        fig.savefig(root/"figures"/f"{name}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    generate(args.results)
