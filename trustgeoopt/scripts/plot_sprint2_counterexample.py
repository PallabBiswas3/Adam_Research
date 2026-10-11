import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import _bootstrap  # noqa: F401
from trustgeoopt.utils.logging import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    path = Path(args.results)
    source = path/"published_counterexample/cycles.csv"
    frame = pd.read_csv(source)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for method, group in frame.groupby("method"):
        axes[0].plot(group.t, group.average_regret, label=method)
        axes[1].plot(group.t, group.x_next, label=method)
    axes[0].axhline(2/3, color="black", linestyle="--", linewidth=1, label="Adam theorem bound")
    axes[0].set(xscale="log", xlabel="online rounds T", ylabel="pre-update average regret R_T/T")
    axes[1].set(xscale="log", xlabel="online rounds T (cycle end)", ylabel="next iterate")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    (path/"figures").mkdir(exist_ok=True)
    fig.savefig(path/"figures/published_counterexample.png", dpi=180)
    fig.savefig(path/"figures/published_counterexample.pdf")
    plt.close(fig)
    write_json(path/"processed/analysis_provenance_counterexample_plot.json", dict(
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=f"python scripts/plot_sprint2_counterexample.py --results {path.as_posix()}"))
