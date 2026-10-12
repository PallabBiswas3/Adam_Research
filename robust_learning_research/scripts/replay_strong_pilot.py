"""Replay the frozen pilot in a separate prepared directory; no campaign mode."""
import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits
from run_strong_pilot import ROOT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if ROOT not in destination.parents:
        raise ValueError("output must stay within robust_learning_research")
    config = json.loads((ROOT/"configs"/"strong_baseline_pilot.json").read_text())
    if not (destination/"dataset_provenance.json").exists():
        raise FileNotFoundError("run prepare_cifar10.py --output on this fresh directory first")
    with threadpool_limits(limits=config["threads"]):
        run(config, destination)
