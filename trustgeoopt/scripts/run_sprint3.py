"""Immutable bounded CPU campaigns. Run from the project directory."""
import argparse
import _bootstrap  # noqa: F401
from trustgeoopt.training.sprint2 import create_campaign
from trustgeoopt.training.sprint3 import run_stage
from trustgeoopt.utils.logging import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic/sprint3.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=["init", "replication", "stochastic", "geometry", "neural"])
    args = parser.parse_args()
    if args.stage == "init":
        create_campaign(load_config(args.config), args.output)
    else:
        run_stage(args.output, args.stage)
