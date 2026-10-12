"""Run immutable bounded CPU stages from the project directory."""
import argparse

import _bootstrap  # noqa: F401
from trustgeoopt.training.sprint4 import initialize, run_stage
from trustgeoopt.utils.logging import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic/sprint4.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=["init", "pilot", "tuning", "evaluation", "estimators", "neural"])
    args = parser.parse_args()
    if args.stage == "init":
        print(initialize(load_config(args.config), args.output))
    else:
        run_stage(args.output, args.stage)
