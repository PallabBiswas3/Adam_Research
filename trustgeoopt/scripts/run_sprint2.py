"""Create once, then execute bounded stages; no existing run is overwritten."""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from trustgeoopt.training.sprint2 import create_campaign, execute_stage, published_counterexample
from trustgeoopt.utils.logging import load_config, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic/sprint2.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["init", "baseline", "reliability", "corrections", "counterexample"], required=True)
    args = parser.parse_args()
    if args.stage == "init":
        create_campaign(load_config(args.config), args.output)
    elif args.stage == "counterexample":
        path = Path(args.output)
        destination = path/"published_counterexample"
        destination.mkdir(exist_ok=False)
        state = json.loads((path/"campaign.json").read_text())
        config = state["protocol"]["published_counterexample"]
        write_json(destination/"provenance.json", dict(config=config, environment=state["environment"],
                  snapshot_sha256=state["snapshot_sha256"]))
        pd.DataFrame(published_counterexample(config)).to_csv(destination/"cycles.csv", index=False)
    else:
        execute_stage(args.output, args.stage)
