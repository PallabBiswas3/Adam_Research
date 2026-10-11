import argparse
import copy
import itertools
from pathlib import Path

import _bootstrap  # noqa: F401
from trustgeoopt.training.synthetic_runner import run_synthetic
from trustgeoopt.training.trainer import train
from trustgeoopt.utils.logging import load_config, write_json


def run_suite(path, output, max_runs=None):
    suite = load_config(path)
    base = load_config(suite["base"])
    dimensions = list(suite.get("grid", {}))
    choices = [suite["grid"][key] for key in dimensions]
    products = list(itertools.product(*choices)) if choices else [()]
    combinations = [dict(dict(zip(dimensions, product)), **case)
                    for product in products for case in suite.get("cases", [{}])]
    seeds = suite.get("seeds", [0])
    count = len(combinations)*len(seeds)
    # Explicit CLI limit keeps large planned grids from launching accidentally.
    limit = max_runs if max_runs is not None else suite.get("max_runs", 50)
    if count > limit:
        raise ValueError(f"Suite has {count} runs, exceeds explicit max-runs {limit}")
    outcomes = []
    for seed in seeds:
        for combination in combinations:
            config = copy.deepcopy(base)
            config["seed"] = seed
            for key, value in combination.items():
                target = config
                bits = key.split(".")
                for bit in bits[:-1]:
                    target = target[bit]
                target[bits[-1]] = value
            config["name"] = suite.get("name", "suite")
            try:
                runner = run_synthetic if "objective" in config else train
                result = runner(config, output)
                outcomes.append(dict(seed=seed, settings=combination, status="success", path=str(result)))
            except Exception as error:
                outcomes.append(dict(seed=seed, settings=combination, status="failed", error=str(error)))
    dest = Path(output)/"processed"
    dest.mkdir(parents=True, exist_ok=True)
    import uuid
    write_json(dest/f"suite-{uuid.uuid4().hex[:8]}.json", dict(suite=suite, outcomes=outcomes))
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/synthetic/pilot_suite.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    results = run_suite(args.suite, args.output, args.max_runs)
    if any(r["status"] != "success" for r in results):
        raise SystemExit("Some trials failed; all outcomes recorded")


if __name__ == "__main__":
    main()
