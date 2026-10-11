import argparse

import _bootstrap  # noqa: F401
from trustgeoopt.training.synthetic_runner import run_synthetic
from trustgeoopt.utils.logging import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic/smoke.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(run_synthetic(load_config(args.config), args.output, args.resume))


if __name__ == "__main__":
    main()
