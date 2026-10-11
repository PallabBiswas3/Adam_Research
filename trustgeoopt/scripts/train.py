import argparse

import _bootstrap  # noqa: F401
from trustgeoopt.training.trainer import train
from trustgeoopt.utils.logging import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic/mlp_smoke.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-test", action="store_true", help="use once after validation model selection")
    args = parser.parse_args()
    print(train(load_config(args.config), args.output, args.resume, args.evaluate_test))


if __name__ == "__main__":
    main()
