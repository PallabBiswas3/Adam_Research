"""Mechanism pilots only; no unsupported TrustGeoOpt ablation is presented."""
import argparse

import _bootstrap  # noqa: F401
from run_benchmarks import run_suite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/synthetic/diagnostic_ablations.yaml")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    outcomes = run_suite(args.suite, args.output)
    if any(r["status"] != "success" for r in outcomes):
        raise SystemExit("Some ablations failed; inspect suite manifest")
