"""Standalone geometry cost measurement, not optimizer speedup evidence."""
import argparse
import time

import torch

import _bootstrap  # noqa: F401
from trustgeoopt.geometry import GradientHistory, diagonal_low_rank_solve
from trustgeoopt.utils.logging import RunLog
from trustgeoopt.utils.hardware import synchronize
from trustgeoopt.utils.seeds import set_seed


def benchmark(output="results/geometry", device="cpu", seed=0):
    set_seed(seed)
    config = dict(name="geometry-cost", seed=seed, device=device,
                  dataset=dict(name="random-gradient-history", dimensions=[128, 1024], ranks=[2, 4], repetitions=10),
                  optimizer=dict(name="standalone-woodbury-no-optimizer"))
    log = RunLog(config, output, device)
    try:
        for dimension in [128, 1024]:
            for rank in [2, 4]:
                history = GradientHistory(rank, 2*rank)
                for _ in range(2*rank):
                    history.update(torch.randn(dimension, dtype=torch.float64, device=device))
                diagonal = torch.rand(dimension, dtype=torch.float64, device=device)+0.1
                rhs = torch.randn(dimension, dtype=torch.float64, device=device)
                factor, _ = history.factor()
                # Warm up kernels/allocators; warmup cost remains in total runtime.
                diagonal_low_rank_solve(diagonal, factor, rhs)
                for repeat in range(10):
                    g = torch.randn(dimension, dtype=torch.float64, device=device)
                    synchronize(device)
                    start = time.perf_counter()
                    history.update(g)
                    factor, tail = history.factor()
                    synchronize(device)
                    refresh_seconds = time.perf_counter()-start
                    start = time.perf_counter()
                    solution = diagonal_low_rank_solve(diagonal, factor, rhs)
                    synchronize(device)
                    solve_seconds = time.perf_counter()-start
                    residual = (diagonal*solution + factor @ (factor.T @ solution)-rhs).norm()/rhs.norm()
                    log.write(dict(dimension=dimension, rank=rank, repeat=repeat, refresh_seconds=refresh_seconds,
                        solve_seconds=solve_seconds, discarded_trace=tail, relative_residual=float(residual),
                        retained_state_bytes=(history.history.numel()+factor.numel()+diagonal.numel())*8,
                        elapsed_seconds=log.elapsed()))
        return log.finish(summary=dict(note="Random history cost only; no end-to-end optimizer comparison"))
    except Exception as error:
        log.finish(status="failed", error=error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/geometry")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(benchmark(args.output, args.device))
