"""Separate uninstrumented optimizer/correction timing; never a GPU claim."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from trustgeoopt.optimizers import make_optimizer
from trustgeoopt.utils.hardware import environment_metadata
from trustgeoopt.utils.logging import write_json
from trustgeoopt.utils.seeds import set_seed


def benchmark(destination):
    path = Path(destination)
    state = json.loads((path/"campaign.json").read_text())
    output = path/"correction_timing"
    output.mkdir(exist_ok=False)
    set_seed(40, True, 1)
    configuration = dict(seed=40, dimensions=[16, 16384], blocks=9, warmup_steps=30,
                         timed_steps=100, lr=.01, betas=[.9, .999], weight_decay=0.0,
                         step_cap=.1, safeguard_scale=200.0, gradient="precomputed standard normal; identical across methods")
    rows = []
    methods = ["adamw", "adamw_step_clip", "adamw_momentum_reset", "adamw_lr_safeguard"]
    begin = time.perf_counter()
    for dimension in configuration["dimensions"]:
        generator = torch.Generator().manual_seed(configuration["seed"]+dimension)
        gradients = torch.randn(130, dimension, generator=generator, dtype=torch.float64)
        for block in range(configuration["blocks"]):
            # Rotate method order to distribute drift/load effects.
            for method in methods[block % 4:]+methods[:block % 4]:
                p = torch.nn.Parameter(torch.ones(dimension, dtype=torch.float64))
                optimizer = make_optimizer([p], dict(name=method, lr=.01, weight_decay=0.0))
                for gradient in gradients[:30]:
                    p.grad = gradient
                    optimizer.step()
                start = time.perf_counter_ns()
                for gradient in gradients[30:]:
                    p.grad = gradient
                    optimizer.step()
                seconds = (time.perf_counter_ns()-start)*1e-9
                rows.append(dict(dimension=dimension, block=block, method=method,
                                 us_per_step=seconds/100*1e6,
                                 correction_events=getattr(optimizer, "correction_events", 0)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"blocks.csv", index=False)
    ratios = []
    for dimension, group in frame.groupby("dimension"):
        paired = group.pivot(index="block", columns="method", values="us_per_step")
        for method in methods:
            values = paired[method]/paired.adamw
            ratios.append(dict(dimension=int(dimension), method=method,
                               median_us_per_step=float(paired[method].median()),
                               median_paired_time_ratio=float(values.median()),
                               ratio_iqr=values.quantile([.25, .75]).tolist()))
    runtime = time.perf_counter()-begin
    write_json(output/"summary.json", dict(config=configuration, runtime_seconds=runtime,
                environment=environment_metadata("cpu"), results=ratios,
                timing_scope="Warmed CPU optimizer plus corrections and gradient assignment; no autograd, HVP, monitor, CSV, init or tracing in timed blocks"))
    state["experiment_seconds"] += runtime
    state["stages"].append("correction_timing")
    if state["experiment_seconds"] > state["protocol"]["max_experiment_seconds"]:
        raise RuntimeError("experiment cap exceeded")
    write_json(path/"campaign.json", state)
    print(json.dumps(ratios, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    benchmark(args.results)
