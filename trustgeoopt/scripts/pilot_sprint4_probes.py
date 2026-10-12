"""Recorded pilot command; uses unchanged frozen experiment source."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project/"src"))
from trustgeoopt.training.sprint4 import estimator_trial, neural_trial
from trustgeoopt.utils.hardware import environment_metadata
from trustgeoopt.utils.logging import write_json

def run(path):
    state = json.loads((path/"campaign.json").read_text())
    marker = path/"probe_pilot.started.json"
    if "probe_pilot" in state["stages"] or marker.exists():
        raise ValueError("pilot already started; preserve attempts")
    if environment_metadata("cpu")["source_sha256"] != state["environment"]["source_sha256"]:
        raise ValueError("source changed since campaign initialization")
    if state["trials"]+2 > state["protocol"]["max_trials"]:
        raise ValueError("pilot would exceed trial budget")
    write_json(marker, {"stage": "probe_pilot"})
    started = time.perf_counter()
    old_dtype = torch.get_default_dtype()
    for kind in ("estimators", "neural"):
        protocol = copy.deepcopy(state["protocol"])
        protocol["estimators"].update(steps=128, phases=[80, 128], paired_samples=8)
        protocol["neural"].update(epochs=1, phases=[0, 1], paired_samples=8, gradient_samples=8)
        config = {"stage": "probe_pilot", "kind": kind, "seed": 50, "split": "pilot",
                  "regime": "antialigned", "task_seed": 800, "pilot_protocol": protocol}
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        folder = path/"raw"/f"probe-pilot-{kind}-{digest[:16]}"
        folder.mkdir(exist_ok=False)
        write_json(folder/"config.json", config)
        torch.set_default_dtype(torch.float32 if kind == "neural" else torch.float64)
        before = time.perf_counter()
        deadline = started+protocol["max_experiment_seconds"]-state["experiment_seconds"]
        result = {"status": "budget_not_started"}
        if before < deadline:
            try:
                result = (neural_trial if kind == "neural" else estimator_trial)(config, folder, deadline, protocol)
                result.pop("metrics", None)
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        result.update(config=config, config_sha256=digest, path=folder.relative_to(path).as_posix(),
                      source_sha256=state["environment"]["source_sha256"], runtime_seconds=time.perf_counter()-before)
        write_json(folder/"metadata.json", result)
        with (path/"trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False)+"\n")
        state["trials"] += 1
        print(kind, result["status"], result.get("error"), result["runtime_seconds"], flush=True)
    state["experiment_seconds"] += time.perf_counter()-started
    state["stages"].append("probe_pilot")
    write_json(path/"campaign.json", state)
    torch.set_default_dtype(old_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    run(Path(args.campaign))
