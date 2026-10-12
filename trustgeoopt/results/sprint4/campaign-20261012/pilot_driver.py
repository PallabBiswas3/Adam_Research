"""Recorded pilot command; uses unchanged frozen experiment source."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

project = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project/"src"))
from trustgeoopt.training.sprint4 import estimator_trial, neural_trial
from trustgeoopt.utils.logging import write_json

path = Path(__file__).parent
state = json.loads((path/"campaign.json").read_text())
if "probe_pilot" in state["stages"]:
    raise ValueError("pilot already executed")
started = time.perf_counter()
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
    try:
        result = (neural_trial if kind == "neural" else estimator_trial)(
            config, folder, started+1800-state["experiment_seconds"], protocol)
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
