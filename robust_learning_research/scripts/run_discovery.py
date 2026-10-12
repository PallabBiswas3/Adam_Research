"""Frozen short CPU discovery campaign; no TrustGeoOpt imports or mutations."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from robust_learning.data import make_problem
from robust_learning.metrics import selection_metrics, operating_curve
from robust_learning.training import evaluate, run_training


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def source_hash():
    digest = hashlib.sha256()
    for path in sorted([*(ROOT/"src").rglob("*.py"), *(ROOT/"scripts").glob("*.py")]):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def initialize(protocol, output):
    if set(protocol["development_seeds"])&set(protocol["evaluation_seeds"]):
        raise ValueError("seed splits must be disjoint")
    if not 0 < protocol["max_seconds"] <= 300 or not 0 < protocol["max_trials"] <= 260:
        raise ValueError("initial CPU limits exceeded")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("raw", "processed", "plots"):
        (output/folder).mkdir()
    dump(output/"protocol.json", protocol)
    snapshot = output/"source_snapshot.zip"
    with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts", "tests", "configs", "research"):
            for item in sorted((ROOT/folder).rglob("*")):
                if item.is_file() and item.suffix in {".py", ".json", ".md", ".csv"} and item.name != "trustgeoopt_preservation_before.json":
                    archive.write(item, item.relative_to(ROOT))
    dump(output/"campaign.json", {"protocol": protocol, "experiment_seconds": 0., "attempts": 0,
        "stages": [], "source_sha256": source_hash(), "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"}})


def run(output, stage):
    state = json.loads((output/"campaign.json").read_text())
    protocol = state["protocol"]
    marker = output/(stage+".started.json")
    if stage in state["stages"] or marker.exists():
        raise ValueError("stage already started; preserve attempts")
    if source_hash() != state["source_sha256"]:
        raise ValueError("source changed since freeze")
    dump(marker, {"stage": stage, "source_sha256": state["source_sha256"]})
    start = time.perf_counter()
    deadline = start+max(0, protocol["max_seconds"]-state["experiment_seconds"])
    records = []

    def attempt(config):
        if state["attempts"] >= protocol["max_trials"]:
            raise ValueError("attempt cap reached")
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        folder = output/"raw"/(stage+"-"+digest[:16])
        folder.mkdir(exist_ok=False)
        dump(folder/"config.json", config)
        before = time.perf_counter()
        result = {"status": "budget_not_started"}
        if before < deadline:
            try:
                regime = config["regime"]
                data = make_problem(config["seed"], config["seed"]+1000, counts=regime["counts"],
                    noise_kind=regime["noise_kind"], noise_rate=protocol["nominal_noise_rate"], task_seed=regime["task_seed"], std=regime["std"])
                np.savez_compressed(folder/"data.npz", train_x=data.train.x, observed_y=data.train.observed_y,
                    true_y=data.oracle.true_y, flipped=data.oracle.flipped, ambiguity=data.oracle.ambiguity,
                    validation_x=data.validation.x, validation_y=data.validation.y, test_x=data.test.x, test_y=data.test.y)
                result, model = run_training(data.train, data.validation, config, deadline)
                rows = result.pop("rows", [])
                if rows:
                    with (folder/"trajectory.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                if result["status"] == "success":
                    scores, keep = result.pop("scores"), result.pop("actual_keep")
                    # No oracle or TEST evaluation before development policy freeze.
                    if config["split"] in {"evaluation", "held_out"}:
                        test = evaluate(model, data.test)
                        result.update(test=test, actual_selection=selection_metrics(keep, data.oracle),
                            realized_noise_rate=float(data.oracle.flipped.mean()))
                    np.savez_compressed(folder/"selection.npz", scores=scores, keep=keep)
                    if config["split"] in {"evaluation", "held_out"}:
                        dump(folder/"operating_curve.json", operating_curve(scores, data.oracle, protocol["fractions"]))
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        result.update(config=config, config_sha256=digest, source_sha256=state["source_sha256"],
                      path=folder.relative_to(output).as_posix(), total_seconds=time.perf_counter()-before)
        dump(folder/"metadata.json", result)
        with (output/"runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False)+"\n")
        records.append(result)
        state["attempts"] += 1
        dump(output/"campaign.json", {**state, "experiment_seconds": state["experiment_seconds"]+time.perf_counter()-start,
                                      "interrupted_stage": stage})
        if len(records)%10 == 0:
            print(stage, len(records), round(time.perf_counter()-start, 2), flush=True)

    try:
        if stage == "pilot":
            for method in protocol["methods"]:
                attempt({**protocol, "method": method, "seed": 201, "lr": .003, "epochs": 2,
                         "regime": protocol["regimes"][0], "split": "pilot"})
        elif stage == "development":
            for seed in protocol["development_seeds"]:
                for regime in [r for r in protocol["regimes"] if r["name"] in protocol["development_regimes"]]:
                    development_regime = {**regime, "task_seed": 100, "split": "development"}
                    for rate in protocol["rates"]:
                        for method in protocol["methods"]:
                            attempt({**protocol, "method": method, "seed": seed, "lr": rate,
                                     "regime": development_regime, "split": "development"})
            selected = {}
            for method in protocol["methods"]:
                candidates = []
                for rate in protocol["rates"]:
                    group = [r for r in records if r["config"]["method"] == method and r["config"]["lr"] == rate]
                    if len(group) == 4 and all(r["status"] == "success" for r in group):
                        candidates.append((np.mean([r["validation_macro_f1"] for r in group]), rate))
                if not candidates:
                    raise RuntimeError("no complete development candidate for "+method)
                selected[method] = float(max(candidates)[1])
            dump(output/"processed"/"selection.json", selected)
        elif stage == "evaluation":
            selection = json.loads((output/"processed"/"selection.json").read_text())
            for seed in protocol["evaluation_seeds"]:
                for regime in protocol["regimes"]:
                    for method in protocol["methods"]:
                        attempt({**protocol, "method": method, "seed": seed, "lr": selection[method],
                                 "regime": regime, "split": regime["split"]})
        else:
            raise ValueError(stage)
        state["stages"].append(stage)
    finally:
        state["experiment_seconds"] += time.perf_counter()-start
        dump(output/"campaign.json", state)
    print("Completed", stage, state["attempts"], round(state["experiment_seconds"], 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/discovery_cpu.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["init", "pilot", "development", "evaluation"], required=True)
    args = parser.parse_args()
    if args.stage == "init":
        initialize(json.loads((ROOT/args.config).read_text()), ROOT/args.output)
    else:
        run(ROOT/args.output, args.stage)
