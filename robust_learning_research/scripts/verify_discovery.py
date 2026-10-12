"""Read-only evidence verification and preservation of every TrustGeoOpt byte."""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from robust_learning.data import make_problem
from robust_learning.metrics import selection_metrics, operating_curve
from robust_learning.training import run_training, evaluate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output, preservation):
    started = time.perf_counter()
    state = json.loads((output/"campaign.json").read_text())
    runs = [json.loads(line) for line in (output/"runs.jsonl").read_text().splitlines()]
    assert len(runs) == state["attempts"] == 245
    assert state["stages"] == ["pilot", "development", "evaluation"]
    assert state["experiment_seconds"] <= state["protocol"]["max_seconds"]
    assert all(r["status"] == "success" for r in runs)
    assert len({r["path"] for r in runs}) == len(runs)
    assert len({r["source_sha256"] for r in runs}) == 1
    snapshot = output/"source_snapshot.zip"
    assert sha(snapshot) == state["snapshot_sha256"]
    digest = hashlib.sha256()
    with zipfile.ZipFile(snapshot) as archive:
        sources = sorted(n for n in archive.namelist() if n.endswith(".py") and n.split("/")[0] in {"src", "scripts"})
        for name in sources:
            content = archive.read(name)
            digest.update(name.encode()); digest.update(content)
            # Later added analysis files are separate; frozen existing files must match.
            assert (ROOT/name).read_bytes() == content, name
    assert digest.hexdigest() == state["source_sha256"]
    evaluation_count = 0
    development = []
    for r in runs:
        config = r["config"]
        assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == r["config_sha256"]
        folder = output/r["path"]
        assert json.loads((folder/"config.json").read_text()) == config
        assert json.loads((folder/"metadata.json").read_text()) == r
        if config["split"] in {"pilot", "development"}:
            assert "test" not in r and "actual_selection" not in r and "realized_noise_rate" not in r
            assert not (folder/"operating_curve.json").exists()
            if config["split"] == "development":
                development.append(r)
            continue
        evaluation_count += 1
        regime = config["regime"]
        problem = make_problem(config["seed"], config["seed"]+1000, counts=regime["counts"],
            noise_kind=regime["noise_kind"], noise_rate=config["nominal_noise_rate"], task_seed=regime["task_seed"], std=regime["std"])
        with np.load(folder/"data.npz") as stored:
            for key, value in [("train_x", problem.train.x), ("observed_y", problem.train.observed_y),
                               ("true_y", problem.oracle.true_y), ("flipped", problem.oracle.flipped),
                               ("ambiguity", problem.oracle.ambiguity), ("validation_x", problem.validation.x),
                               ("validation_y", problem.validation.y), ("test_x", problem.test.x), ("test_y", problem.test.y)]:
                np.testing.assert_array_equal(stored[key], value)
        with np.load(folder/"selection.npz") as selected:
            assert selection_metrics(selected["keep"], problem.oracle) == r["actual_selection"]
            curve = operating_curve(selected["scores"], problem.oracle, config["fractions"])
            assert curve == json.loads((folder/"operating_curve.json").read_text())
    assert evaluation_count == 200 and len(development) == 40
    chosen = json.loads((output/"processed"/"selection.json").read_text())
    for method in state["protocol"]["methods"]:
        choices = []
        for rate in state["protocol"]["rates"]:
            values = [r["validation_macro_f1"] for r in development if r["config"]["method"] == method and r["config"]["lr"] == rate]
            assert len(values) == 4
            choices.append((np.mean(values), rate))
        assert max(choices)[1] == chosen[method]
    # Deterministic replay of one sealed evaluation; no policy/model selection.
    reference = next(r for r in runs if r["config"]["split"] == "evaluation" and r["config"]["method"] == "ce")
    c = reference["config"]; regime = c["regime"]
    problem = make_problem(c["seed"], c["seed"]+1000, counts=regime["counts"], noise_kind=regime["noise_kind"],
        noise_rate=c["nominal_noise_rate"], task_seed=regime["task_seed"], std=regime["std"])
    replay, model = run_training(problem.train, problem.validation, c, time.perf_counter()+20)
    assert replay["validation_macro_f1"] == reference["validation_macro_f1"]
    assert evaluate(model, problem.test) == reference["test"]
    before = json.loads(preservation.read_text())["files"]
    prior = {item["path"]: item for item in before}
    trust = ROOT.parent/"trustgeoopt"
    current = {p.relative_to(ROOT.parent).as_posix(): p for p in trust.rglob("*") if p.is_file()}
    assert set(prior) == set(current), "TrustGeoOpt file set changed"
    trust_digest = hashlib.sha256()
    for name in sorted(prior):
        original = prior[name]
        assert current[name].stat().st_size == original["bytes"] and sha(current[name]) == original["sha256"], name
        trust_digest.update((name+"\0"+original["sha256"]+"\0"+str(original["bytes"])+"\n").encode())
    # Raw files stay local and immutable; publish hashes, summaries and plots.
    raw_records = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                   for p in sorted((output/"raw").rglob("*")) if p.is_file()]
    (output/"processed"/"raw_manifest.json").write_text(json.dumps(raw_records, indent=2))
    source_records = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)}
                      for folder in ("src", "scripts", "tests", "configs") for p in sorted((ROOT/folder).rglob("*"))
                      if p.is_file() and p.suffix in {".py", ".json"}]
    report = {"all_checks_passed": True, "attempts": len(runs), "evaluation_successes": evaluation_count,
        "frozen_source_sha256": state["source_sha256"], "trustgeoopt_files_checked": len(prior),
        "trustgeoopt_all_files_unchanged": True, "trustgeoopt_inventory_sha256": trust_digest.hexdigest(),
        "raw_files_hashed": len(raw_records), "deterministic_evaluation_replay": reference["config_sha256"],
        "verification_seconds": time.perf_counter()-started, "current_source_files": source_records}
    (output/"processed"/"verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "current_source_files"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--preservation", default="research/trustgeoopt_preservation_before.json")
    args = parser.parse_args()
    verify(ROOT/args.output, ROOT/args.preservation)
