"""Validate frozen source/configuration provenance and run completeness."""
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

script = Path(__file__).resolve()
project = next(p for p in script.parents if (p/"pyproject.toml").exists())
sys.path.insert(0, str(project/"src"))
from trustgeoopt.training.sprint4 import select_policies
from trustgeoopt.utils.logging import write_json


def audit(path, create_manifest=False):
    state = json.loads((path/"campaign.json").read_text())
    protocol = yaml.safe_load((path/"protocol.yaml").read_text())
    assert protocol == state["protocol"]
    archive = path/"source_snapshot.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == state["snapshot_sha256"]
    with zipfile.ZipFile(archive) as z:
        digest = hashlib.sha256()
        for name in sorted(n for n in z.namelist() if n.endswith(".py") and
                           (n.startswith("src/") or n.startswith("scripts/"))):
            digest.update(name.replace("/", "\\").encode())  # captured Windows relative Path naming
            digest.update(z.read(name))
        assert digest.hexdigest() == state["environment"]["source_sha256"], "snapshot source hash mismatch"
    rows = [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]
    assert len(rows) == state["trials"] <= protocol["max_trials"] <= 1000
    assert state["experiment_seconds"] <= protocol["max_experiment_seconds"] <= 1800
    assert set(protocol["development_seeds"]) == set(range(50, 60))
    assert set(protocol["evaluation_seeds"]) == set(range(60, 70))
    assert {r["config"]["seed"] for r in rows if r["config"]["split"] == "tuning"} == set(range(50, 60))
    expected_counts = {"pilot": 4, "probe_pilot": 2, "tuning": 640, "evaluation": 240, "estimators": 60, "neural": 20}
    assert dict(Counter(r["config"]["stage"] for r in rows)) == expected_counts
    seen = set()
    for row in rows:
        c = row["config"]
        digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        assert digest == row["config_sha256"]
        assert digest not in seen
        seen.add(digest)
        folder = path/row["path"]
        assert c == json.loads((folder/"config.json").read_text())
        assert row == json.loads((folder/"metadata.json").read_text())
        assert row["source_sha256"] == state["environment"]["source_sha256"]
        if row["status"] == "success":
            if c["stage"] in {"tuning", "evaluation", "pilot"}:
                trajectory = pd.read_csv(folder/"trajectory.csv")
                assert len(trajectory) == c["steps"] == row["steps_completed"]
                assert trajectory.step.tolist() == list(range(1, c["steps"]+1))
            if c["stage"] == "estimators":
                measurements = pd.read_csv(folder/"estimators.csv")
                assert set(measurements.phase) == set(protocol["estimators"]["phases"])
                assert len(measurements) == 33
                assert row["steps_completed"] == protocol["estimators"]["steps"]
            if c["stage"] == "neural":
                measurements = pd.read_csv(folder/"estimators.csv")
                assert set(measurements.epoch) == set(protocol["neural"]["phases"])
                assert set(measurements.layer) == {0, 1}
                assert len(measurements) == 66
                assert row["epochs_completed"] == protocol["neural"]["epochs"]
    selection, _ = select_policies([r for r in rows if r["config"]["stage"] == "tuning"], protocol)
    assert selection == json.loads((path/"processed"/"selection.json").read_text())
    for row in rows:
        c = row["config"]
        if c["stage"] == "evaluation":
            method = c["optimizer"]["name"]
            assert c["optimizer"] == selection[method]["optimizer"]
            assert c["seed"] in protocol["evaluation_seeds"]
    earlier = json.loads((path/"raw"/"earlier_inventory.json").read_text())
    for entry in earlier:
        assert hashlib.sha256((project/entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    report = {"attempts": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
        "stage_counts": expected_counts, "earlier_artifacts_verified": len(earlier),
        "source_snapshot_verified": True, "config_hashes_verified": True,
        "selection_recomputed_without_evaluation": True, "successful_horizons_verified": True,
        "experiment_seconds": state["experiment_seconds"]}
    if create_manifest:
        write_json(path/"processed"/"audit.json", report)
        manifest = [{"path": p.relative_to(path).as_posix(), "bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in sorted(path.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"]
        write_json(path/"artifact_manifest.json", manifest)
    if (path/"artifact_manifest.json").exists():
        manifest = json.loads((path/"artifact_manifest.json").read_text())
        for entry in manifest:
            assert hashlib.sha256((path/entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        report["manifest_files_verified"] = len(manifest)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--create-manifest", action="store_true")
    args = parser.parse_args()
    audit(Path(args.campaign), args.create_manifest)
