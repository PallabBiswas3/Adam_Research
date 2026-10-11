"""Audit immutable attempts, frozen selection, snapshots and preserved Sprint 2."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path, PureWindowsPath

import _bootstrap  # noqa: F401
from trustgeoopt.training.sprint3 import select_candidates
from trustgeoopt.utils.logging import write_json


def audit(path):
    state = json.loads((path/"campaign.json").read_text())
    protocol = state["protocol"]
    assert state["experiment_seconds"] <= protocol["max_experiment_seconds"]+1
    assert state["trials"] <= protocol["max_trials"]
    assert state["stages"] == ["replication", "stochastic", "geometry", "neural"]
    snapshot = path/"source_snapshot.zip"
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == state["snapshot_sha256"]
    with zipfile.ZipFile(snapshot) as archive:
        sources = [n for n in archive.namelist() if n.endswith(".py") and (n.startswith("src/") or (n.startswith("scripts/") and n.count("/") == 1))]
        windows = state["environment"]["platform"].startswith("Windows")
        style = (lambda name: str(PureWindowsPath(name))) if windows else (lambda name: name)
        digest = hashlib.sha256()
        for name in sorted(sources, key=style):
            digest.update(style(name).encode())
            digest.update(archive.read(name))
        assert digest.hexdigest() == state["environment"]["source_sha256"]
    records = [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]
    replication = [json.loads(line) for line in (path/"replication.jsonl").read_text().splitlines()]
    assert len(records)+len(replication) == state["trials"]
    for r in records:
        folder = path/"raw"/PureWindowsPath(r["path"]).name
        c = json.loads((folder/"config.json").read_text())
        assert c == r["config"]
        assert hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest() == r["config_sha256"]
        assert json.loads((folder/"metadata.json").read_text()) == r
        assert r["source_sha256"] == state["environment"]["source_sha256"]
        assert c["seed"] in protocol["development_seeds" if c["split"] == "tuning" else "evaluation_seeds"]
        if r["status"] == "success":
            with (folder/"metrics.csv").open() as handle:
                assert sum(1 for _ in handle)-1 == (c["epochs"] if c["stage"] == "neural" else c["steps"])
    for stage in ("stochastic", "geometry", "neural"):
        tuning = [r for r in records if r["config"]["stage"] == stage and r["config"]["split"] == "tuning"]
        selection = select_candidates(tuning, protocol["development_seeds"], "validation_loss" if stage == "neural" else "tail_loss")
        assert selection == json.loads((path/"processed"/f"{stage}_selection.json").read_text())
        for r in records:
            c = r["config"]
            if c["stage"] != stage or c["split"] == "tuning":
                continue
            regime = c["regime"]
            if c["split"] == "held_out":
                regime = "isotropic_noise/"+c["objective"]["rotation"] if stage == "geometry" else c["model"]+"/task700"
            assert c["optimizer"] == selection[f"{regime}/{c['optimizer']['name']}"]
    root = Path(__file__).resolve().parents[1]
    old = root/"results/sprint2/campaign-20261012"
    old_manifest = json.loads((old/"artifact_manifest.json").read_text())
    historical_count = 0
    for name, expected in old_manifest["artifacts"].items():
        assert hashlib.sha256((old/name).read_bytes()).hexdigest() == expected["sha256"], f"Sprint 2 artifact changed: {name}"
        historical_count += 1
    # Replay comparison uses old frozen records; none enter new model selection.
    old_trials = [json.loads(line) for line in (old/"trials.jsonl").read_text().splitlines()]
    max_replay_difference = 0
    for r in replication:
        c = r["config"]
        previous = next(x for x in old_trials if x["config"]["split"] == "evaluation" and x["config"]["regime"] == "noisy" and x["config"]["seed"] == c["seed"] and x["config"]["optimizer"]["name"] == c["optimizer"]["name"])
        assert r["status"] == "success"
        max_replay_difference = max(max_replay_difference, abs(r["tail_loss"]-previous["tail_loss"]))
    assert max_replay_difference < 1e-12
    workspace = root.parent
    git = ["git", "-c", f"safe.directory={workspace.as_posix()}"]
    changes = subprocess.run(git+["diff", "7587551", "--name-only", "--", "trustgeoopt/results/sprint2"], cwd=workspace, capture_output=True, text=True, check=True).stdout.strip()
    assert not changes, "tracked Sprint 2 results modified"
    manifest = json.loads((path/"artifact_manifest.json").read_text())
    for entry in manifest:
        assert hashlib.sha256((path/entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    report = {"attempts": state["trials"], "nonreplication_attempts": len(records),
        "status_counts": {status: sum(r["status"] == status for r in records+replication) for status in ("success", "failed", "budget_censored", "budget_not_started")},
        "experiment_seconds": state["experiment_seconds"], "sprint2_artifacts_verified": historical_count,
        "max_replay_tail_loss_difference": max_replay_difference, "manifest_entries_verified": len(manifest),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Prior artifacts read only; source/configuration hashes, complete horizons and tuning-only frozen selections verified"}
    write_json(path/"processed/audit.json", report)
    # Incorporate the audit without circularly hashing the manifest itself.
    manifest = [{"path": source.relative_to(path).as_posix(), "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()} for source in sorted(path.rglob("*"))
        if source.is_file() and source.name != "artifact_manifest.json"]
    write_json(path/"artifact_manifest.json", manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    audit(Path(parser.parse_args().output))
