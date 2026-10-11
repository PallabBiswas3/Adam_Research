"""Audit provenance, pairing, frozen selections, and preservation of tracked history."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from trustgeoopt.training.sprint2 import select_rates
from trustgeoopt.utils.logging import write_json


def audit(path):
    path = Path(path)
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    state = json.loads((path/"campaign.json").read_text())
    trials = [json.loads(line) for line in (path/"trials.jsonl").read_text().splitlines()]
    assert len(trials) == state["trials"] <= state["protocol"]["max_trials"]
    assert state["experiment_seconds"] <= state["protocol"]["max_experiment_seconds"]
    # The timing run preceded only this audit tool. Preserve and hash-check that
    # exact source tree, excluding the newly added audit script.
    timing_snapshot = path/"source_snapshot_correction_timing.zip"
    timing = json.loads((path/"correction_timing/summary.json").read_text())
    if not timing_snapshot.exists():
        all_sources = sorted(list((project/"src").rglob("*.py"))+list((project/"scripts").glob("*.py")))
        matched = False
        for exclude_audit in (False, True):
            candidate = hashlib.sha256()
            for source in all_sources:
                if exclude_audit and source.resolve() == Path(__file__).resolve():
                    continue
                candidate.update(str(source.relative_to(project)).encode())
                candidate.update(source.read_bytes())
            if candidate.hexdigest() == timing["environment"]["source_sha256"]:
                matched = True
                break
        if not matched:
            raise ValueError("Timing source has changed: recover its source snapshot before auditing")
        with zipfile.ZipFile(timing_snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("src", "scripts", "tests", "configs"):
                for source in sorted((project/folder).rglob("*")):
                    if source.suffix in {".py", ".yaml"} and not (exclude_audit and source.resolve() == Path(__file__).resolve()):
                        archive.write(source, source.relative_to(project))
    timing_digest = hashlib.sha256()
    with zipfile.ZipFile(timing_snapshot) as archive:
        sources = [n for n in archive.namelist() if n.endswith(".py") and
                   (n.startswith("src/") or (n.startswith("scripts/") and n.count("/") == 1))]
        for source in sorted(sources, key=lambda n: str(Path(n))):
            timing_digest.update(str(Path(source)).encode())
            timing_digest.update(archive.read(source))
    assert timing_digest.hexdigest() == timing["environment"]["source_sha256"]
    write_json(path/"processed/source_provenance_correction_timing.json", dict(
        source_snapshot=timing_snapshot.name,
        snapshot_sha256=hashlib.sha256(timing_snapshot.read_bytes()).hexdigest(),
        source_sha256=timing_digest.hexdigest(),
        note="Source tree preserved after timing and verified against its original source hash; the later audit tool is excluded if necessary"))
    snapshots = {}
    for trial in trials:
        meta = json.loads((path/"raw"/trial["run_id"]/"metadata.json").read_text())
        config = meta["config"]
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert digest == meta["config_sha256"]
        assert meta["status"] == trial["status"]
        assert config == trial["config"]
        environment = meta["environment"]
        name = environment.get("source_snapshot", "source_snapshot.zip")
        if name not in snapshots:
            archive_path = path/name
            snapshot_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            digest = hashlib.sha256()
            with zipfile.ZipFile(archive_path) as archive:
                sources = [n for n in archive.namelist() if n.endswith(".py") and
                           (n.startswith("src/") or (n.startswith("scripts/") and n.count("/") == 1))]
                for source in sorted(sources, key=lambda n: str(Path(n))):
                    digest.update(str(Path(source)).encode())
                    digest.update(archive.read(source))
            snapshots[name] = dict(zip_sha256=snapshot_digest, source_sha256=digest.hexdigest())
        assert environment["source_sha256"] == snapshots[name]["source_sha256"], trial["run_id"]
        expected_zip = environment.get("snapshot_sha256", state["snapshot_sha256"])
        assert expected_zip == snapshots[name]["zip_sha256"]
        if trial["status"] == "success":
            assert meta["rows"] == config["steps"]
    for prefix, filename in ((False, "selected_rates.json"), (True, "correction_selected_rates.json")):
        tuning = [trial for trial in trials if trial["config"]["split"] == "tuning" and
                  trial["config"]["optimizer"]["name"].startswith("adamw_") == prefix]
        selected = select_rates(tuning, state["protocol"]["development_seeds"])
        assert selected == json.loads((path/"processed"/filename).read_text())
        split = "correction_evaluation" if prefix else "evaluation"
        for trial in trials:
            config = trial["config"]
            if config["split"] == split:
                key = f"{config['regime']}/{config['optimizer']['name']}"
                assert config["optimizer"]["lr"] == selected[key]
                assert config["seed"] in state["protocol"]["evaluation_seeds"]
    git = ["git", "-c", f"safe.directory={workspace.as_posix()}"]
    listing = subprocess.run(git+["ls-files", "--stage", "--", "trustgeoopt/results"],
                             cwd=workspace, text=True, capture_output=True, check=True).stdout
    historical = []
    for line in listing.splitlines():
        info, filename = line.split("\t", 1)
        _, expected, stage = info.split()
        assert stage == "0"
        contents = (workspace/filename).read_bytes()
        # hash-object applies checkout filters (e.g. Windows CRLF) before comparison.
        actual = subprocess.run(git+["hash-object", "--path", filename, "--", filename],
                                cwd=workspace, text=True, capture_output=True, check=True).stdout.strip()
        assert actual == expected, f"Historical result changed: {filename}"
        historical.append(dict(path=filename, git_blob=expected, sha256=hashlib.sha256(contents).hexdigest()))
    output = path/"artifact_manifest.json"
    artifacts = {file.relative_to(path).as_posix(): dict(bytes=file.stat().st_size,
                  sha256=hashlib.sha256(file.read_bytes()).hexdigest()) for file in sorted(path.rglob("*"))
                  if file.is_file() and file != output}
    report = dict(trials=len(trials), success=sum(t["status"] == "success" for t in trials),
                  failed=sum(t["status"] == "failed" for t in trials),
                  experiment_seconds=state["experiment_seconds"],
                  source_snapshots=snapshots, historical_tracked_results_verified=len(historical),
                  historical_results=historical, artifacts=artifacts,
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note="Tracked historical result Git blobs verified after checkout filters; new trial hashes, steps and frozen selections verified")
    write_json(output, report)
    print(json.dumps({key: report[key] for key in ("trials", "success", "failed", "experiment_seconds", "historical_tracked_results_verified")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    audit(args.results)
