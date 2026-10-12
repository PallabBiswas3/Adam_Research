"""Freeze preservation inventories and inspect all prior saved evidence."""
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"results"/"strong-baseline-20261012"


def inventory(root, relative_root):
    return [{"path": p.relative_to(relative_root).as_posix(), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob("*")) if p.is_file()]


if __name__ == "__main__":
    started = time.perf_counter()
    # Read ALL pre-existing saved artifacts, including ignored raw files.
    prior = inventory(ROOT/"results", ROOT)
    inspected = []
    for record in prior:
        path = ROOT/record["path"]
        info = {"path": record["path"], "bytes": record["bytes"], "sha256": record["sha256"]}
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8")); info["parsed"] = "json"
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            info["rows"] = len(rows)
            info["status_counts"] = {s: sum(r.get("status") == s for r in rows) for s in {r.get("status") for r in rows}}
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                info["rows"] = sum(1 for _ in csv.DictReader(f))
        inspected.append(info)
    trust = inventory(ROOT.parent/"trustgeoopt", ROOT.parent)
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/"raw").mkdir()
    (OUTPUT/"raw"/"preservation_before.json").write_text(json.dumps({"prior_results": prior, "trustgeoopt": trust}, indent=2))
    (OUTPUT/"prior_evidence_review.json").write_text(json.dumps({"date": "2026-10-12", "artifacts_read": len(prior),
        "trustgeoopt_files": len(trust), "prior_results_inventory_sha256": hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest(),
        "trustgeoopt_inventory_sha256": hashlib.sha256(json.dumps(trust, sort_keys=True).encode()).hexdigest(),
        "artifact_details": inspected, "seconds": time.perf_counter()-started}, indent=2))
    print(json.dumps({"prior_artifacts_read": len(prior), "trustgeoopt_files": len(trust), "seconds": time.perf_counter()-started}))
