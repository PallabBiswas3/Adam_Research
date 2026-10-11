"""Inventory source artifacts and actual run metadata without rewriting raw evidence."""
import json
from pathlib import Path


def inventory(root):
    directories = ["src", "scripts", "tests", "configs", "experiments", "research", "paper"]
    artifacts = sorted(str(p.relative_to(root)).replace("\\", "/")
        for directory in directories for p in (root/directory).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "artifact_manifest.json")
    artifacts.extend(["README.md", "pyproject.toml", "requirements.txt", "LICENSE", ".gitignore"])
    runs = []
    for path in sorted((root/"results").rglob("metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        runs.append(dict(path=str(path.parent.relative_to(root)).replace("\\", "/"),
                         id=meta["experiment_id"], status=meta["status"], seed=meta["seed"],
                         rows=meta["rows"], source_sha256=meta["environment"]["source_sha256"]))
    report = dict(date="2026-10-12", artifacts=sorted(artifacts), source_artifact_count=len(artifacts),
                  runs=runs, raw_run_count=len(runs), failures=sum(r["status"] != "success" for r in runs),
                  note="Raw runs are engineering/pilot evidence; not all are independent efficacy trials.")
    (root/"research"/"artifact_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"artifacts", "runs"}}, indent=2))


if __name__ == "__main__":
    inventory(Path(__file__).resolve().parents[1])
