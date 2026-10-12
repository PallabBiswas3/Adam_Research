"""Read-only upstream/source/license inventory. Never execute downloaded code."""
import argparse
import hashlib
import http.client
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="results/reference-audit-20261012")
args = parser.parse_args()
DEST = ROOT/args.output
DEST.mkdir(parents=True, exist_ok=False)
(DEST/"raw").mkdir()
repositories = {"coteaching": "bhanML/Co-teaching", "confidence_tracking": "Aliinton/ConfidenceTracking",
                "dss": "Aliinton/DSS", "rolt": "weitongseu/RoLT", "slrlnl": "yangbo1973/SLRLNL",
                "dividemix": "LiJunnan1992/DivideMix", "pass": "arpit2412/PASS-NoisyLabel"}


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "RobustLearningResearch-audit", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return response.read()


records = []
for name, repository in repositories.items():
    record = {"name": name, "repository": repository, "retrieved_on": "2026-10-12"}
    try:
        metadata = json.loads(get(f"https://api.github.com/repos/{repository}"))
        branch = metadata["default_branch"]
        revision = json.loads(get(f"https://api.github.com/repos/{repository}/git/ref/heads/{branch}"))["object"]["sha"]
        tree = json.loads(get(f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1"))
        paths = [x["path"] for x in tree["tree"] if x["type"] == "blob"]
        license_paths = [p for p in paths if Path(p).name.lower().startswith(("license", "licence", "copying"))]
        record.update(revision=revision, default_branch=branch, api_license=metadata.get("license"),
                      license_files=license_paths, tree_truncated=tree.get("truncated"), files=[])
        folder = DEST/"raw"/name
        folder.mkdir()
        (folder/"tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
        # Bounded inspection of training/selection sources and READMEs.
        requested = [p for p in paths if p in {"README.md", "loss.py", "train.py", "utils.py", "main.py", "requirements.txt"}
                     or (p.startswith("algorithm/") and p.endswith(".py"))]
        for source in (license_paths+requested)[:18]:
            url = f"https://raw.githubusercontent.com/{repository}/{revision}/{source}"
            payload = get(url)
            target = folder/source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            record["files"].append({"path": source, "url": url, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        record["status"] = "inspected"
    except (urllib.error.URLError, OSError, http.client.HTTPException, KeyError, ValueError) as error:
        record.update(status="retrieval_failed", error=str(error))
    records.append(record)
    (DEST/"provenance.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(name, record.get("status"), record.get("revision"), record.get("license_files"), flush=True)
(DEST/"provenance.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print("Source inventory saved; downloaded source is inspection-only")
