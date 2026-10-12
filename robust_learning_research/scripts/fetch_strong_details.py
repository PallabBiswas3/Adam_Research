"""Additional pinned audit assets; never execute unlicensed DSS files."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"strong-baseline-20261012"
PIN = "ebbe5dd705584814d1214fe6cc1f1fb40658cf58"


if __name__ == "__main__":
    records = []
    for path in ["model/PreResNet.py", "algorithm/dynamic_sample_selection.py"]:
        url = f"https://raw.githubusercontent.com/Aliinton/DSS/{PIN}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        destination = OUT/"raw"/"sources"/"dss"/path
        destination.parent.mkdir(exist_ok=True, parents=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(content)
        records.append({"path": path, "url": url, "revision": PIN, "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content), "reuse": "inspection only"})
    (OUT/"additional_source_audit.json").write_text(json.dumps(records, indent=2))
