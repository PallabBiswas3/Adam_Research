"""Pinned primary-source inspection; only MIT architecture/license are vendored."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"strong-baseline-20261012"
PINS = {"dss": ("Aliinton/DSS", "ebbe5dd705584814d1214fe6cc1f1fb40658cf58"),
        "dividemix": ("LiJunnan1992/DivideMix", "d9d3058fa69a952463b896f84730378cdee6ec39")}
ASSETS = {"dss": ["README.md", "requirements.txt", "train.py", "train_DSS_plus.py", "model/SampleSelector.py", "model/resnet.py",
                  "dataloader/dataloaders.py", "dataloader/cifar.py", "algorithm/selection.py"],
          "dividemix": ["LICENSE", "README.md", "PreResNet.py", "Train_cifar.py", "dataloader_cifar.py"]}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "RobustLearning-Research-Audit"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


if __name__ == "__main__":
    raw = OUT/"raw"/"sources"; raw.mkdir(exist_ok=False)
    records = []
    for method, (repo, pin) in PINS.items():
        (raw/method).mkdir()
        tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{pin}?recursive=1"))
        (raw/method/"tree.json").write_text(json.dumps(tree, indent=2))
        license_paths = [x["path"] for x in tree["tree"] if "license" in x["path"].lower()]
        paths = {x["path"] for x in tree["tree"]}
        assets = [p for p in ASSETS[method] if p in paths]
        if method == "dss":
            assets += [p for p in paths if p.startswith("script/") or (p.startswith("dataloader/") and p.endswith(".py"))]
        record = {"method": method, "repository": repo, "revision": pin, "license_paths": license_paths,
                  "tree_truncated": tree.get("truncated", False), "date": "2026-10-12", "assets": []}
        for path in sorted(set(assets+license_paths)):
            url = f"https://raw.githubusercontent.com/{repo}/{pin}/{path}"
            item = {"path": path, "url": url}
            try:
                content = fetch(url)
                destination = raw/method/path; destination.parent.mkdir(exist_ok=True, parents=True)
                destination.write_bytes(content)
                item.update(status="retrieved", bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            except Exception as error:
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
            record["assets"].append(item)
        records.append(record)
        (OUT/"source_audit.json").write_text(json.dumps(records, indent=2))
        print(method, "assets", len(record["assets"]), "licenses", license_paths, flush=True)
    # Explicitly permitted source reuse, including original attribution/license.
    source = raw/"dividemix"
    vendor = ROOT/"src"/"robust_learning"/"strong"/"vendor_dividemix"
    vendor.mkdir(parents=True, exist_ok=True)
    for name in ["LICENSE", "PreResNet.py"]:
        content = (source/name).read_bytes(); destination = vendor/name
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(content)
