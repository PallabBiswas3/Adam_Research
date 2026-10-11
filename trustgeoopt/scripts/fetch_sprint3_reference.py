"""Download a pinned, unmodified authors' reference and optional public data.

No package installation or execution of downloaded code occurs here.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url, limit=50_000_000):
    with urlopen(Request(url, headers={"User-Agent": "TrustGeoOpt-research"}), timeout=60) as response:
        payload = response.read(limit+1)
    if len(payload) > limit:
        raise ValueError("download size limit exceeded")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashion", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    vendor = root/"src/trustgeoopt/optimizers/vendor"
    vendor.mkdir(exist_ok=True)
    if (vendor/"soap.py").exists():
        raise FileExistsError("reference already exists; preserve its pin")
    commit = json.loads(fetch("https://api.github.com/repos/nikhilvyas/SOAP/commits/main"))["sha"]
    sources = {}
    for name in ("soap.py", "LICENSE"):
        url = f"https://raw.githubusercontent.com/nikhilvyas/SOAP/{commit}/{name}"
        payload = fetch(url)
        (vendor/name).write_bytes(payload)
        sources[name] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
    (vendor/"__init__.py").write_text('"""Unmodified external references; see provenance.json and LICENSE."""\n')
    (vendor/"provenance.json").write_text(json.dumps({"repository": "https://github.com/nikhilvyas/SOAP",
        "commit": commit, "files": sources, "status": "authors preliminary reference; not Meta distributed Shampoo"}, indent=2))
    print("SOAP reference pinned:", commit, flush=True)
    if args.fashion:
        folder = root/"data/FashionMNIST/raw"
        folder.mkdir(parents=True, exist_ok=True)
        data = {}
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
            # Official dataset repository; torchvision validates these same resources.
            url = f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}.gz"
            payload = fetch(url)
            target = folder/name
            if target.exists():
                raise FileExistsError(target)
            target.write_bytes(gzip.decompress(payload))
            data[name] = {"url": url, "compressed_sha256": hashlib.sha256(payload).hexdigest(),
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            print("Fetched", name, flush=True)
        (root/"research/sprint3_data_provenance.json").write_text(json.dumps(data, indent=2))
