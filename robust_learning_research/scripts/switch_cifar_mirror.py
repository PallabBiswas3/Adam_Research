"""Preserve slow partial download and install a checksum-identical archive."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import psutil
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"strong-baseline-20261012"
CACHE = ROOT/"data"/"strong_cifar10"


def checksum(path, name):
    digest = hashlib.new(name)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-download-pid", type=int, help="explicit task-owned download process only")
    args = parser.parse_args()
    if (OUT/"mirror_provenance.json").exists():
        raise FileExistsError("preserve existing mirror provenance")
    repository = "MIT-OL-AI-D/cifar-10-python"
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repository}", timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/cifar-10-python.tar.gz"
    destination = CACHE/"cifar-10-python.mirror.tar.gz"
    with urllib.request.urlopen(url, timeout=45) as response, destination.open("xb") as f:
        while chunk := response.read(1024*1024):
            f.write(chunk)
    assert checksum(destination, "md5") == CIFAR10.tgz_md5
    assert destination.stat().st_size == 170498071
    record = {"original_url": CIFAR10.url, "mirror_repository": repository, "mirror_revision": revision, "mirror_url": url,
              "md5": checksum(destination, "md5"), "sha256": checksum(destination, "sha256"), "bytes": destination.stat().st_size,
              "setup_note": "Original-host download was slow; S3 candidate returned404; verified identical archive replaces partial file",
              "stopped_download_processes": []}
    # No process discovery/automatic termination on replay. Require an explicit
    # PID and verify both its cwd and full script path belong to this workspace.
    if args.stop_download_pid is not None:
        process = psutil.Process(args.stop_download_pid)
        command = process.cmdline()
        directory = Path(process.cwd()).resolve()
        if len(command) < 2 or directory != ROOT.resolve():
            raise ValueError("PID is not this workspace's download process")
        script = Path(command[1])
        script = script.resolve() if script.is_absolute() else (directory/script).resolve()
        if script != (ROOT/"scripts"/"prepare_cifar10.py").resolve():
            raise ValueError("PID is not prepare_cifar10.py")
        process.terminate(); process.wait(timeout=15)
        record["stopped_download_processes"].append(process.pid)
    archive = CACHE/CIFAR10.filename
    if archive.exists():
        preserved = CACHE/"cifar-10-python.original-partial.tar.gz"
        if preserved.exists():
            raise FileExistsError(preserved)
        assert archive.resolve().is_relative_to(ROOT.resolve())
        assert preserved.resolve().is_relative_to(ROOT.resolve())
        record["preserved_original_partial"] = {"bytes": archive.stat().st_size, "sha256": checksum(archive, "sha256"), "path": preserved.relative_to(ROOT).as_posix()}
        archive.rename(preserved)
    destination.rename(archive)
    (OUT/"mirror_provenance.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
