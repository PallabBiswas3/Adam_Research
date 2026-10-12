"""Prepare isolated benchmark arrays using torchvision's verified CIFAR archive."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from robust_learning.strong.cifar import make_splits, corrupt


if __name__ == "__main__":
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="fresh results directory; existing artifacts are preserved")
    args = parser.parse_args()
    config = json.loads((ROOT/"configs"/"strong_baseline_pilot.json").read_text())
    destination = args.output.resolve() if args.output else ROOT/"results"/config["name"]
    if ROOT not in destination.parents:
        raise ValueError("output must stay within robust_learning_research")
    out = destination/"raw"/"cifar10"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("preserve existing prepared artifacts")
    out.mkdir(exist_ok=True, parents=True)
    train = CIFAR10(str(ROOT/"data"/"strong_cifar10"), train=True, download=True)
    test = CIFAR10(str(ROOT/"data"/"strong_cifar10"), train=False, download=False)
    labels = np.asarray(train.targets, dtype=np.int64)
    split = make_splits(labels, config["split_seed"], config["validation_per_class"], config["imbalance_factor"])
    ids = split["train"]
    observed, flipped, intervention = corrupt(labels[ids], config["noise_seed"], config["noise_rate"], config["noise_kind"])
    np.savez_compressed(out/"observed_training.npz", images=train.data[ids], observed_y=observed, ids=ids)
    np.savez_compressed(out/"evaluation_oracle.npz", true_y=labels[ids], flipped=flipped, intervention=intervention)
    np.savez_compressed(out/"clean_validation.npz", images=train.data[split["validation"]], y=labels[split["validation"]], ids=split["validation"])
    np.savez_compressed(out/"prior_calibration.npz", y=labels[split["prior"]], ids=split["prior"])
    np.savez_compressed(out/"clean_test.npz", images=test.data, y=np.asarray(test.targets), ids=np.arange(10000)+50000)
    # Hash archive verified by torchvision; retain provenance, not distributed images.
    archive = ROOT/"data"/"strong_cifar10"/CIFAR10.filename
    hashes = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob("*.npz"))]
    metadata = {"dataset": "CIFAR-10", "source_url": CIFAR10.url, "torchvision_archive_md5": CIFAR10.tgz_md5,
        "archive_md5": hashlib.md5(archive.read_bytes()).hexdigest(), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "train_count": len(ids), "validation_count": len(split["validation"]), "prior_count": len(split["prior"]),
        "test_count": len(test.targets), "prior_from_clean_calibration": (np.bincount(labels[split["prior"]], minlength=10)/len(split["prior"])).tolist(),
        "config_sha256": hashlib.sha256((ROOT/"configs"/"strong_baseline_pilot.json").read_bytes()).hexdigest(),
        "artifacts": hashes, "preparation_seconds": time.perf_counter()-start,
        "training_oracle_sealed": True, "note": "Construction reads clean labels; training interface/files contain no clean training labels or flags"}
    assert metadata["archive_md5"] == CIFAR10.tgz_md5
    (out.parent.parent/"dataset_provenance.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != "artifacts"}, indent=2))
