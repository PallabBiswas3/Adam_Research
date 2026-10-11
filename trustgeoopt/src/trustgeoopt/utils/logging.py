import csv
import hashlib
import json
import logging
import math
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .hardware import environment_metadata, process_peak_rss_bytes, synchronize

LOGGER = logging.getLogger("trustgeoopt")


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


class RunLog:
    """Immutable directory per attempt, streaming CSV, explicit status/errors."""

    def __init__(self, config, output_root, device):
        self.config, self.device = config, torch.device(device)
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        self.run_id = f"{config.get('name', 'run')}-{digest}-{uuid.uuid4().hex[:8]}"
        self.path = Path(output_root)/"raw"/self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        self.metadata = dict(experiment_id=self.run_id, config=config,
                             created_utc=datetime.now(timezone.utc).isoformat(),
                             environment=environment_metadata(device), seed=config["seed"],
                             dataset=config.get("dataset", config.get("objective")),
                             optimizer=config["optimizer"], status="running", errors=[], warnings=[],
                             timing_scope="Timed execution after model/dataset/optimizer initialization; imports and setup excluded")
        write_json(self.path/"metadata.json", self.metadata)
        self.handle = (self.path/"metrics.csv").open("w", newline="", encoding="utf-8")
        self.writer = None
        self.rows = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        tracemalloc.start()
        synchronize(self.device)
        self.start = time.perf_counter()
        LOGGER.info("Started %s", self.run_id)

    def elapsed(self):
        synchronize(self.device)
        return time.perf_counter()-self.start

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.handle.flush()
        self.rows += 1

    def finish(self, status="success", summary=None, error=None):
        runtime = self.elapsed()
        self.metadata.update(status=status, runtime_seconds=runtime, rows=self.rows,
            cpu_process_peak_rss_bytes=process_peak_rss_bytes(),
            python_peak_traced_bytes=tracemalloc.get_traced_memory()[1],
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            summary=summary or {})
        if error is not None:
            self.metadata["errors"].append(str(error))
        self.metadata["warnings"].append("CPU RSS is process-lifetime high-water; traced memory excludes tensor storage.")
        tracemalloc.stop()
        self.handle.close()
        write_json(self.path/"metadata.json", self.metadata)
        LOGGER.info("Finished %s: %s (%.3fs)", self.run_id, status, runtime)
        return self.path
