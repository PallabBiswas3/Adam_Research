"""One-seed real-image kernel feasibility only; no benchmark quality claims."""
import gc
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from robust_learning.strong.cifar import ImageTrainingView, batch_images, image_transforms
from robust_learning.strong.equations import classwise_keep, fit_clean_probability, peer_policies
from robust_learning.strong.kernels import DSSState, dss_update, dividemix_update
from robust_learning.strong.vendor_dividemix.PreResNet import ResNet18


class Censored(RuntimeError):
    pass


def tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel()*obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def run(config, output):
    pilot = output/"pilot"; pilot.mkdir(exist_ok=False)
    torch.set_num_threads(config["threads"]); torch.set_num_interop_threads(1)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(config["threads"])
    process = psutil.Process()
    source = [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in sorted((ROOT/"src"/"robust_learning"/"strong").rglob("*")) if p.is_file() and p.suffix in {".py", ""}]
    source.append({"path": "scripts/run_strong_pilot.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    metadata = {"config": config, "config_sha256": hashlib.sha256((ROOT/"configs"/"strong_baseline_pilot.json").read_bytes()).hexdigest(),
        "source_files": source, "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": "cpu", "logical_cores": psutil.cpu_count(), "ram_mib": psutil.virtual_memory().total/2**20,
        "available_memory_before_mib": psutil.virtual_memory().available/2**20}, "quality_evaluation_executed": False,
        "paid_gpu_used": False, "full_campaign_executed": False, "oracle_opened_by_pilot": False}
    (pilot/"freeze.json").write_text(json.dumps(metadata, indent=2))
    # No evaluation_oracle.npz, clean_test.npz or validation labels are opened.
    data_file = output/"raw"/"cifar10"/"observed_training.npz"
    with np.load(data_file) as arrays:
        full_count = len(arrays["observed_y"])
        indices = np.random.default_rng(np.random.SeedSequence([config["pilot_seed"], 404])).choice(full_count, config["pilot_samples"], replace=False)
        view = ImageTrainingView(arrays["images"][indices].copy(), arrays["observed_y"][indices].copy(), arrays["ids"][indices].copy())
    with np.load(output/"raw"/"cifar10"/"prior_calibration.npz") as calibration:
        prior_counts = np.bincount(calibration["y"], minlength=10)
        estimated_prior = torch.tensor(prior_counts/prior_counts.sum(), dtype=torch.float32)
    weak, strong, inference = image_transforms()
    observed = torch.tensor(view.observed_y, dtype=torch.long)
    batch = config["pilot_batch_size"]
    global_begin = time.perf_counter()
    deadline = global_begin+config["max_training_seconds"]
    records = []

    def guard():
        memory = psutil.virtual_memory()
        if time.perf_counter() >= deadline:
            raise Censored("shared training wall-time cap")
        if process.memory_info().rss/2**20 > config["max_process_rss_mib"]:
            raise Censored("process RSS guard")
        if memory.available/2**20 < config["minimum_available_memory_mib"]:
            raise Censored("available-memory guard")

    def scan(model):
        model.eval(); losses = []
        begin = time.perf_counter()
        with torch.no_grad():
            for ids in np.array_split(np.arange(len(observed)), len(observed)//batch):
                guard(); x = batch_images(view, ids, inference)
                losses.extend(F.cross_entropy(model(x), observed[ids], reduction="none").tolist())
        return np.asarray(losses), time.perf_counter()-begin

    for method in config["methods"]:
        method_begin = time.perf_counter()
        record = {"method": method, "seed": config["pilot_seed"], "status": "started", "full_train_count": full_count,
                  "pilot_train_count": len(observed), "batch_size": batch, "warmup_updates": [], "main_updates": []}
        current_peak = [process.memory_info().rss]
        stopped = threading.Event()
        def monitor():
            while not stopped.wait(.02):
                current_peak[0] = max(current_peak[0], process.memory_info().rss)
        thread = threading.Thread(target=monitor, daemon=True); thread.start()
        models, optimizers, states = [], [], []
        model = optimizer = state = loss = losses = policies = None
        try:
            guard()
            plus = method == "dss_plus_estimated_prior"
            dual = plus or method == "dividemix"
            section = "dss_plus" if plus else "dividemix" if method == "dividemix" else "dss"
            settings = config[section]
            for net in range(2 if dual else 1):
                torch.manual_seed(config["pilot_seed"]+10000*net)
                models.append(ResNet18(num_classes=10))
                optimizers.append(torch.optim.SGD(models[-1].parameters(), lr=settings["lr"], momentum=.9, weight_decay=settings["weight_decay"]))
                prior = torch.ones(10)/10 if method == "dss_uniform" else estimated_prior
                states.append(DSSState(len(observed), 10, prior))
            torch.manual_seed(config["pilot_seed"]+5000)
            rng = np.random.default_rng(config["pilot_seed"]+7000)
            mix_rng = np.random.default_rng(config["pilot_seed"]+8000)
            generator = torch.Generator().manual_seed(config["pilot_seed"]+9000)
            for epoch in range(config["warmup_microepochs"]):
                order = rng.permutation(len(observed))
                for net, (model, optimizer, state) in enumerate(zip(models, optimizers, states)):
                    for ids in np.array_split(order, len(observed)//batch):
                        guard(); begin = time.perf_counter()
                        x = batch_images(view, ids, weak)
                        if method.startswith("dss"):
                            result = dss_update(model, optimizer, state, ids, x, observed[ids],
                                torch.ones(len(ids), dtype=torch.bool), torch.zeros(len(ids), 10, dtype=torch.bool),
                                batch_images(view, ids, strong) if plus else None, settings if plus else None, mix_rng, generator)
                        else:
                            model.train(); optimizer.zero_grad(set_to_none=True)
                            loss = F.cross_entropy(model(x), observed[ids])
                            loss.backward(); optimizer.step(); result = {"loss": float(loss.detach())}
                        record["warmup_updates"].append({"network": net, "microepoch": epoch, "seconds": time.perf_counter()-begin, **result})
                    if method.startswith("dss"):
                        state.finish_epoch()
            probabilities = []
            if method == "dividemix":
                scans = []
                for net, model in enumerate(models):
                    losses, seconds = scan(model)
                    before = time.perf_counter(); probabilities.append(fit_clean_probability(losses, config["pilot_seed"]+net))
                    scans.append({"network": net, "forward_scan_seconds": seconds, "gmm_seconds": time.perf_counter()-before,
                                  "clean_probability_fraction": float((probabilities[-1] > settings["clean_threshold"]).mean())})
                record["scan"] = scans
            elif method == "classwise_small_loss":
                loss, seconds = scan(models[0])
                keep_class = classwise_keep(torch.tensor(loss), observed, .5)
                record["scan"] = [{"forward_scan_seconds": seconds}]
            for iteration in range(config["timed_updates_per_method"]):
                guard(); begin = time.perf_counter(); details = []
                if method == "dividemix":
                    # Frozen peer posterior: both partitions were computed before either update.
                    for net in (0, 1):
                        peer_probability = probabilities[1-net]
                        labeled = np.where(peer_probability > settings["clean_threshold"])[0]
                        unlabeled = np.where(peer_probability <= settings["clean_threshold"])[0]
                        if not len(labeled) or not len(unlabeled):
                            raise Censored("empty GMM partition; no artificial fallback")
                        ix = rng.choice(labeled, batch, replace=len(labeled) < batch)
                        iu = rng.choice(unlabeled, batch, replace=len(unlabeled) < batch)
                        guard(); details.append(dividemix_update(models[net], models[1-net], optimizers[net], view, ix, iu,
                            peer_probability, weak, settings, mix_rng, generator, ramp_fraction=1.))
                else:
                    ids = rng.permutation(len(observed))[:batch]
                    policies = [s.policy(observed) for s in states]
                    if plus:
                        policies = peer_policies(*policies)
                    for net, (model, optimizer, state) in enumerate(zip(models, optimizers, states)):
                        guard(); x = batch_images(view, ids, weak)
                        if method.startswith("dss"):
                            keep, exclude = policies[net]
                            details.append(dss_update(model, optimizer, state, ids, x, observed[ids], keep[ids], exclude[ids],
                                batch_images(view, ids, strong) if plus else None, settings if plus else None, mix_rng, generator))
                            state.finish_epoch()  # partial microepoch: unvisited entries retain previous probabilities
                        else:
                            model.train(); optimizer.zero_grad(set_to_none=True)
                            losses = F.cross_entropy(model(x), observed[ids], reduction="none")
                            loss = losses.mean() if method == "ce" else losses[keep_class[ids]].mean() if keep_class[ids].any() else losses.sum()*0
                            loss.backward(); optimizer.step(); details.append({"loss": float(loss.detach())})
                if not all(np.isfinite(d["loss"]) for d in details):
                    raise FloatingPointError("nonfinite pilot update")
                record["main_updates"].append({"round": iteration, "seconds": time.perf_counter()-begin, "network_details": details})
            record.update(status="completed_kernel_pilot", networks=len(models), parameter_bytes=tensor_bytes([list(m.parameters()) for m in models]),
                sgd_state_bytes=tensor_bytes([o.state for o in optimizers]), history_bytes=tensor_bytes([s.history for s in states]) if method.startswith("dss") else 0)
        except Censored as error:
            record.update(status="censored", reason=str(error))
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            stopped.set(); thread.join()
            record.update(total_seconds=time.perf_counter()-method_begin, sampled_process_peak_rss_mib=current_peak[0]/2**20)
            (pilot/(method+".json")).write_text(json.dumps(record, indent=2, allow_nan=False))
            records.append(record)
            print(method, record["status"], round(record["total_seconds"], 2), "seconds", round(record["sampled_process_peak_rss_mib"], 1), "MiB", flush=True)
            del models, optimizers, states
            model = optimizer = state = loss = losses = policies = None
            gc.collect()
    metadata.update({"executed_method_records": records, "training_seconds": time.perf_counter()-global_begin,
        "OS_process_peak_rss_mib": getattr(process.memory_info(), "peak_wset", process.memory_info().rss)/2**20,
        "actual_pilot_subset_ids_sha256": hashlib.sha256(view.ids.tobytes()).hexdigest()})
    (pilot/"results.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    config = json.loads((ROOT/"configs"/"strong_baseline_pilot.json").read_text())
    with threadpool_limits(limits=config["threads"]):
        run(config, ROOT/"results"/config["name"])
