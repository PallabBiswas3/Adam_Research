"""Bounded, timing-only CIFAR-10 CUDA pilot for DSS+ and DivideMix.

This runner does not evaluate accuracy, read the test set, or launch a campaign.
The Kaggle notebook embeds this script and the independently implemented modules.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robust_learning.strong.cifar import ImageTrainingView, batch_images, corrupt, image_transforms, make_splits
from robust_learning.strong.equations import fit_clean_probability, peer_policies
from robust_learning.strong.kernels import DSSState, dividemix_update, dss_update
from robust_learning.strong.vendor_dividemix.PreResNet import ResNet18


class Censored(RuntimeError):
    pass


def batches(indices, batch_size):
    """Visit each ID exactly once, including the final short batch."""
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def guard(deadline):
    if time.perf_counter() >= deadline:
        raise Censored("shared 30-minute pilot deadline")


def measured(name, operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        result = operation()
        torch.cuda.synchronize()
        return {"stage": name, "seconds": time.perf_counter() - start,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                **result}
    except Exception as error:
        error.stage_measurement = {"stage": name, "status": "incomplete",
                                   "seconds_before_failure": time.perf_counter() - start,
                                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        raise


def load_training(cifar_root, config):
    # CIFAR10 checks the official batch MD5s. An attached, extracted dataset is
    # required; no download, network access, or dataset upload occurs here.
    dataset = CIFAR10(root=str(cifar_root), train=True, download=False)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    split = make_splits(labels, config["split_seed"], config["validation_per_class"], config["imbalance_factor"])
    ids = split["train"]
    observed, _, _ = corrupt(labels[ids], config["noise_seed"], config["noise_rate"], config["noise_kind"])
    prior_counts = np.bincount(labels[split["prior"]], minlength=10)
    prior = torch.as_tensor(prior_counts / prior_counts.sum(), dtype=torch.float32)
    view = ImageTrainingView(dataset.data[ids].copy(), observed.copy(), ids.copy())
    provenance = {"train_count": len(ids), "prior_calibration_count": len(split["prior"]),
                  "observed_labels_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                  "train_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                  "clean_calibration_prior": prior.tolist(),
                  "model_validation_opened": False, "test_opened": False,
                  "training_oracle_exposed_to_models": False}
    del dataset, labels, observed, ids, split
    return view, prior, provenance


def model_pair(seed, settings, device):
    models, optimizers = [], []
    for net in range(2):
        torch.manual_seed(seed + 10000 * net)
        model = ResNet18(num_classes=10).to(device)
        models.append(model)
        optimizers.append(torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=settings["momentum"],
                                          weight_decay=settings["weight_decay"]))
    return models, optimizers


def epoch_order(count, seed, phase):
    return np.random.default_rng(np.random.SeedSequence([seed, phase, 913])).permutation(count)


def warmup_dss(models, optimizers, states, view, weak, strong, settings, batch_size, seed, deadline, device, progress):
    order = epoch_order(len(view.ids), seed, 0)
    observed = torch.as_tensor(view.observed_y, dtype=torch.long)
    rng = np.random.default_rng(seed + 8000)
    generator = torch.Generator().manual_seed(seed + 9000)
    updates = 0
    for net in range(2):
        for ix in batches(order, batch_size):
            guard(deadline)
            x = batch_images(view, ix, weak).to(device)
            s = batch_images(view, ix, strong).to(device)
            dss_update(models[net], optimizers[net], states[net], ix, x, observed[ix].to(device),
                       torch.ones(len(ix), dtype=torch.bool), torch.zeros(len(ix), 10, dtype=torch.bool),
                       s, settings, rng, generator)
            updates += 1
            progress.update(network=net, network_updates=updates, images_processed=progress.get("images_processed", 0) + len(ix))
        states[net].finish_epoch()
    return {"images_per_network": len(order), "network_updates": updates, "completed_full_epoch": True,
            "host_history_mib": sum(h.numel() * h.element_size() for state in states for h in state.history) / 2**20}


def mature_dss(models, optimizers, states, view, weak, strong, settings, batch_size, seed, deadline, device, progress):
    order = epoch_order(len(view.ids), seed, 1)
    observed = torch.as_tensor(view.observed_y, dtype=torch.long)
    # Freeze both policies before either learner updates; each receives its peer's.
    policies = peer_policies(*(state.policy(observed) for state in states))
    rng = np.random.default_rng(seed + 8100)
    generator = torch.Generator().manual_seed(seed + 9100)
    updates, accepted = 0, []
    for net in range(2):
        keep, exclude = policies[net]
        accepted.append(int(keep.sum()))
        for ix in batches(order, batch_size):
            guard(deadline)
            x = batch_images(view, ix, weak).to(device)
            s = batch_images(view, ix, strong).to(device)
            dss_update(models[net], optimizers[net], states[net], ix, x, observed[ix].to(device),
                       keep[ix], exclude[ix], s, settings, rng, generator)
            updates += 1
            progress.update(network=net, network_updates=updates, images_processed=progress.get("images_processed", 0) + len(ix))
        states[net].finish_epoch()
    return {"images_per_network": len(order), "network_updates": updates,
            "accepted_labels_per_peer_policy": accepted, "completed_full_epoch": True,
            "ccs_history_epochs_before_mature_operation": 1,
            "host_history_mib": sum(h.numel() * h.element_size() for state in states for h in state.history) / 2**20}


def warmup_dividemix(models, optimizers, view, weak, batch_size, seed, deadline, device, progress):
    order = epoch_order(len(view.ids), seed, 0)
    observed = torch.as_tensor(view.observed_y, dtype=torch.long)
    updates = 0
    for net in range(2):
        model, optimizer = models[net], optimizers[net]
        for ix in batches(order, batch_size):
            guard(deadline)
            x = batch_images(view, ix, weak).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), observed[ix].to(device))
            loss.backward()
            optimizer.step()
            updates += 1
            progress.update(network=net, network_updates=updates, images_processed=progress.get("images_processed", 0) + len(ix))
    return {"images_per_network": len(order), "network_updates": updates, "completed_full_epoch": True}


def scan_dividemix(models, view, inference, batch_size, seed, deadline, device, progress):
    observed = torch.as_tensor(view.observed_y, dtype=torch.long)
    probabilities, fits = [], []
    for net, model in enumerate(models):
        model.eval()
        losses = np.empty(len(observed), dtype=np.float64)
        with torch.no_grad():
            for ix in batches(np.arange(len(observed)), batch_size):
                guard(deadline)
                x = batch_images(view, ix, inference).to(device)
                losses[ix] = F.cross_entropy(model(x), observed[ix].to(device), reduction="none").cpu().numpy()
                progress.update(network=net, images_scanned=progress.get("images_scanned", 0) + len(ix))
        start = time.perf_counter()
        probabilities.append(fit_clean_probability(losses, seed + net))
        fits.append({"network": net, "gmm_seconds": time.perf_counter() - start})
    return probabilities, {"images_per_network": len(observed), "fits": fits,
                           "completed_full_scan": True}


def mature_dividemix(models, optimizers, view, probabilities, weak, settings, batch_size, seed, deadline, progress):
    rng = np.random.default_rng(seed + 8200)
    generator = torch.Generator().manual_seed(seed + 9200)
    partitions = []
    for net in range(2):
        peer_prob = probabilities[1 - net]
        labeled = np.flatnonzero(peer_prob > settings["clean_threshold"])
        unlabeled = np.flatnonzero(peer_prob <= settings["clean_threshold"])
        if not len(labeled) or not len(unlabeled):
            raise Censored("empty GMM partition; no synthetic fallback")
        partitions.append((labeled, unlabeled, peer_prob))
    counts = []
    for net, (labeled, unlabeled, peer_prob) in enumerate(partitions):
        labeled_order = rng.permutation(labeled)
        unlabeled_order = rng.permutation(unlabeled)
        cursor, updates, used_unlabeled = 0, 0, 0
        for ix in batches(labeled_order, batch_size):
            guard(deadline)
            # Cycle only the unlabeled pool, as the official MixMatch loader does.
            if cursor + len(ix) > len(unlabeled_order):
                unlabeled_order = rng.permutation(unlabeled)
                cursor = 0
            iu = unlabeled_order[cursor:cursor + len(ix)]
            if len(iu) < len(ix):
                iu = rng.choice(unlabeled, len(ix), replace=True)
            cursor += len(ix)
            dividemix_update(models[net], models[1 - net], optimizers[net], view, ix, iu,
                             peer_prob, weak, settings, rng, generator, ramp_fraction=1.0)
            updates += 1
            used_unlabeled += len(iu)
            progress.update(network=net, network_updates=progress.get("network_updates", 0) + 1,
                            labeled_seen=progress.get("labeled_seen", 0) + len(ix),
                            unlabeled_draws=progress.get("unlabeled_draws", 0) + len(iu))
        counts.append({"network": net, "labeled_partition": len(labeled), "unlabeled_partition": len(unlabeled),
                       "labeled_seen_once": len(labeled_order), "unlabeled_draws": used_unlabeled,
                       "network_updates": updates})
    return {"partitions": counts, "completed_full_partition_epoch": True,
            "mature_loss_ramp_fraction": 1.0}


def run(args):
    if not 0 < args.time_limit_seconds <= 1800 or args.batch_size < 1:
        raise ValueError("time limit must be <=1800 seconds and batch size positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no CPU fallback")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    deadline = start + args.time_limit_seconds
    config = json.loads((ROOT / "configs" / "strong_baseline_pilot.json").read_text())
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config["pilot_seed"])
    torch.cuda.manual_seed_all(config["pilot_seed"])
    view, prior, provenance = load_training(args.cifar_root, config)
    weak, strong, inference = image_transforms()
    report = {"purpose": "timing_and_memory_only", "quality_evaluation_executed": False,
              "full_training_campaign_executed": False, "pilot_limit_seconds": args.time_limit_seconds,
              "batch_size": args.batch_size, "seed": config["pilot_seed"], "data": provenance,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                              "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20},
              "conventions": {"dss_plus_target": "paper_raw_weak_p_to_power_1_over_T",
                              "dividemix_mse": "official_code_class_mean_lambda_25",
                              "dividemix_profile_ramp": "one_for_operation_cost_only"},
              "methods": {}, "elapsed_seconds": None}
    report_path = output / "gpu_pilot_report.json"
    def save():
        report["elapsed_seconds"] = time.perf_counter() - start
        temporary = output / "gpu_pilot_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(report_path)
    save()
    for method in ("dss_plus_estimated_prior", "dividemix"):
        record = {"status": "started", "stages": [], "progress": {}}
        report["methods"][method] = record
        save()
        models = optimizers = states = probabilities = None
        try:
            guard(deadline)
            settings = config["dss_plus" if method.startswith("dss") else "dividemix"]
            models, optimizers = model_pair(config["pilot_seed"], settings, device)
            if method.startswith("dss"):
                states = [DSSState(len(view.ids), 10, prior) for _ in range(2)]
                operations = [("warmup_full_epoch", lambda: warmup_dss(models, optimizers, states, view, weak, strong, settings,
                                                                        args.batch_size, config["pilot_seed"], deadline, device, record["progress"])),
                              ("mature_operation_full_epoch", lambda: mature_dss(models, optimizers, states, view, weak, strong, settings,
                                                                                   args.batch_size, config["pilot_seed"], deadline, device, record["progress"]))]
            else:
                operations = [("warmup_full_epoch", lambda: warmup_dividemix(models, optimizers, view, weak,
                                                                              args.batch_size, config["pilot_seed"], deadline, device, record["progress"]))]
            for name, operation in operations:
                guard(deadline)
                record["active_stage"] = name
                record["progress"] = {}
                save()
                record["stages"].append(measured(name, operation))
                record.pop("active_stage", None)
                save()
            if method == "dividemix":
                guard(deadline)
                record["active_stage"] = "full_data_gmm_scan"
                record["progress"] = {}
                save()
                holder = {}
                def scan():
                    holder["probabilities"], details = scan_dividemix(models, view, inference, args.batch_size,
                                                                       config["pilot_seed"], deadline, device, record["progress"])
                    return details
                record["stages"].append(measured("full_data_gmm_scan", scan))
                probabilities = holder["probabilities"]
                record.pop("active_stage", None)
                save()
                record["active_stage"] = "mature_operation_full_epoch"
                record["progress"] = {}
                save()
                record["stages"].append(measured("mature_operation_full_epoch", lambda: mature_dividemix(
                    models, optimizers, view, probabilities, weak, settings, args.batch_size, config["pilot_seed"], deadline, record["progress"])))
                record.pop("active_stage", None)
                save()
            record["status"] = "completed_timing_pilot"
        except (Censored, torch.cuda.OutOfMemoryError) as error:
            if hasattr(error, "stage_measurement"):
                record["stages"].append(error.stage_measurement)
            record.update(status="censored", reason=f"{type(error).__name__}: {error}")
        except Exception as error:
            if hasattr(error, "stage_measurement"):
                record["stages"].append(error.stage_measurement)
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        finally:
            record.pop("active_stage", None)
            save()
            del models, optimizers, states, probabilities
            torch.cuda.empty_cache()
    report["budget_exceeded_at_boundary"] = report["elapsed_seconds"] > args.time_limit_seconds
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-root", type=Path, required=True,
                        help="attached Kaggle dataset directory containing cifar-10-batches-py")
    parser.add_argument("--output", type=Path, required=True, help="fresh /kaggle/working output directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--time-limit-seconds", type=int, default=1800)
    run(parser.parse_args())
