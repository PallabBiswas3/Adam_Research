import copy
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..datasets.synthetic import supervised_splits
from ..datasets.vision import vision_splits
from ..diagnostics.gradient_statistics import prospective_adam_state
from ..diagnostics.reliability import ReliabilityMonitor
from ..models import make_model
from ..optimizers import make_optimizer
from ..utils.hardware import synchronize
from ..utils.logging import RunLog
from ..utils.seeds import set_seed
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluation import evaluate


def train(config, output_root="results", resume=None, evaluate_test=False):
    set_seed(config["seed"], config.get("deterministic", True), config.get("threads", 1))
    device = torch.device(config.get("device", "cpu"))
    data_config = config["dataset"]
    data = supervised_splits(data_config, config["seed"]) if data_config["name"] == "synthetic" else vision_splits(data_config)
    loader_rng = torch.Generator().manual_seed(config["seed"]+900)
    loaders = [DataLoader(dataset, batch_size=config.get("batch_size", 64), shuffle=i == 0,
                          generator=loader_rng if i == 0 else None, num_workers=0) for i, dataset in enumerate(data)]
    task = data_config.get("task", "classification")
    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    model = make_model(config["model"], data_config).to(device)
    optimizer = make_optimizer(model.parameters(), config["optimizer"])
    monitor = ReliabilityMonitor(**config.get("diagnostics", {}))
    best_loss, best_weights, start_epoch, steps = float("inf"), None, 0, 0
    optimizer_time = diagnostic_time = 0.0
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        extra = checkpoint["extra"]
        old, new = dict(extra["config"]), dict(config)
        old.pop("epochs", None)
        new.pop("epochs", None)
        if old != new:
            raise ValueError("resume config changed beyond epochs budget")
        start_epoch = checkpoint["step"]+1
        loader_rng.set_state(extra["loader_rng"].cpu())
        monitor.load_state_dict(extra["monitor"])
        best_loss, best_weights, steps = extra["best_loss"], extra["best_weights"], extra["steps"]
        optimizer_time, diagnostic_time = extra["optimizer_seconds"], extra["diagnostic_seconds"]
    log = RunLog(config, output_root, device)
    log.metadata.update(dataset_sizes=[len(d) for d in data], resume_from=str(resume) if resume else None,
                        model_parameters=sum(p.numel() for p in model.parameters()), test_evaluated=False)
    try:
        if start_epoch >= config.get("epochs", 5):
            raise ValueError("checkpoint already meets epochs budget")
        initial_validation = evaluate(model, loaders[1], criterion, device, task)
        log.metadata["initial_validation"] = initial_validation
        for epoch in range(start_epoch, config.get("epochs", 5)):
            model.train()
            total_loss = gradient_sum = step_sum = 0.0
            seen, batch_count, flags = 0, 0, 0
            for x, y in loaders[0]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite train loss")
                loss.backward()
                parameters = [p for p in model.parameters() if p.grad is not None]
                flat_g = torch.cat([p.grad.detach().flatten() for p in parameters])
                if not torch.isfinite(flat_g).all():
                    raise FloatingPointError("nonfinite train gradient")
                gradient_sum += float(flat_g.norm())
                if config.get("gradient_clip") is not None:
                    nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
                synchronize(device)
                start = time.perf_counter()
                state = prospective_adam_state(optimizer)
                signals = monitor.observe(torch.cat([p.grad.flatten() for p in parameters])) if state is None else monitor.observe(*state)
                synchronize(device)
                diagnostic_time += time.perf_counter()-start
                before = [p.detach().clone() for p in parameters]
                start = time.perf_counter()
                optimizer.step()
                synchronize(device)
                optimizer_time += time.perf_counter()-start
                step_sum += math.sqrt(sum(float((p.detach()-b).square().sum()) for p, b in zip(parameters, before)))
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite model parameter")
                flags += signals["flag"]
                batch_count += 1
                steps += 1
                total_loss += float(loss)*x.shape[0]
                seen += x.shape[0]
            evaluation_start = time.perf_counter()
            validation = evaluate(model, loaders[1], criterion, device, task)
            synchronize(device)
            evaluation_seconds = time.perf_counter()-evaluation_start
            if not math.isfinite(validation["loss"]):
                raise FloatingPointError("nonfinite validation loss")
            if validation["loss"] < best_loss:
                best_loss, best_weights = validation["loss"], copy.deepcopy(model.state_dict())
            log.write(dict(epoch=epoch, steps=steps, train_loss=total_loss/seen,
                validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
                mean_gradient_norm=gradient_sum/batch_count, mean_step_norm=step_sum/batch_count,
                diagnostic_flags=flags, optimizer_seconds=optimizer_time,
                diagnostic_seconds=diagnostic_time, evaluation_seconds=evaluation_seconds,
                elapsed_seconds=log.elapsed()))
            save_checkpoint(log.path/"checkpoint.pt", model, optimizer, epoch, dict(config=config,
                loader_rng=loader_rng.get_state(), monitor=monitor.state_dict(), best_loss=best_loss,
                best_weights=best_weights, steps=steps, optimizer_seconds=optimizer_time,
                diagnostic_seconds=diagnostic_time))
        test_metrics = None
        if evaluate_test:
            model.load_state_dict(best_weights)
            test_metrics = evaluate(model, loaders[2], criterion, device, task)
            log.metadata["test_evaluated"] = True
        return log.finish(summary=dict(best_validation_loss=best_loss, final_validation=validation,
            test=test_metrics, total_steps=steps, optimizer_seconds=optimizer_time,
            diagnostic_seconds=diagnostic_time))
    except Exception as error:
        log.finish(status="failed", error=error)
        raise
