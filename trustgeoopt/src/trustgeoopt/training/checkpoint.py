from pathlib import Path

import torch

from ..utils.seeds import rng_state, restore_rng


def save_checkpoint(path, model, optimizer, step, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    rng=rng_state(), extra=extra or {}), temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, device="cpu", restore_random=True):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_random:
        restore_rng(checkpoint["rng"])
    return checkpoint
