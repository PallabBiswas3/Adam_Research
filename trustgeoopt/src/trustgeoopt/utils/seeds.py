import random
import os

import numpy as np
import torch


def set_seed(seed, deterministic=True, threads=1):
    # Needed by deterministic CUDA GEMM; set before CUDA context initialization.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=name, keys=torch.tensor(keys.astype(np.int64)),
                pos=pos, has_gauss=has_gauss, cached=cached), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].cpu().numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
