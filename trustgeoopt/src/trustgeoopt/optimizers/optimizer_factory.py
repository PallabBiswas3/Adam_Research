import torch

from .adabelief import AdaBelief
from .preconditioners import ReferenceShampoo


def make_optimizer(params, config):
    """Reject unknown knobs rather than changing an experiment silently."""
    options = dict(config)
    name = options.pop("name").lower()
    factories = dict(sgd=torch.optim.SGD, momentum=torch.optim.SGD, adam=torch.optim.Adam,
                     adamw=torch.optim.AdamW, adamw_monitor=torch.optim.AdamW,
                     rmsprop=torch.optim.RMSprop, amsgrad=torch.optim.Adam,
                     adagrad=torch.optim.Adagrad, radam=torch.optim.RAdam,
                     adabelief=AdaBelief, shampoo=ReferenceShampoo)
    if name.startswith("trustgeo") or name in {"fixed_geo", "periodic_geo", "gated_geo"}:
        raise NotImplementedError("Prototype gated by research evidence; see research/novelty_assessment.md")
    if name not in factories:
        raise ValueError(f"Unknown optimizer: {name}")
    if name == "momentum":
        options.setdefault("momentum", 0.9)
    if name == "amsgrad":
        options["amsgrad"] = True
    # Explicit defaults keep AdamW's implicit 0.01 decay out of comparisons.
    options.setdefault("weight_decay", 0.0)
    return factories[name](params, **options)
