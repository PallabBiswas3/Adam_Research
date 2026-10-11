from .mlp import MLP
from .cnn import SmallCNN


def make_model(config, dataset):
    options = dict(config)
    name = options.pop("name")
    outputs = 1 if dataset.get("task") == "regression" else dataset.get("classes", 2)
    if name in {"mlp", "deep_mlp"}:
        dimension = options.pop("input_dim", dataset.get("dimension", 20))
        options.setdefault("hidden", [64] if name == "mlp" else [64]*6)
        return MLP(dimension, outputs, **options)
    if name == "cnn":
        return SmallCNN(classes=outputs, **options)
    if name == "resnet18":
        from .resnet import resnet18
        return resnet18(classes=outputs, **options)
    raise ValueError(f"Unknown architecture {name}")
