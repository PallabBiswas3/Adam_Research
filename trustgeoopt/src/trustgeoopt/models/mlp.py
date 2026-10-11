import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, outputs, hidden=(64,), activation="relu", initialization="default"):
        super().__init__()
        acts = dict(relu=nn.ReLU, tanh=nn.Tanh, gelu=nn.GELU, softplus=nn.Softplus)
        if activation not in acts:
            raise ValueError("unknown activation")
        layers = [nn.Flatten()]
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), acts[activation]()])
            previous = width
        layers.append(nn.Linear(previous, outputs))
        self.network = nn.Sequential(*layers)
        if initialization not in {"default", "kaiming", "xavier"}:
            raise ValueError("unknown initialization")
        if initialization != "default":
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    if initialization == "kaiming":
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    else:
                        nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)
