import torch

from .covariance import history_factor


class GradientHistory:
    """Bounded rolling window, not an online PCA/FD implementation."""

    def __init__(self, rank=4, window=8, centered=False):
        if rank < 1 or window < rank:
            raise ValueError("window >= rank >= 1 required")
        self.rank, self.window, self.centered = rank, window, centered
        self.history = None

    @torch.no_grad()
    def update(self, gradient):
        g = gradient.detach().flatten()
        if not torch.isfinite(g).all():
            raise ValueError("finite gradient required")
        if self.history is not None and (self.history.shape[0] != g.numel() or
                self.history.device != g.device or self.history.dtype != g.dtype):
            raise ValueError("history shape/device/dtype changed")
        self.history = g[:, None].clone() if self.history is None else torch.cat(
            (self.history[:, -(self.window-1):], g[:, None]), dim=1) if self.window > 1 else g[:, None].clone()

    def factor(self):
        if self.history is None:
            raise ValueError("empty history")
        return history_factor(self.history, self.rank, self.centered)

    def state_dict(self):
        return dict(history=self.history, rank=self.rank, window=self.window, centered=self.centered)

    def load_state_dict(self, state):
        if (state["rank"], state["window"], state["centered"]) != (self.rank, self.window, self.centered):
            raise ValueError("incompatible history configuration")
        self.history = state["history"]
