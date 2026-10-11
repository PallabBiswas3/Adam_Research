import torch
from torch.utils.data import TensorDataset


def spd_matrix(dimension, condition, seed, rotated=True, dtype=torch.float64, device="cpu"):
    if not 1 <= dimension <= 512 or condition < 1:
        raise ValueError("toy SPD dimension in [1,512], condition >=1 required")
    spectrum = torch.logspace(0, float(torch.log10(torch.tensor(float(condition)))), dimension, dtype=dtype)
    if rotated:
        rng = torch.Generator().manual_seed(seed)
        q = torch.linalg.qr(torch.randn(dimension, dimension, generator=rng, dtype=dtype)).Q
        matrix = (q*spectrum) @ q.T
    else:
        matrix = torch.diag(spectrum)
    return matrix.to(device)


class SyntheticObjective:
    """Exact toy objectives. Additive gradient noise does not change true f_t."""

    def __init__(self, config, seed=0, dtype=torch.float64, device="cpu"):
        self.config = dict(config)
        self.kind = config["kind"]
        if self.kind not in {"quadratic", "rotated", "rosenbrock", "nonstationary", "noisy"}:
            raise ValueError(f"Unknown synthetic objective {self.kind}")
        self.dimension = config.get("dimension", 16)
        if not 2 <= self.dimension <= 512:
            raise ValueError("synthetic dimension must be in [2,512]")
        self.dtype, self.device = dtype, torch.device(device)
        self.a = spd_matrix(self.dimension, config.get("condition", 100), seed+100,
                            config.get("rotated", self.kind != "quadratic"), dtype, device)
        self.a2 = spd_matrix(self.dimension, config.get("condition_after", 10), seed+101, True, dtype, device)
        noise_condition = config.get("noise_condition", 1)
        noise_cov = spd_matrix(self.dimension, noise_condition, seed+102, True, dtype, device)
        self.noise_factor = torch.linalg.cholesky(noise_cov/noise_condition)

    def geometry(self, step):
        changed = self.kind == "nonstationary" and step >= self.config.get("change_step", 100)
        center = torch.full((self.dimension,), self.config.get("optimum_shift", 1.0) if changed else 0.0,
                            dtype=self.dtype, device=self.device)
        return self.a2 if changed else self.a, center

    def optimum(self, step):
        return torch.ones(self.dimension, dtype=self.dtype, device=self.device) if self.kind == "rosenbrock" else self.geometry(step)[1]

    def loss(self, parameter, step=0):
        x = parameter.flatten()
        if self.kind == "rosenbrock":
            return (100*(x[1:]-x[:-1].square()).square()+(1-x[:-1]).square()).sum()
        matrix, center = self.geometry(step)
        error = x-center
        return 0.5*torch.dot(error, matrix @ error)

    def gradient(self, parameter, step=0):
        x = parameter.detach().clone().requires_grad_()
        return torch.autograd.grad(self.loss(x, step), x)[0].detach()

    def sample_noise(self, step, generator):
        scale = self.config.get("noise_std", 0.0)
        if step >= self.config.get("noise_change_step", 10**12):
            scale = self.config.get("noise_std_after", scale)
        z = torch.randn(self.dimension, generator=generator, dtype=self.dtype).to(self.device)
        return scale*(self.noise_factor @ z)

    def curvature(self, parameter, direction, step):
        d = direction.flatten()
        norm_sq = d.square().sum()
        if norm_sq <= 1e-24:
            return float("nan")
        if self.kind != "rosenbrock":
            return float(torch.dot(d, self.geometry(step)[0] @ d)/norm_sq)
        x = parameter.detach().clone().requires_grad_()
        g = torch.autograd.grad(self.loss(x, step), x, create_graph=True)[0]
        hv = torch.autograd.grad((g*direction).sum(), x)[0]
        return float((direction*hv).sum()/norm_sq)


def supervised_splits(config, seed):
    """One teacher/feature map, independent samples, train-only normalization."""
    dimension, classes = config.get("dimension", 20), config.get("classes", 2)
    task = config.get("task", "classification")
    data_seed = config.get("data_seed", 1234)
    rng = torch.Generator().manual_seed(data_seed)
    condition = config.get("condition", 100)
    covariance = spd_matrix(dimension, condition, data_seed+1, True, torch.float32)
    factor = torch.linalg.cholesky(covariance/condition)
    teacher = torch.randn(dimension, classes if task == "classification" else 1, generator=rng)
    sizes = [config.get("train_size", 512), config.get("validation_size", 128), config.get("test_size", 128)]
    xs, ys = [], []
    for index, size in enumerate(sizes):
        x = torch.randn(size, dimension, generator=rng) @ factor.T
        logits = x @ teacher
        if task == "classification":
            y = logits.argmax(1)
            if index == 0:
                mask = torch.rand(size, generator=rng) < config.get("label_noise", 0.0)
                y[mask] = torch.randint(classes, (int(mask.sum()),), generator=rng)
        elif task == "regression":
            y = logits + config.get("target_noise", 0.0)*torch.randn(size, 1, generator=rng)
        else:
            raise ValueError("task must be classification or regression")
        if index == 0:
            x = x + config.get("input_noise", 0.0)*torch.randn(x.shape, generator=rng)
        xs.append(x)
        ys.append(y)
    mean, std = xs[0].mean(0), xs[0].std(0).clamp_min(1e-6)
    return [TensorDataset((x-mean)/std, y) for x, y in zip(xs, ys)]
