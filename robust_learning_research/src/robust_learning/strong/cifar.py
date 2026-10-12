"""Benchmark construction separates true labels from observed-label training."""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class ImageTrainingView:
    images: np.ndarray
    observed_y: np.ndarray
    ids: np.ndarray

    def __post_init__(self):
        for array in (self.images, self.observed_y, self.ids):
            array.setflags(write=False)


def make_splits(labels, seed, validation_per_class=500, imbalance_factor=10,
                prior_majority=100, model_val_per_class=200):
    labels = np.asarray(labels)
    if imbalance_factor < 1 or validation_per_class < prior_majority+model_val_per_class:
        raise ValueError("invalid prior ratio/validation reservation")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 401]))
    train, validation, prior = [], [], []
    for c in range(10):
        available = rng.permutation(np.where(labels == c)[0])
        if len(available) <= validation_per_class:
            raise ValueError("insufficient per-class images")
        reserve = available[:validation_per_class]
        n_prior = max(1, int(prior_majority*imbalance_factor**(-c/9)))
        prior.extend(reserve[:n_prior])
        validation.extend(reserve[n_prior:n_prior+model_val_per_class])
        pool = available[validation_per_class:]
        train.extend(pool[:max(1, int(len(pool)*imbalance_factor**(-c/9)))])
    return {"train": np.array(train, dtype=np.int64), "validation": np.array(validation, dtype=np.int64),
            "prior": np.array(prior, dtype=np.int64)}


def corrupt(true, seed, rate, kind):
    if not 0 <= rate <= 1:
        raise ValueError("noise rate must be in [0,1]")
    true = np.asarray(true)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    intervention = rng.random(len(true)) < rate
    observed = true.copy()
    if kind == "symmetric_including_original":
        observed[intervention] = rng.integers(0, 10, intervention.sum())
    elif kind == "asymmetric":
        transition = np.array([0, 1, 0, 5, 7, 3, 6, 7, 8, 1])
        observed[intervention] = transition[true[intervention]]
    else:
        raise ValueError("unregistered noise family")
    return observed, observed != true, intervention


def image_transforms():
    mean, std = (.4914, .4822, .4465), (.2023, .1994, .2010)
    weak = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
    strong = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), transforms.ToTensor(), transforms.Normalize(mean, std)])
    inference = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return weak, strong, inference


def batch_images(view, indices, transform):
    import torch
    return torch.stack([transform(Image.fromarray(view.images[i])) for i in indices])
