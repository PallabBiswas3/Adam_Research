import torch
from torch.utils.data import random_split

from .transforms import vision_transform


def vision_splits(config):
    from torchvision import datasets
    name = config["name"]
    mapping = dict(mnist=datasets.MNIST, fashion_mnist=datasets.FashionMNIST,
                   cifar10=datasets.CIFAR10, cifar100=datasets.CIFAR100)
    if name not in mapping:
        raise ValueError(f"Unknown dataset {name}")
    options = dict(root=config.get("root", "data"), transform=vision_transform(name),
                   download=config.get("download", False))
    train = mapping[name](train=True, **options)
    test = mapping[name](train=False, **options)
    count = config.get("validation_size", 5000)
    if not 0 < count < len(train):
        raise ValueError("invalid validation size")
    training, validation = random_split(train, [len(train)-count, count],
        generator=torch.Generator().manual_seed(config.get("split_seed", 1234)))
    return training, validation, test
