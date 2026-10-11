def vision_transform(name):
    from torchvision import transforms
    if name in {"mnist", "fashion_mnist"}:
        # Fixed values, no fitting to validation/test examples.
        mean, std = (0.1307,), (0.3081,)
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
