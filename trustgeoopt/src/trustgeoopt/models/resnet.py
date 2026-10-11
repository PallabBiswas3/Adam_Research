from torch import nn


def resnet18(classes=10, channels=3, cifar_stem=True):
    """torchvision ResNet-18, optional explicit CIFAR 3x3/stride-1 stem."""
    from torchvision.models import resnet18 as reference
    model = reference(weights=None, num_classes=classes)
    if cifar_stem:
        model.conv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif channels != 3:
        model.conv1 = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
    return model
