from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, channels=1, classes=10, width=16):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(width, 2*width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.classifier = nn.Linear(2*width*16, classes)

    def forward(self, x):
        return self.classifier(self.features(x))
