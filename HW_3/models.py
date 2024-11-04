import timm
import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        a1 = self.feature_extractor(x)
        a1 = torch.flatten(a1, 1)
        a2 = self.classifier(a1)
        return a2


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


def get_model(name, num_classes):
    """Load model based on dataset and config."""
    if name == "resnet":
        model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    elif name == "mlp":
        model = MLP()
    elif name == "lenet":
        model = LeNet()
    else:
        raise ValueError("Model not supported.")

    return model
