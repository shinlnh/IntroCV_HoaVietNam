from torch import nn
from torchvision import models

BACKBONES = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "resnet50": models.resnet50,
}

WEIGHTS = {
    "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
}


def build_model(backbone: str, num_classes: int, dropout: float) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone {backbone}")
    net = BACKBONES[backbone](weights=WEIGHTS[backbone])
    if backbone.startswith("efficientnet"):
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    else:
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    return net
