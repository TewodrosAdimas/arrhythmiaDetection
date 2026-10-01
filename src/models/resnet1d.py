import torch
import torch.nn as nn
from typing import List


class ResidualBlock1D(nn.Module):
    """Basic 1D Residual Block with 2 Conv1D layers and a shortcut connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D Residual Network (ResNet-1D) for ECG Sequence Classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        layers: List[int] = [2, 2, 2, 2],
        base_channels: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels_curr = base_channels

        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Stages
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1, kernel_size=kernel_size, dropout=dropout)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2, kernel_size=kernel_size, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2, kernel_size=kernel_size, dropout=dropout)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2, kernel_size=kernel_size, dropout=dropout)

        # Global Pooling and Linear Classifier Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1, kernel_size: int = 5, dropout: float = 0.1) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock1D(self.in_channels_curr, out_channels, stride=stride, kernel_size=kernel_size, dropout=dropout))
        self.in_channels_curr = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1, kernel_size=kernel_size, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_1d(in_channels: int = 1, num_classes: int = 1, **kwargs) -> ResNet1D:
    return ResNet1D(in_channels=in_channels, num_classes=num_classes, layers=[2, 2, 2, 2], base_channels=32, **kwargs)


def resnet34_1d(in_channels: int = 1, num_classes: int = 1, **kwargs) -> ResNet1D:
    return ResNet1D(in_channels=in_channels, num_classes=num_classes, layers=[3, 4, 6, 3], base_channels=32, **kwargs)
