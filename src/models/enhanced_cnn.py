import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Conv1D + BatchNorm1D + ReLU + Dropout + MaxPool1D"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, dropout: float = 0.1, pool: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.dropout(self.act(self.bn(self.conv(x)))))


class EnhancedCNN(nn.Module):
    """
    Modern 1D-CNN with Batch Normalization, Dropout Regularization,
    and Global Average Pooling for ECG classification.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 32, dropout: float = 0.25):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock1D(in_channels, base_filters, kernel_size=7, dropout=dropout, pool=True),
            ConvBlock1D(base_filters, base_filters * 2, kernel_size=5, dropout=dropout, pool=True),
            ConvBlock1D(base_filters * 2, base_filters * 4, kernel_size=3, dropout=dropout, pool=True),
            ConvBlock1D(base_filters * 4, base_filters * 8, kernel_size=3, dropout=dropout, pool=True),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
