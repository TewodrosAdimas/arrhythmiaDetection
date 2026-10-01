import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Exact reproduction of the baseline 1D-CNN architecture from the original project.
    
    Structure:
      - 3x Conv1d (kernels=3) + ReLU + MaxPool1d(2)
      - Flatten -> Linear(2688, 64) -> ReLU -> Linear(64, 1)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, input_len: int = 187):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Conv layers
        self.layer1 = nn.Conv1d(in_channels, 32, kernel_size=3)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool1d(2)

        self.layer4 = nn.Conv1d(32, 64, kernel_size=3)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool1d(2)

        self.layer7 = nn.Conv1d(64, 128, kernel_size=3)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.MaxPool1d(2)
        self.layer10 = nn.Flatten()

        # Input length 187 -> after 3 convs & maxpool: 128 * 21 = 2688
        # Calculate dynamically to support varying input lengths
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_len)
            x = self.layer3(self.layer2(self.layer1(dummy)))
            x = self.layer6(self.layer5(self.layer4(x)))
            x = self.layer9(self.layer8(self.layer7(x)))
            flatten_dim = x.numel()

        self.layer11 = nn.Linear(flatten_dim, 64)
        self.layer12 = nn.ReLU()
        self.layer13 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer6(self.layer5(self.layer4(x)))
        x = self.layer9(self.layer8(self.layer7(x)))
        x = self.layer10(x)
        x = self.layer12(self.layer11(x))
        x = self.layer13(x)
        return x
