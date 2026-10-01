import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TimeFrequencyEncoder(nn.Module):
    """
    Differentiable 1D-to-2D Time-Frequency converter.
    Computes magnitude spectrogram via STFT and maps to (B, 3, 64, 64) for 2D vision models.
    """
    def __init__(self, n_fft: int = 32, hop_length: int = 3, out_size: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out_size = out_size
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 187)
        B, C, L = x.shape
        signal = x.squeeze(1)  # (B, L)

        # STFT computation
        stft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        mag = torch.abs(stft).unsqueeze(1)  # (B, 1, F, T)

        # Resize to square image (e.g. 64x64)
        img = F.interpolate(mag, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)

        # Repeat to 3 channels for RGB vision backbones
        img_3ch = img.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        return img_3ch


class PretrainedTransferModel(nn.Module):
    """
    Pretrained Vision Backbone (e.g., ResNet18, MobileNetV3) adapted for 1D ECG signals
    via an internal differentiable Time-Frequency / Spectrogram encoder.
    """
    def __init__(
        self,
        in_channels: int = 1,
        backbone_name: str = "resnet18",
        num_classes: int = 1,
        pretrained: bool = False,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.tf_encoder = TimeFrequencyEncoder(n_fft=32, hop_length=3, out_size=64)

        if backbone_name.lower() == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                self.backbone = models.resnet18(weights=weights)
            except Exception:
                self.backbone = models.resnet18(weights=None)
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
            
        elif backbone_name.lower() == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            try:
                self.backbone = models.mobilenet_v3_small(weights=weights)
            except Exception:
                self.backbone = models.mobilenet_v3_small(weights=None)
                
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 1D (B, 1, L), pass through TF encoder
        if x.dim() == 3:
            x = self.tf_encoder(x)  # (B, 3, 64, 64)

        features = self.backbone(x)
        out = self.classifier(features)
        return out
