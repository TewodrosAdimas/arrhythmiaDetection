import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Compose1D:
    """Composes several 1D transforms together."""
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class AddGaussianNoise:
    """Adds random Gaussian noise to ECG signal."""
    def __init__(self, std=0.015, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class RandomScale:
    """Multiplies ECG amplitude by a random factor."""
    def __init__(self, min_scale=0.9, max_scale=1.1, p=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
            return x * scale
        return x


class BaselineShift:
    """Adds a small constant or baseline drift to signal."""
    def __init__(self, max_shift=0.05, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            shift = torch.empty(1).uniform_(-self.max_shift, self.max_shift).item()
            return x + shift
        return x


class TimeShift:
    """Circularly or zero-padded shifts the 1D signal along the temporal axis."""
    def __init__(self, max_shift=5, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
        return x


class ContinuousWaveletTransform:
    """
    Converts 1D ECG signal of shape (1, L) into a 2D Time-Frequency representation
    of shape (3, H, W) suitable for pretrained 2D vision models (ResNet, MobileNet).
    
    Uses Morlet/Ricker wavelet bank or STFT-based multi-resolution representation.
    """
    def __init__(self, output_size=(64, 64), in_channels=3):
        self.output_size = output_size
        self.in_channels = in_channels

    def __call__(self, x):
        # x is (1, L)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # We can construct a multi-scale spectrogram/CWT representation using 1D multi-scale convs or STFT
        # Compute multi-window STFT or wavelet-like representation
        signal = x.squeeze(0)  # (L,)
        
        # Spectrogram via STFT with small n_fft for short 187-length signal
        n_fft = 32
        hop_length = 3
        window = torch.hann_window(n_fft, device=signal.device)
        
        # stft produces (freq, time, 2) or complex
        stft = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        # Magnitude spectrogram: (F, T)
        mag = torch.abs(stft).unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
        
        # Resize to target output size (e.g. 64x64 or 128x128 for vision backbones)
        img_2d = F.interpolate(mag, size=self.output_size, mode='bilinear', align_corners=False)
        img_2d = img_2d.squeeze(0)  # (1, H, W)
        
        # Normalize between 0 and 1
        img_min = img_2d.min()
        img_max = img_2d.max()
        if img_max > img_min:
            img_2d = (img_2d - img_min) / (img_max - img_min)
            
        # Expand to 3 channels for standard ImageNet pretrained models
        if self.in_channels == 3:
            img_3ch = img_2d.repeat(3, 1, 1)  # (3, H, W)
            return img_3ch
            
        return img_2d
