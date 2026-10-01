import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings for temporal sequences."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class Transformer1D(nn.Module):
    """
    1D Temporal Transformer with Multi-Head Self-Attention for ECG signal classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        patch_size: int = 5,
        stride: int = 2
    ):
        super().__init__()
        
        # Patch/feature projection stem
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 187)
        feat = self.proj(x)  # (batch, d_model, L')
        feat = feat.permute(0, 2, 1)  # (batch, L', d_model)

        # Prepend CLS token
        batch_size = feat.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, feat), dim=1)  # (batch, 1 + L', d_model)

        x_emb = self.pos_encoder(x_emb)
        trans_out = self.transformer_encoder(x_emb)
        cls_out = self.layer_norm(trans_out[:, 0, :])  # Extract [CLS] token representation
        out = self.head(cls_out)
        return out
