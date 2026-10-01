import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling1D(nn.Module):
    """Computes a learned weighted sum across temporal sequence steps."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        scores = self.attn(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)  # (batch, hidden_dim)
        return context


class BiLSTMNet(nn.Module):
    """
    Bidirectional LSTM with 1D Conv feature tokenizer and Attention Pooling
    for ECG sequence classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 1D Conv feature tokenizer to compress sequence length and extract local waveform tokens
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attn_pool = AttentionPooling1D(out_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 187)
        feat = self.conv_stem(x)  # (batch, 64, L')
        feat = feat.permute(0, 2, 1)  # (batch, L', 64)

        lstm_out, _ = self.lstm(feat)  # (batch, L', out_dim)
        context = self.attn_pool(lstm_out)  # (batch, out_dim)
        out = self.fc(context)  # (batch, num_classes)
        return out
