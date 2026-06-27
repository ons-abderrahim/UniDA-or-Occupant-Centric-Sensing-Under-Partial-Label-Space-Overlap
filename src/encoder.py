"""
Shared Temporal Encoder G_tp — identical across all four UniDA methods.

Architecture (Figure 2):
    Input (1D Sensor Window)
    → Conv1D(64, k=3) + ReLU
    → Batch Normalization
    → Dropout(0.3)
    → Flatten
    → Dense(128) + ReLU
    → Dense(64)  + ReLU
    → z = G_tp(x_i)   [Eq. 40]

Reference:
    "Universal Domain Adaptation for Smart-Building Sensor Data:
     A Comparative Study of PPOT, MLNet, EIAKDA, and LEAD"
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Shared 1-D temporal encoder used by all four UniDA methods.

    Args:
        in_channels (int): Number of sensor features (input channels).
        seq_len (int): Length of the sliding window (time steps).
        dropout (float): Dropout probability (default 0.3).
    """

    def __init__(self, in_channels: int, seq_len: int, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        # Conv1D block
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
        )

        # Compute flattened size after convolution
        flat_size = 64 * seq_len

        # Fully-connected projection
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_channels, seq_len)
        Returns:
            z: Tensor of shape (B, 64)  — the latent embedding z = G_tp(x_i)
        """
        h = self.conv(x)   # (B, 64, seq_len)
        z = self.fc(h)     # (B, 64)
        return z


class LinearClassifier(nn.Module):
    """Simple linear head used on top of the shared encoder."""

    def __init__(self, feat_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
