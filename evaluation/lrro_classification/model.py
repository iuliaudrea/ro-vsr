"""
Attention pooling + MLP classifier.

Architecture (matching the training notebook):
  - Attention pooling over the temporal dimension of encoder embeddings,
    producing a single fixed-size vector per clip.
  - 2-layer MLP with BatchNorm + ReLU + Dropout for word classification.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Attention pooling over a (B, T, D) tensor with optional mask."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x, mask=None):
        # x: (B, T, D); mask: (B, 1, T) bool, True = valid frame
        weights = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            weights.masked_fill_(~mask.squeeze(1), -1e9)
        attn_weights = torch.softmax(weights, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled, attn_weights


class MLP(nn.Module):
    """Attention pooling followed by a small MLP head."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.6):
        super().__init__()
        self.pool = AttentionPooling(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask=None):
        pooled, attn = self.pool(x, mask)
        return self.net(pooled), attn
