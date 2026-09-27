"""
model.py — Embedding MLP + Edge classifier.

Stage 1 (baseline, no message passing):
  node: (x, y, z) -> MLP -> 32-dim embedding
  edge: [emb_i, emb_j, |emb_i - emb_j|] -> MLP -> P(signal)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeClassifier(nn.Module):
    def __init__(self, emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        # edge feature = [emb_i, emb_j, |emb_i-emb_j|] -> 3*emb_dim
        self.net = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb_i, emb_j):
        diff = (emb_i - emb_j).abs()
        h = torch.cat([emb_i, emb_j, diff], dim=-1)
        return self.net(h).squeeze(-1)


class TrackNet(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, emb_dim: int = 32):
        super().__init__()
        self.encoder = NodeEncoder(in_dim, hidden, emb_dim)
        self.edge_clf = EdgeClassifier(emb_dim, hidden)

    def forward(self, x, edge_index):
        emb = self.encoder(x)
        src, dst = edge_index[0], edge_index[1]
        logits = self.edge_clf(emb[src], emb[dst])
        return logits, emb
