"""
train.py — 训练 edge classifier on toy tracking graph.

Usage:
    python -m src.train --epochs 30 --batch-size 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from .dataset import load_events
from .model import TrackNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, opt, device, pos_weight):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits, _ = model(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch.edge_attr,
                                                  pos_weight=pos_weight)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index)
        all_logits.append(logits.cpu())
        all_y.append(batch.edge_attr.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    probs = torch.sigmoid(logits).numpy()
    auc = roc_auc_score(y, probs)
    # accuracy at 0.5
    pred = (probs > 0.5).astype(np.float32)
    acc = (pred == y).mean()
    return auc, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/toy_hits.npz"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    device = get_device()
    print(f"[device] {device}")

    print("[data] loading events ...")
    graphs = load_events(args.data, k=args.k, max_events=args.max_events)
    n_train = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:n_train], graphs[n_train:]
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    print(f"[data] train={len(train_graphs)} val={len(val_graphs)} "
          f"(hits/event ~ {graphs[0].num_nodes})")

    model = TrackNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # pos_weight = n_background / n_signal (~ 4.7)
    pos_weight = torch.tensor(4.7, device=device)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device, pos_weight)
        auc, acc = evaluate(model, val_loader, device)
        marker = ""
        if auc > best_auc:
            best_auc = auc
            marker = " *"
        print(f"[epoch {epoch:3d}] loss={loss:.4f}  val_auc={auc:.4f}  "
              f"val_acc={acc:.4f}{marker}")

    print(f"[done] best val AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
