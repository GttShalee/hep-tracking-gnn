"""
dataset.py — 把 .npz 里的 hits 转成 PyG Data 对象列表。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .build_graph import build_knn_graph


def load_events(npz_path: Path, k: int = 8, max_events: int | None = None):
    d = np.load(npz_path, allow_pickle=False)
    hits, offsets = d["hits"], d["event_offsets"]
    n_events = int(d["n_events"])
    if max_events is not None:
        n_events = min(n_events, max_events)

    graphs = []
    for ev_id in range(n_events):
        lo, hi = offsets[ev_id], offsets[ev_id + 1]
        ev = hits[lo:hi]
        xyz, edge_index, edge_y, pid = build_knn_graph(ev, k=k)

        # 节点特征: 归一化的 (x, y, z) + layer one-hot 太占空间, 先用 (x, y, z)
        x = torch.tensor(xyz, dtype=torch.float32)
        # 简单归一化 (按探测器尺寸)
        x = x / 2.5  # r_max=2.0, z_max=1.0, 粗归一化

        graphs.append(Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_y, dtype=torch.float32),
            pid=torch.tensor(pid, dtype=torch.long),
            num_nodes=len(ev),
        ))
    return graphs
