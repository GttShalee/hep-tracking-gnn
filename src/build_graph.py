"""
build_graph.py — 从 hit 点云构建 kNN 碰撞图。

输入: 单个事件的 hits (N, 5): [x, y, z, layer, particle_id]
输出:
    node_feats  (N, 3): x, y, z (归一化前)
    edge_index  (2, E): 有向边 (kNN 的 i->j 和 j->i 都保留, 供 PyG 用)
    edge_y      (E,): 1 = 两 hit 同属一条真实粒子径迹; 0 = 假边
    node_pid    (N,):  每个 hit 的 truth particle_id (-1 = 噪声)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def build_knn_graph(hits: np.ndarray, k: int = 8):
    """对单事件 hits 做 kNN, 返回 edge_index + edge truth。"""
    xyz = hits[:, :3]
    pid = hits[:, 4].astype(int)
    n = len(hits)

    tree = cKDTree(xyz)
    # query k+1 因为最近邻包含自己
    _, idx = tree.query(xyz, k=k + 1)

    src_list, dst_list = [], []
    for i in range(n):
        for j in idx[i][1:]:  # 跳过自己
            src_list.append(i)
            dst_list.append(int(j))

    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)

    # edge truth: 两个 hit 都是真实粒子 (pid >= 0) 且 pid 相同
    src_pid = pid[edge_index[0]]
    dst_pid = pid[edge_index[1]]
    edge_y = ((src_pid >= 0) & (dst_pid >= 0) & (src_pid == dst_pid)).astype(np.float32)

    return xyz, edge_index, edge_y, pid


def summarize(edge_y: np.ndarray):
    n_edges = len(edge_y)
    n_signal = int(edge_y.sum())
    return n_edges, n_signal, n_signal / n_edges if n_edges else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/toy_hits.npz"))
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--event", type=int, default=0)
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=False)
    hits, offsets = d["hits"], d["event_offsets"]
    lo, hi = offsets[args.event], offsets[args.event + 1]
    ev = hits[lo:hi]

    xyz, edge_index, edge_y, pid = build_knn_graph(ev, k=args.k)
    n_edges, n_signal, frac = summarize(edge_y)
    print(f"[event {args.event}] N_hits={len(ev)}, k={args.k}")
    print(f"  edges total = {n_edges}")
    print(f"  signal edges (same particle) = {n_signal} ({frac*100:.1f}%)")
    print(f"  background edges = {n_edges - n_signal} ({(1-frac)*100:.1f}%)")


if __name__ == "__main__":
    main()
