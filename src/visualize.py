"""visualize.py — 把生成的 hits 画成探测器截面图。"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    return d["hits"], d["event_offsets"], d["layer_radii"]


def draw_event(hits: np.ndarray, layer_radii: np.ndarray, event_id: int,
               offsets: np.ndarray, out_path: Path):
    lo, hi = offsets[event_id], offsets[event_id + 1]
    ev = hits[lo:hi]

    real = ev[ev[:, 4] >= 0]
    noise = ev[ev[:, 4] < 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---- xy 投影 ----
    ax = axes[0]
    # 画探测器层圆周
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in layer_radii:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="lightgray", lw=0.6, zorder=1)
    # 噪声
    ax.scatter(noise[:, 0], noise[:, 1], s=4, c="lightgray", alpha=0.5,
               label=f"noise (n={len(noise)})", zorder=2)
    # 真实 hit 按粒子着色
    pids = np.unique(real[:, 4].astype(int))
    cmap = plt.get_cmap("hsv", max(len(pids), 1))
    for i, pid in enumerate(pids):
        trk = real[real[:, 4].astype(int) == pid]
        order = np.argsort(trk[:, 3])
        ax.plot(trk[order, 0], trk[order, 1], "-o", ms=3, lw=0.8,
                color=cmap(i), zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Event {event_id}: xy projection "
                 f"({len(pids)} tracks, {len(real)} real hits)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ---- rz 投影 ----
    ax = axes[1]
    r = np.sqrt(ev[:, 0] ** 2 + ev[:, 1] ** 2)
    ax.scatter(noise[:, 2], np.sqrt(noise[:, 0]**2 + noise[:, 1]**2),
               s=4, c="lightgray", alpha=0.5, label="noise")
    r_real = np.sqrt(real[:, 0]**2 + real[:, 1]**2)
    for i, pid in enumerate(pids):
        mask = real[:, 4].astype(int) == pid
        trk = real[mask]
        rr = np.sqrt(trk[:, 0]**2 + trk[:, 1]**2)
        order = np.argsort(trk[:, 3])
        ax.plot(trk[order, 2], rr[order], "-o", ms=3, lw=0.8, color=cmap(i))
    # 画层位置水平线
    for lr in layer_radii:
        ax.axhline(lr, color="lightgray", lw=0.6, zorder=1)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("r [m]")
    ax.set_title("r-z projection")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] event {event_id} -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/toy_hits.npz"))
    p.add_argument("--event", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/event_0.png"))
    args = p.parse_args()

    hits, offsets, radii = load(args.data)
    draw_event(hits, radii, args.event, offsets, args.out)


if __name__ == "__main__":
    main()
