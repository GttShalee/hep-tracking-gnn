"""
toy_detector.py — 简化硅像素桶形探测器（barrel）模拟器

物理模型
--------
- 磁场 B 沿 +z 方向，带电粒子在横向 (xy) 平面走圆弧，在 z 方向匀速漂移。
- 回旋半径 R = p_T / (0.3 * B)，单位约定：p_T [GeV], B [T], R [m]。
- 探测器由 n_layers 层同心圆柱组成，半径递增；粒子穿过每层时留下一个 hit。
- 叠加一定比例的噪声 hit（不对应任何真实粒子）。

输出
----
每个事件生成一张 hits 表 (np.ndarray, shape=(N_hits, 5)):
    列 0: x [m]
    列 1: y [m]
    列 2: z [m]
    列 3: layer_id [int]
    列 4: particle_id [int]   # >=0 真实粒子；-1 为噪声
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorGeometry:
    """桶形探测器几何。"""
    n_layers: int = 10
    r_min: float = 0.5   # m，最内层半径
    r_max: float = 2.0   # m，最外层半径
    z_max: float = 1.0   # m，|z| 截断

    @property
    def layer_radii(self) -> np.ndarray:
        return np.linspace(self.r_min, self.r_max, self.n_layers)


@dataclass
class EventParams:
    """单个事件的粒子数 / 噪声数参数。"""
    n_tracks: int = 200          # 每个事件的真实粒子数
    noise_fraction: float = 0.15  # 噪声 hit 占总 hit 的比例
    B_field: float = 2.0          # T，和 CMS 芯径磁场同量级
    pT_min: float = 0.5           # GeV
    pT_max: float = 20.0          # GeV
    seed: int | None = None


def _helix_intersection(phi0: float, q: int, pT: float, B: float, r: float):
    """
    计算从原点出发的螺旋线与半径 r 的圆柱面的交点。

    返回 (x, y, delta_theta) 或 None（若轨道达不到该半径）。
    delta_theta 用于沿弧长推进 z。
    """
    R = pT / (0.3 * B)  # 回旋半径 [m]
    # 起点处 alpha0 = theta - phi0 = -q*pi/2
    alpha0 = -q * np.pi / 2.0
    # sin(alpha) = q * ((r/R)^2 / 2 - 1)
    sin_alpha = q * (0.5 * (r / R) ** 2 - 1.0)
    if abs(sin_alpha) > 1.0:
        return None  # 该层在轨道曲率外面，打不到
    # 取粒子前进方向上的解：alpha 从 alpha0 开始单调增加
    # 两个解: arcsin(sin_alpha) 和 pi - arcsin(sin_alpha)
    a1 = np.arcsin(sin_alpha)
    a2 = np.pi - a1
    candidates = [a for a in (a1, a2) if a > alpha0 + 1e-6]
    if not candidates:
        return None
    alpha = min(candidates)  # 第一次到达该半径
    theta = phi0 + alpha
    x = -q * R * np.sin(phi0) + R * np.cos(theta)
    y = q * R * np.cos(phi0) + R * np.sin(theta)
    delta_theta = alpha - alpha0
    return x, y, delta_theta


def generate_event(params: EventParams, geo: DetectorGeometry):
    """生成一个事件，返回 hits 数组。"""
    rng = np.random.default_rng(params.seed)
    radii = geo.layer_radii

    hits = []  # list of (x, y, z, layer, particle_id)

    for pid in range(params.n_tracks):
        # 随机发射参数
        phi0 = rng.uniform(0.0, 2.0 * np.pi)
        q = int(rng.choice([-1, 1]))
        pT = np.exp(rng.uniform(np.log(params.pT_min), np.log(params.pT_max)))
        # z 方向斜率：z = z_slope * r
        z_slope = rng.uniform(-1.5, 1.5)

        for layer_id, r in enumerate(radii):
            res = _helix_intersection(phi0, q, pT, params.B_field, r)
            if res is None:
                continue
            x, y, _ = res
            # 用 r 近似弧长（小角度近似下 z 线性增长）
            z = z_slope * r
            if abs(z) > geo.z_max:
                continue
            hits.append((x, y, z, layer_id, pid))

    # 注入噪声 hit
    n_real = len(hits)
    if n_real > 0 and params.noise_fraction > 0:
        n_noise = int(n_real * params.noise_fraction / (1 - params.noise_fraction))
        for _ in range(n_noise):
            # 在探测器体积内均匀撒点
            r_noise = np.sqrt(rng.uniform(radii[0] ** 2, radii[-1] ** 2))
            phi_noise = rng.uniform(0.0, 2 * np.pi)
            z_noise = rng.uniform(-geo.z_max, geo.z_max)
            layer_noise = rng.integers(0, geo.n_layers)
            hits.append((
                r_noise * np.cos(phi_noise),
                r_noise * np.sin(phi_noise),
                z_noise,
                int(layer_noise),
                -1,  # 噪声
            ))

    return np.asarray(hits, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Generate toy tracking-detector events.")
    parser.add_argument("--n-events", type=int, default=64)
    parser.add_argument("--out", type=str, default="data/toy_hits.npz")
    parser.add_argument("--n-tracks", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    geo = DetectorGeometry()
    all_hits = []
    event_offsets = [0]
    for i in range(args.n_events):
        ev = EventParams(n_tracks=args.n_tracks, noise_fraction=args.noise,
                         seed=args.seed + i)
        hits = generate_event(ev, geo)
        all_hits.append(hits)
        event_offsets.append(event_offsets[-1] + len(hits))
        if i == 0:
            print(f"[event 0] {len(hits)} hits "
                  f"({(hits[:, 4] >= 0).sum()} real, {(hits[:, 4] < 0).sum()} noise)")

    all_hits = np.concatenate(all_hits, axis=0)
    np.savez(
        args.out,
        hits=all_hits,
        event_offsets=np.asarray(event_offsets, dtype=np.int64),
        layer_radii=geo.layer_radii,
        n_events=args.n_events,
    )
    print(f"[done] wrote {args.out}: {all_hits.shape[0]} hits total, "
          f"{args.n_events} events")


if __name__ == "__main__":
    main()
