# hep-tracking-gnn

**Mini Exa.TrkX**: 在 macOS (Apple Silicon) 上从零实现基于图神经网络的粒子径迹重建 pipeline。

## 为什么做这个

高亮度大型强子对撞机 (HL-LHC) 升级后，每次质子对撞会在硅像素探测器里产生 **数百万个 hit**。
传统的基于规则的径迹重建算法在这个数据量级下已经逼近极限，**图神经网络 (GNN)** 路线
(参见 [Exa.TrkX](https://exatrkx.github.io/) 合作组) 是 ATLAS / CMS 下一代重建的核心方向。

本项目在 M1 Pro / 16GB 上完整复现 Exa.TrkX 的核心阶段：

```
点云 hits → (kNN) 碰撞图 → embedding MLP → edge filtering → GNN edge classifier → 径迹后处理
```

## 探测器模型

简化的硅像素桶形探测器 (barrel)：

- 10 层同心圆柱，半径从 0.5 m 到 2.0 m
- 轴向磁场 B = 2 T (与 CMS 芯径同量级)
- 粒子从原点附近发射，横动量 p_T 在 [0.5, 20] GeV 对数均匀采样
- 带电粒子在横向走圆弧 (回旋半径 R = p_T / (0.3·B))，z 方向匀速漂移
- 叠加 15% 的噪声 hit

数据格式：每个事件一张 `(N_hits, 5)` 表，列 `[x, y, z, layer, particle_id]`，
`particle_id = -1` 表示噪声。

## 项目结构

```
hep-tracking-gnn/
├── src/
│   ├── toy_detector.py   # 螺旋线 + 探测器模拟 (当前已完成)
│   ├── visualize.py      # xy / rz 投影可视化
│   ├── build_graph.py    # TODO: kNN 碰撞图
│   ├── embedding.py      # TODO:  hit → latent 空间映射
│   ├── edge_classifier.py # TODO: GNN 边分类
│   └── tracking.py       # TODO: 边打分 → 径迹拼接
├── data/                 # 生成的 .npz
├── results/              # 图 / 指标
└── notebooks/
```

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 生成 64 个事件 (~9 万 hits)
python -m src.toy_detector --n-events 64 --out data/toy_hits.npz

# 画 event 0 的 xy + rz 投影
python -m src.visualize --data data/toy_hits.npz --event 0 --out results/event_0.png
```

## 当前进度

- [x] 物理探测器模拟 (螺旋线 hit + 噪声)
- [x] 探测器可视化
- [ ] kNN 碰撞图构建
- [ ] Embedding MLP (同粒子 hit 在 latent 空间拉近)
- [ ] Edge filtering (距离阈值剪枝)
- [ ] GNN edge classifier (判断两 hit 是否同属一条径迹)
- [ ] 径迹后处理 (Union-Find / connected components)
- [ ] 指标：edge purity / reconstruction efficiency / fake rate
- [ ] MPS vs CPU 性能剖析

## 参考

- Exa.TrkX: https://exatrkx.github.io/
- Exa.TrkX NeurIPS'19: https://arxiv.org/abs/2103.16148
- TrackML Kaggle Challenge
