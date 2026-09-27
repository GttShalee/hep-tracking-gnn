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
- [x] 探测器可视化 (xy + rz)
- [x] kNN 碰撞图构建 (scipy cKDTree, k=8)
- [x] Baseline edge classifier (NodeEncoder MLP + EdgeClf MLP, 无 message passing)
- [x] 训练 pipeline (MPS 后端, BCE + pos_weight)
- [ ] 升级到真正的 GNN (GCN/GAT message passing)
- [ ] Edge filtering (embedding 空间距离剪枝)
- [ ] 径迹后处理 (Union-Find / connected components)
- [ ] 指标：reconstruction efficiency / fake rate
- [ ] MPS vs CPU 性能剖析

## 训练结果 (baseline, 30 epochs, M1 Pro MPS)

| epoch | train loss | val AUC | val acc |
|---|---|---|---|
| 1 | 1.159 | 0.577 | 0.181 |
| 10 | 0.518 | 0.935 | 0.847 |
| 20 | 0.231 | 0.983 | 0.941 |
| 30 | 0.169 | **0.989** | 0.956 |

数据: 64 events × ~1382 hits, k=8 kNN graph, edge signal fraction ~17.6%。
注意这是 **无 message passing 的纯 MLP baseline**——只用每个 hit 的坐标和边上的简单特征，
还没上 GNN。下一步换 GCN/GAT 后 AUC 应该能逼近 0.995+。

## 参考

- Exa.TrkX: https://exatrkx.github.io/
- Exa.TrkX NeurIPS'19: https://arxiv.org/abs/2103.16148
- TrackML Kaggle Challenge
