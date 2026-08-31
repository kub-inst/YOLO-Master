# A2：小目标自适应标签分配实验

Link 📎：[A2 实验目录](https://github.com/kub-inst/YOLO-Master/A2)

## 如果有跑完官方协议 baseline 的同学请务必分享给我……求求了

## 目前进度

- [x] 冒烟测试
- [x] （基于自己协议的）A2 基线
- [x] 小样本增加 Stride=4 特征点的训练尝试
- [ ] 比例型 topK 自适应预实验——进行中 🚴
- [ ] （基于官方协议的）A2 基线——进行中 🚴
- [ ] TAL-only Baseline
- [ ] 协议迁移
- [ ] 其他参数的实验（包括 alpha/beta 权重、候选区域扩展等）

## 实验说明

小、中、大目标按原图标注框面积划分：small < (32^2) px²，medium 为 (32^2)–(96^2) px²，large ≥ (96^2) px²。正式 AP 分档使用该口径；训练阶段的正样本统计则在增强后的 640 输入空间计算，两者不直接等同。

| 项目 | 当前设置/状态 |
|---|---|
| 数据与模型 | VisDrone2019-DET，YOLO26n，输入 640，batch=2，AdamW，seed=42 |
| 已完成基线 | 完整训练 50 epoch；标准 COCO `maxDets=100` 下 AP/APs/APm/APl 为 15.19/7.31/23.23/41.98 |
| 评测与选点 | 使用原图面积 COCO 分档；以总体 AP 最优 checkpoint 报告正式结果，`maxDets=300` 仅作密集场景补充 |
| 当前限制 | 目前仅单个 seed；后续增益需要重复实验或置信区间验证 |
| 官方协议 | 尚未迁移，后续将单独建立官方协议基线，避免与现有结果混用 |

## 小样本增加 Stride=4 特征点的训练尝试

该尝试在原有 P3/8、P4/16、P5/32 检测头之外增加 P2/4 检测头，并仅向面积小于 1024 px² 的小目标开放 P2 候选点。其出发点是：小目标在较稀疏特征层中可参与分配的候选点不足，部分目标甚至没有正样本；更密集的 stride=4 网格可提高候选覆盖。

训练统计确实显示小目标正样本覆盖改善，但 APs 反而下降。初步分析是：虽然增加了可选特征点，但固定 top-10 可能纳入较多低质量候选；与此同时，小目标又需要从周围区域扩充信息以获得更多特征。这一矛盾尚未得到有效解决，因此暂时不继续该方向。

实验结果图如下：

<img width="1568" height="776" alt="P2 stride-4 experiment result" src="https://github.com/user-attachments/assets/da586ca7-a2b9-4c12-ac39-7158d8f63a5e" />

## 比例型 topK 自适应预实验

该方向仅对小目标调整 TAL 的正样本选择数：不再固定选择 K 个候选点，而是按候选区域内候选点数 (x) 取 (K=\lceil\lambda x\rceil)，其中 (\lambda) 为比例系数；中、大目标及其余训练设置保持不变。目的在于让不同尺寸的小目标获得与其可用候选数相匹配的监督信号。

在最早的猜想验证阶段，我进行了一个预预实验，在全集上用 $\lambda=0.8$ 跑了 10 个 epoch，结果如下图所示。

<img width="1781" height="1243" alt="Dynamic topK lambda 0.8 preliminary result" src="https://github.com/user-attachments/assets/2afe5d7d-2811-4aab-aabb-145826d5a055" />

虽然整体上区别不是很明显，但是个别 epoch 上相较 baseline 效果还行，考虑进一步扩大实验规模进行验证。

当前进度：

- [x] 预预实验
- [ ] 0.05 stride 的多组 10 epoch 验证
- [ ] 更多 epoch
- [ ] 精细化参数
