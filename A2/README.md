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

## 实验协议与当前基线

小、中、大目标按原图标注框面积划分：small < (32^2) px²，medium 为 (32^2)–(96^2) px²，large ≥ (96^2) px²。正式 AP 分档使用该口径；训练阶段的正样本统计则在增强后的 640 输入空间计算，两者不直接等同。

| 项目 | 当前设置/状态 |
|---|---|
| 数据与模型 | VisDrone2019-DET，YOLO26n，输入 640，batch=2，AdamW，seed=42 |
| 已完成基线 | 完整训练 50 epoch；标准 COCO `maxDets=100` 下 AP/APs/APm/APl 为 15.19/7.31/23.23/41.98 |
| 评测与选点 | 使用原图面积 COCO 分档；以总体 AP 最优 checkpoint 报告正式结果，`maxDets=300` 仅作密集场景补充 |
| 当前限制 | 目前仅单个 seed；后续增益需要重复实验或置信区间验证 |
| 官方协议 | 尚未迁移，后续将单独建立官方协议基线，避免与现有结果混用 |

## 冒烟测试：1-Epoch Baseline（8.24 准入检查）

使用 YOLO26n 官方预训练权重在 VisDrone2019-DET 上跑 1 epoch 冒烟，验证训练管线、assigner 注入点与指标采集日志均可正常工作。

### 代码复用说明

本 baseline 未新增任何训练代码，全部复用项目已有脚本：

- 训练脚本：`examples/YOLO-Master-EsMoE-VisDrone-Edge/scripts/train_visdrone.py`（项目已有）
- 数据集配置：[configs/visdrone.yaml](configs/visdrone.yaml)
- 训练配置：[configs/train.yaml](configs/train.yaml)
- 模型权重：`yolo26n.pt`（ultralytics 官方预训练权重）

### 复现命令

```bash
python examples/YOLO-Master-EsMoE-VisDrone-Edge/scripts/train_visdrone.py \\
    --model yolo26n.pt \\
    --data A2/configs/visdrone.yaml \\
    --epochs 1 \\
    --batch 2 \\
    --device 0 \\
    --workers 0
```

| 参数 | 值 | 说明 |
|------|-----|------|
| model | yolo26n.pt | YOLO26 Nano COCO 预训练权重（2.57M 参数） |
| epochs | 1 | 冒烟验证，非正式训练 |
| batch | 2 | 显存限制，单卡可调 |
| device | 0 | 单卡 GPU |
| workers | 0 | Windows 多进程兼容 |

## TAL 标签分配与配置注入点

**assigner 定义**：`ultralytics/utils/tal.py#L14`

```python
class TaskAlignedAssigner(nn.Module):
    topk: int          # 候选框数，默认 10
    alpha: float       # 分类权重，默认 0.5
    beta: float        # 定位权重，默认 6.0
```

**配置注入点**：`ultralytics/utils/loss.py#L371-L378`

```python
# v8DetectionLoss.__init__ 中
self.assigner = TaskAlignedAssigner(
    topk=tal_topk,          # 可配置
    num_classes=self.nc,
    alpha=0.5,              # 可配置
    beta=6.0,               # 可配置
    stride=self.stride.tolist(),
    topk2=tal_topk2,        # 可配置
)
```

修改 `topk`、`alpha`、`beta` 即可调整标签分配策略。TAL 变体还包括：

| 类 | 文件 | 用途 |
|----|------|------|
| `TaskAlignedAssigner` | `tal.py` | 水平框检测 |
| `RotatedTaskAlignedAssigner` | `tal.py` | 旋转框检测 (OBB) |
| `E2EDetectLoss` | `loss.py:1181` | YOLO10 端到端 |
| `E2ELoss` | `loss.py:1199` | YOLO11/12 端到端 |
| `TVPDetectLoss` | `loss.py:1234` | YOLO26 TVP 变体 |

### 冒烟测试结果

产物目录：[results/](results/)

| 指标 | 值 |
|------|-----|
| mAP50 | 0.00043 |
| mAP50-95 | 0.0002 |
| train/box_loss | 2.599 |
| train/cls_loss | 5.102 |
| train/dfl_loss | 0.005 |
| precision | 0.051 |
| recall | 0.013 |
| 耗时 | 1067.6s（~17.8min） |

> 1 epoch 仅验证管线通畅，mAP 接近零属正常。正式训练需 50-300 epoch。

### 准入检查项

- [x] VisDrone 子集 1 epoch 冒烟
- [x] assigner 注入点：`ultralytics/utils/loss.py` → `v8DetectionLoss.__init__` → `TaskAlignedAssigner`
- [x] TAL 变体：`TaskAlignedAssigner` / `RotatedTaskAlignedAssigner` / `E2EDetectLoss` / `TVPDetectLoss`
- [x] 指标采集日志：`results.csv` 末行含 box_loss / cls_loss / dfl_loss / mAP50 / mAP50-95
- [x] 数据统计回调：`ultralytics/utils/callbacks/` → `save_metrics` / `moe_diag`
- [x] 训练配置：`visdrone.yaml` + `yolo26n.pt` + 命令行超参

### 风险与降级

| 风险 | 现象 | 降级方案 |
|------|------|----------|
| Windows 多进程虚拟内存不足 | `workers=8` 报 `页面文件太小` | `--workers 0` |
| GPU 显存不足 | auto_batch 搜索时 OOM | `--batch 2` 或 `--batch 1` |
| 单 epoch mAP 无参考意义 | mAP ≈ 0 | 标注为冒烟测试，正式训练需更多epoch |

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
