# YOLO26n VisDrone 1-Epoch Baseline（8.24 准入检查）

使用 YOLO26n 官方预训练权重在 VisDrone2019-DET 上跑 1 epoch 冒烟，
验证训练管线、assigner 注入点与指标采集日志均可正常工作。

## 代码复用说明

本 baseline 未新增任何训练代码，全部复用项目已有脚本：

- 训练脚本：`examples/YOLO-Master-EsMoE-VisDrone-Edge/scripts/train_visdrone.py`（项目已有）
- 数据集配置：[configs/visdrone.yaml](configs/visdrone.yaml)
- 训练配置：[configs/train.yaml](configs/train.yaml)
- 模型权重：`yolo26n.pt`（ultralytics 官方预训练权重）

## 复现命令

```bash
python examples/YOLO-Master-EsMoE-VisDrone-Edge/scripts/train_visdrone.py \
    --model yolo26n.pt \
    --epochs 1 \
    --batch 2 \
    --device 0 \
    --workers 0
```

| 参数 | 值 | 说明 |
|------|-----|------|
| model | yolo26n.pt | YOLO26 Nano COCO 预训练权重（2.57M 参数） |
| epochs | 1 | 冒烟验证，非正式训练 |
| batch | 2 | 显存限制，单卡可调 |
| device | 0 | 单卡 GPU |
| workers | 0 | Windows 多进程兼容 |

## assigner 与配置注入点

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

## 结果

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

## 准入检查项

- [x] VisDrone 子集 1 epoch 冒烟
- [x] assigner 注入点：`ultralytics/utils/loss.py` → `v8DetectionLoss.__init__` → `TaskAlignedAssigner`
- [x] TAL 变体：`TaskAlignedAssigner` / `RotatedTaskAlignedAssigner` / `E2EDetectLoss` / `TVPDetectLoss`
- [x] 指标采集日志：`results.csv` 末行含 box_loss / cls_loss / dfl_loss / mAP50 / mAP50-95
- [x] 数据统计回调：`ultralytics/utils/callbacks/` → `save_metrics` / `moe_diag`
- [x] 训练配置：`visdrone.yaml` + `yolo26n.pt` + 命令行超参

## 风险与降级

| 风险 | 现象 | 降级方案 |
|------|------|----------|
| Windows 多进程虚拟内存不足 | `workers=8` 报 `页面文件太小` | `--workers 0` |
| GPU 显存不足 | auto_batch 搜索时 OOM | `--batch 2` 或 `--batch 1` |
| 单 epoch mAP 无参考意义 | mAP ≈ 0 | 标注为冒烟测试，正式训练需更多epoch |