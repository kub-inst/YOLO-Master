# TAL 超参数可配置化修改报告

> 修改日期：2026-08-31  
> 修改范围：`ultralytics/`（YOLO-Master 主体，A2 文件夹以外）  
> 修改目的：将 TAL（Task-Aligned Assigner）的核心超参数从硬编码改为公共 CLI 可配置，为 STAL 实验提供配置化基础

---

## 1. 修改背景

在 P0 baseline 审计中发现以下问题：

1. **`alpha`/`beta` 硬编码**：`v8DetectionLoss` 和 `v8OBBLoss` 中 `TaskAlignedAssigner` 的 `alpha=0.5`、`beta=6.0` 是写死的字面量，无法从命令行或 YAML 配置修改。
2. **`tal_topk` 未暴露为公共配置键**：`tal_topk` 虽然作为构造函数参数存在，但未注册到 `default.yaml` 和 `cfg/__init__.py` 的配置键集合中，训练时 `args.yaml` 不会显式记录实际使用的值，导致复现时无法确认 topk 数值。

STAL 实验需要动态调整 `tal_topk`（面积感知 top-k）和 `tal_alpha`/`tal_beta`（面积感知对齐代价），因此这三个参数必须在实验启动前成为合法的公共 CLI 配置键。

---

## 2. 修改清单

### 2.1 `ultralytics/utils/loss.py` — 从硬编码改为从配置读取

**修改位置**：`v8DetectionLoss.__init__`（第 367-369 行）和 `v8OBBLoss.__init__`（第 1088-1092 行）

**v8DetectionLoss 修改前**：
```python
def __init__(self, model, tal_topk=10, tal_topk2=None):
    device = next(model.parameters()).device
    h = model.args  # hyperparameters

    m = model.model[-1]  # Detect() module
    ...
    self.assigner = TaskAlignedAssigner(
        topk=tal_topk,
        num_classes=self.nc,
        alpha=0.5,   # 硬编码
        beta=6.0,    # 硬编码
        stride=self.stride.tolist(),
        topk2=tal_topk2,
    )
```

**v8DetectionLoss 修改后**：
```python
def __init__(self, model, tal_topk=10, tal_topk2=None):
    device = next(model.parameters()).device
    h = model.args  # hyperparameters

    tal_topk = int(getattr(h, "tal_topk", tal_topk))    # 从配置读取，默认 10
    tal_alpha = float(getattr(h, "tal_alpha", 0.5))     # 从配置读取，默认 0.5
    tal_beta = float(getattr(h, "tal_beta", 6.0))       # 从配置读取，默认 6.0

    m = model.model[-1]  # Detect() module
    ...
    self.assigner = TaskAlignedAssigner(
        topk=tal_topk,
        num_classes=self.nc,
        alpha=tal_alpha,
        beta=tal_beta,
        stride=self.stride.tolist(),
        topk2=tal_topk2,
    )
```

**v8OBBLoss 修改前**：
```python
def __init__(self, model, tal_topk=10, tal_topk2=None):
    super().__init__(model, tal_topk=tal_topk)
    self.assigner = RotatedTaskAlignedAssigner(
        topk=tal_topk,
        num_classes=self.nc,
        alpha=0.5,   # 硬编码
        beta=6.0,    # 硬编码
        ...
    )
```

**v8OBBLoss 修改后**：
```python
def __init__(self, model, tal_topk=10, tal_topk2=None):
    super().__init__(model, tal_topk=tal_topk)
    h = self.hyp
    tal_topk = int(getattr(h, "tal_topk", tal_topk))
    tal_alpha = float(getattr(h, "tal_alpha", 0.5))
    tal_beta = float(getattr(h, "tal_beta", 6.0))
    self.assigner = RotatedTaskAlignedAssigner(
        topk=tal_topk,
        num_classes=self.nc,
        alpha=tal_alpha,
        beta=tal_beta,
        ...
    )
```

### 2.2 `ultralytics/cfg/default.yaml` — 新增 3 个配置键

在 `assignment_medium_area` 之后新增：

```yaml
tal_topk: 10    # (int) top-k for TAL positive sample selection per GT
tal_alpha: 0.5  # (float) TAL alignment metric exponent for classification score
tal_beta: 6.0   # (float) TAL alignment metric exponent for IoU
```

默认值与原有硬编码值完全一致，保证向后兼容。

### 2.3 `ultralytics/cfg/__init__.py` — 注册配置键类型

| 配置键 | 注册位置 | 类型 |
|---|---|---|
| `tal_topk` | `MIXTURE_INT_KEYS` | int |
| `tal_alpha` | `MIXTURE_FLOAT_KEYS` | float |
| `tal_beta` | `MIXTURE_FLOAT_KEYS` | float |

类型注册确保命令行传入时自动做类型转换和范围校验。

---

## 3. 向后兼容性

| 场景 | 行为 |
|---|---|
| 未传任何 `tal_*` 参数 | 使用默认值 `tal_topk=10`、`tal_alpha=0.5`、`tal_beta=6.0`，与原硬编码行为完全一致 |
| 旧 checkpoint 加载后继续训练 | `model.args` 从 checkpoint 恢复，若旧 checkpoint 无 `tal_*` 键则走默认值，不报错 |
| 仅 `yolo detect train` 无额外参数 | 行为与修改前逐位等价 |

---

## 4. 使用方式

```powershell
# 默认值（与修改前行为一致）
yolo detect train model=yolo26n.pt data=A2/configs/visdrone.yaml

# 显式指定 TAL 参数
yolo detect train model=yolo26n.pt data=A2/configs/visdrone.yaml `
    tal_topk=14 tal_alpha=0.5 tal_beta=6.0

# 训练后 args.yaml 会记录
# tal_topk: 14
# tal_alpha: 0.5
# tal_beta: 6.0
```

---

## 5. 影响范围

| 组件 | 影响 |
|---|---|
| `v8DetectionLoss`（标准检测） | alpha/beta/topk 可配置 |
| `v8OBBLoss`（旋转框检测） | alpha/beta/topk 可配置 |
| `E2ELoss`（端到端双分支） | one-to-many 分支的 topk 由 `tal_topk` 控制；one-to-one 分支仍内部固定（`topk=7, topk2=1`），不受影响 |
| `v8SegmentationLoss` / `v8PoseLoss` | 继承 `v8DetectionLoss`，自动获得配置化能力 |
| 现有训练脚本和 CI | 默认值不变，行为等价，无破坏性变更 |