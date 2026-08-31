# A2：STAL 式小目标自适应标签分配——项目报告与实践计划

> 基于仓库提交 `cfd9966`（`add YOLO26n VisDrone 1-epoch baseline`）的只读审计结果。  
> 审计日期：2026-08-30；主实验对象：YOLO26n + VisDrone2019-DET。

## 1. 执行摘要

YOLO-Master 是基于 Ultralytics 的实时目标检测框架，在标准 YOLO 的 Backbone–Neck–Head–Loss–Validator
训练链路上，扩展了 ES-MoE、MoA、MoT、MoLoRA、Sparse SAHI、MoE 诊断/剪枝和 Agent Skill。A2 的研究对象
不是 MoE 路由本身，而是检测损失中的标签分配器：让小目标按面积获得更合适的正样本候选，从而验证“标签分配
改变”能否稳定转化为 AP_small 和 Recall 提升。

当前 A2 已完成 VisDrone 全量训练集上的 1 epoch 管线冒烟，并留下 checkpoint、曲线和 `results.csv`。但它只证明
训练能跑通，尚未完成 P0：没有小/中/大分档指标，没有逐 epoch 正样本统计，也没有可由配置开关控制的 STAL。
此外，A2 文档中“`topk/alpha/beta` 可配置”的说法与代码不完全一致：当前 `alpha=0.5`、`beta=6.0` 在
`v8DetectionLoss` 中硬编码，`topk` 也未暴露为公共训练配置。

推荐的最小研究路线是：

1. 固定 YOLO26n、数据、增强、seed 和评测口径，先建立可解释的 P0 基线；
2. 只对 one-to-many 分支启用面积感知动态 top-k，one-to-one 分支保持原样；
3. 同时记录候选、冲突消解后正样本、零覆盖 GT、每尺度正样本和面积分档；
4. 先做一次严格 on/off 对照，再扫描阈值、放宽幅度和 warmup；
5. 最后以 3 seeds 和按图像配对 bootstrap 置信区间判断是否达到 `AP_small +1.0`。

## 2. YOLO-Master 框架简要介绍

### 2.1 总体结构

一次检测训练的主链路如下：

```text
dataset YAML / labels
        ↓
ultralytics/data：加载、LetterBox、Mosaic/MixUp 等增强
        ↓
ultralytics/nn/tasks.py：根据模型 YAML 组装 DetectionModel
        ↓
Backbone（特征提取，可含 ES-MoE/MoA/MoT）
        ↓
Neck（FPN/PAN 多尺度融合）
        ↓
Detect Head（P3/P4/P5；P2 变体另含 stride=4）
        ↓
ultralytics/utils/loss.py：解码预测并调用 TAL assigner
        ↓
ultralytics/utils/tal.py：候选筛选、task-aligned 排序、冲突消解
        ↓
box / cls / dfl（YOLO26 中 reg_max=1，实际为 DFL-free 路径）
        ↓
trainer：反向传播、EMA、逐 epoch 验证、results.csv/checkpoint
        ↓
detect validator：P/R/mAP；外部 COCO evaluator 可给 AP_small/medium/large
```

### 2.2 与 A2 直接相关的目录

| 位置 | 作用 | A2 关注点 |
|---|---|---|
| `ultralytics/utils/tal.py` | `TaskAlignedAssigner` 与旋转框变体 | STAL 的核心实现位置 |
| `ultralytics/utils/loss.py` | 构造 assigner、产生 `fg_mask/target_gt_idx` | 配置注入与统计采集 |
| `ultralytics/nn/tasks.py` | 为 DetectionModel 初始化 criterion | YOLO26 end-to-end 双分支入口 |
| `ultralytics/cfg/default.yaml` | CLI/Python API 的公共参数单一事实源 | 增加 `stal_*` 配置 |
| `ultralytics/cfg/__init__.py` | 配置键类型和范围校验 | 注册 bool/int/float/string 键 |
| `ultralytics/engine/trainer.py` | batch/epoch 生命周期、CSV 与 callback | 汇总逐 epoch 分配统计 |
| `ultralytics/models/yolo/detect/val.py` | 检测验证和 COCO JSON 评测 | 分档 AP 的接入或离线评测 |
| `tests/test_tal_mps_regression.py` | 当前唯一直接 TAL 回归测试 | 扩展 STAL 边界与兼容测试 |
| `A2/` | 本课题配置、冒烟结果和文档 | 复现实验包与最终交付物 |

### 2.3 YOLO-Master 的扩展能力

- **ES-MoE**：按输入内容路由到不同专家，以动态计算改善精度–延迟权衡；核心在
  `ultralytics/nn/modules/moe/`。
- **MoA/MoT**：分别对注意力头和 Transformer 专家做内容感知混合。
- **MoLoRA/V-PEFT**：在 `ultralytics/nn/peft/` 中提供参数高效微调和可审计的适配器规划。
- **Sparse SAHI/CW-NMS**：面向高分辨率、小目标和密集场景的推理后处理能力。
- **Agent Skill**：通过 `agent/scripts/run_yolo_master_skill.py` 统一训练、验证、导出、诊断和实验清单。

这些能力说明仓库适合做端到端研究，但 A2 首轮实验不应同时改变 MoE、P2 Head 或 SAHI，否则无法把收益归因于
标签分配。

## 3. 当前 A2 状态审计

### 3.1 已有成果

- 当前提交：`cfd9966 add YOLO26n VisDrone 1-epoch baseline`。
- 数据完整：train/val/test 分别为 6471/548/1610 张图，三者均有对应标签文件。
- `A2/results/` 包含 `best.pt`、`last.pt`、`last_healthy.pt`、训练/验证可视化和 1 行 epoch 指标。
- 1 epoch 结果：mAP50=0.00043，mAP50-95=0.00020，Recall=0.01308；只能作为连通性证据。
- 已定位 TAL 调用链：`DetectionModel.init_criterion` → `E2ELoss/v8DetectionLoss` →
  `TaskAlignedAssigner.forward`。

### 3.2 数据画像

按 YOLO 标签的归一化面积乘以 `640×640` 做近似统计：

| split | 实例数 | 小目标 `<32²` | 中目标 `32²–96²` | 大目标 `≥96²` | 小目标占比 | 面积中位数 |
|---|---:|---:|---:|---:|---:|---:|
| train | 343,205 | 292,751 | 48,603 | 1,851 | 85.30% | 约 189 px² |
| val | 38,759 | 33,280 | 5,349 | 130 | 85.86% | 约 226 px² |

这是基于 640 方形输入的近似画像，不等同于 COCO 在原图像素空间的官方面积口径。正式报告必须明确：

- **训练分配分档**：使用增强和 resize 后进入 loss 的 GT 像素面积；
- **验证 AP 分档**：使用原图坐标和固定 COCO 面积阈值；
- 两种口径用途不同，不应混写。

类别也高度不均衡：训练集中 car 144,867 个、pedestrian 79,337 个，而 awning-tricycle 仅 3,246 个。
因此除面积分档外，最终至少要检查主要类别和尾部类别是否出现系统性退化。

### 3.3 环境状态

| 项目 | 实测状态 |
|---|---|
| conda 环境 | `yolo_master`，Python 3.11.15 |
| 本地包 | Ultralytics 8.4.101，加载自当前仓库 |
| PyTorch | 2.12.1+cu132 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，单卡可用 |
| 数据 | `D:\coding\datasets\VisDrone` |
| 测试工具 | 环境内当前无 `pytest` 命令，开发测试前需补齐 dev 依赖 |

历史 smoke 用时 1067.58 秒/epoch。按此粗估，50 epoch 约 14.8 小时，100 epoch 约 29.7 小时；多 seed
实验必须排队执行，并保留中间 checkpoint，不能把所有关键实验压到最后两天。

### 3.4 必须先修正的复现问题

> **状态：2026-08-31 全部修复完成。**

1. ~~`A2/configs/train.yaml` 当前只是记录文件，训练脚本并不会读取它；脚本内仍硬编码增强和优化器参数。~~
   → **说明**：P0 使用 `yolo detect train` 命令行启动，不依赖 `train_visdrone.py`。`train.yaml` 作为参数记录文件保留，`args.yaml` 才是实际训练参数的真值来源。
2. ~~A2 README 的复现命令未传 `--data A2/configs/visdrone.yaml`；历史 `args.yaml` 实际使用的是 examples 下的 VisDrone 配置，输出项目也在 examples 的 runs 目录。~~
   → **已修复**：README 已添加 `--data A2/configs/visdrone.yaml`。
3. ~~`train.yaml` 注释称 smoke 关闭增强，但配置中 `mosaic=1.0`、`mixup=0.1`，实际并未关闭。~~
   → **已修复**：注释改为 `# Augmentation (enabled for full training; set mosaic=0.0 mixup=0.0 for smoke test)`。
4. ~~`alpha/beta` 当前硬编码，`topk` 只是 Python 构造参数；三者都不是合法的公共 CLI 配置键。~~
   → **已修复**：`ultralytics/utils/loss.py` 中 `v8DetectionLoss` 和 `v8OBBLoss` 的 `alpha`/`beta`/`topk` 已改为从 `model.args` 读取（`tal_alpha`/`tal_beta`/`tal_topk`），默认值保持 0.5/6.0/10。`default.yaml` 和 `cfg/__init__.py` 已注册配置键。
5. ~~checkpoint 中虽记录 `topk=8`，公共代码 FAQ 明确说明内部 `topk/o2m/cls_w` 不从 checkpoint 训练参数读取；当前公开 `E2ELoss` 实际构造 one-to-many `topk=10`，不能把 checkpoint 的 8 当成本次基线。~~
   → **已修复**：`tal_topk` 已注册为公共配置键，训练时 `args.yaml` 将显式记录实际使用的 `tal_topk`。
6. ~~原生 validator 的常规 P/R/mAP 不按面积分档；`AP_small/medium/large` 只在 COCO/LVIS JSON evaluator 路径更新，VisDrone YAML 默认不会自动进入该路径。~~
   → **已解决**：通过 `A2/scripts/evaluate_p0_checkpoints.py` 和 `evaluate_visdrone_area.py` 离线脚本实现面积分档评测，输出 `p0_checkpoint_area_metrics.json` 和 `p0_area_metrics.json`。

## 4. TAL 现有机制与 STAL 切入点

### 4.1 当前 TAL 算法

对每个 GT，当前实现执行：

1. 找到 anchor center 位于 GT 内的候选；
2. 计算 `alignment = class_score^alpha × CIoU^beta`；
3. 每个 GT 选择固定 `topk`；
4. 若一个 anchor 命中多个 GT，按重叠/对齐结果消解冲突；
5. 可用 `topk2` 再筛一次；
6. 产生 `target_scores`、`fg_mask` 和 `target_gt_idx`。

仓库已经存在一个隐式小目标保护：若 GT 的宽或高小于最小 stride（YOLO26 为 8），候选区域会把对应维度
扩大到 `stride_val`（通常为 16）。A2 的 STAL 必须在实验说明中把这个既有机制列为 baseline，而不能声称
“原始 TAL 对小目标完全没有特殊处理”。

### 4.2 YOLO26 双分支约束

YOLO26n 的 `end2end=True`，stride 为 `[8,16,32]`。公开 `E2ELoss` 使用：

- one-to-many：`topk=10`；
- one-to-one：先 `topk=7`，再用 `topk2=1` 收敛到单匹配。

首版 STAL 应仅作用于 one-to-many。one-to-one 是实际 NMS-free 推理头的监督路径，贸然增加最终正样本会破坏其
一对一语义；而 one-to-many 更适合用于提高小目标的监督覆盖。后续只有在 one-to-many 结果充分后，才研究
“面积感知候选集是否帮助 one-to-one 最终匹配”。

### 4.3 推荐的最小 STAL 定义

令增强后输入空间中的 GT 面积为 `A=(x2-x1)(y2-y1)`，基础 top-k 为 `k0`，最大放宽为 `delta_k`：

```text
smallness(A) = clamp((T_high - A) / (T_high - T_low), 0, 1)
warmup(e)    = 0                                  , e < start
               linear/cosine ramp 0 → 1          , start ≤ e < end
               1                                  , e ≥ end
k(A,e)       = round(k0 + delta_k * smallness(A) * warmup(e))
```

实现上先计算 batch 内统一的 `max_k`，再用每个 GT 的 `k(A,e)` 对排名 mask，避免逐 GT Python 循环。首轮建议：

| 参数 | 初始值 | 说明 |
|---|---:|---|
| `stal_enabled` | false | 默认关闭，保证完全向后兼容 |
| `stal_branch` | `one2many` | 首版不改 one-to-one |
| `stal_area_low` | 256 | 极小目标近似阈值，可由数据分位数修正 |
| `stal_area_high` | 1024 | COCO small 上界 `32²` |
| `stal_topk_delta` | 4 | `k0=10` 时小目标最多到 14 |
| `stal_warmup_epochs` | 5 | 前 5 epoch 从 0 渐进到全幅 |
| `stal_schedule` | `linear` | 首版最易解释，后续比较 cosine |
| `stal_stats` | true | 研究阶段开启逐 epoch 统计 |

面积阈值应基于进入 loss 的像素框，而不是原始 YOLO 归一化标签。Mosaic、scale 和 LetterBox 会改变目标实际尺寸，
STAL 应响应模型真正看到的目标。

### 4.4 主路线与备选路线

**主路线：动态 top-k。** 改动局部、语义直观，最直接回答“增加小目标正样本覆盖是否提升 AP_small/Recall”。

**备选路线：面积感知对齐代价。** 对小目标降低 IoU 指数 `beta` 或对 alignment 乘面积权重，使分类证据在小框
定位噪声较大时占比更高。该路线更连续，但更难解释，也可能改变正样本质量；只在动态 top-k 无收益或冲突率过高时
启用。

**暂不作为 STAL 主实验：** P2 Head、1280 输入、SAHI、MoE 模型切换。这些都可能提升小目标性能，但会引入架构、
分辨率或推理策略混杂变量。

## 5. 实现设计

### 5.1 配置层

在 `ultralytics/cfg/default.yaml` 增加所有 `stal_*` 键，并在 `ultralytics/cfg/__init__.py` 的 bool/int/float/string
集合注册类型与下界。所有默认值必须使旧模型逐位复现原行为。

建议允许命令行直接运行：

```powershell
conda run -n yolo_master yolo detect train `
  model=yolo26n.pt data=A2/configs/visdrone.yaml `
  cfg=A2/configs/train.yaml `
  stal_enabled=True stal_branch=one2many `
  stal_area_low=256 stal_area_high=1024 `
  stal_topk_delta=4 stal_warmup_epochs=5
```

### 5.2 loss 与 assigner 层

1. `v8DetectionLoss` 从 `model.args` 读取 STAL 配置并传给 assigner；关闭时仍走固定 top-k。
2. `E2ELoss` 为 one-to-many 和 one-to-one 显式传入分支名，避免靠 top-k 数值猜分支。
3. `TaskAlignedAssigner` 新增面积计算、epoch 进度、动态 k mask 和只读统计快照。
4. trainer 在 `on_train_epoch_start` 更新 assigner 的 epoch/总 epoch，支持 resume 后正确恢复 warmup。
5. CPU OOM fallback、AMP、CUDA 和 DDP 下统计都必须使用 detach 后的基础数值类型，不能保留计算图。

### 5.3 正样本统计设计

每个 epoch 至少输出以下字段到独立 `assignment_stats.csv`，必要时同步到主 `results.csv`：

| 指标 | 含义 |
|---|---|
| `pos/total` | 冲突消解后的正 anchor 总数 |
| `pos/per_gt_mean,p50,p90` | 每 GT 获得的正样本分布 |
| `coverage/zero_gt_rate` | 没有最终正样本的 GT 比例 |
| `pos/small,medium,large` | 按训练输入空间面积分档的正样本数 |
| `pos_per_gt/small,medium,large` | 各面积档平均每 GT 正样本 |
| `candidate/pre_conflict` | top-k 后、冲突消解前的候选数 |
| `conflict/rate` | 被多个 GT 竞争的 anchor 比例 |
| `pos/p3,p4,p5` | 按 stride/特征层统计最终正样本 |
| `quality/mean_iou` | 最终正样本与 GT 的平均 IoU |
| `quality/mean_target_score` | 归一化 target score 均值 |

关键判断不是“`pos/total` 越大越好”，而是小目标 `zero_gt_rate` 是否下降、`pos_per_gt/small` 是否合理上升、
同时 `mean_iou` 和中大目标 AP 是否没有明显恶化。

### 5.4 分档评测

建议在 A2 内新增独立、确定性的评测脚本，不直接把 VisDrone 伪装成内置 COCO 数据集：

1. 将 val 的 YOLO GT 和图像尺寸转成 COCO 格式 JSON；category id 固定为 1–10；
2. `yolo val save_json=True` 生成预测 JSON；
3. 用 `faster-coco-eval` 在原图坐标计算 AP/AP50/AP75、AP_small/medium/large、AR_small/medium/large；
4. 另行保留 VisDrone 官方评测口径，避免把 COCO-style AP 与 VisDrone 官方 AP 混为一谈；
5. 评测脚本记录数据列表 SHA-256、checkpoint SHA-256、commit、配置、依赖版本和命令。

如果原始 VisDrone 的 ignore/truncation/occlusion 标注已在 YOLO 转换时丢弃，COCO-style 结果只能称为“转换后 YOLO
标签口径”。正式答辩应同时说明这一限制。

## 6. 测试与质量门

### 6.1 单元测试

新增 `tests/test_stal_assigner.py`，覆盖：

- `stal_enabled=False` 与原 TAL 输出完全一致；
- 空 GT、单 GT、多 GT、极小/边界/极大面积；
- `A=area_low/area_high` 的边界值；
- warmup 前、中、后以及 resume；
- 动态 k 不超过 anchor 数，`area_low < area_high`；
- 多 GT 冲突消解后一个 anchor 只属于一个 GT；
- one-to-one 最终正样本语义不变；
- FP16/BF16 输入不产生 NaN/Inf；
- CPU 与 CUDA 的 mask/计数一致；
- DDP 统计汇总不重复计数；
- 旋转框 assigner 不受影响，除非显式支持。

同时扩展配置完整性测试，并运行现有 `tests/test_tal_mps_regression.py`。当前 conda 环境缺 `pytest`，开始开发前应执行
`python -m pip install -e ".[dev]"` 或只补齐仓库规定的测试依赖。

### 6.2 每次变更后的验证

```powershell
conda run -n yolo_master ruff check ultralytics/ tests/ A2/
conda run -n yolo_master ruff format --check ultralytics/ tests/ A2/
conda run -n yolo_master pytest tests/test_stal_assigner.py tests/test_tal_mps_regression.py -v
conda run -n yolo_master pytest tests/test_default_config_integrity.py -v
conda run -n yolo_master python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
```

训练 smoke 采用固定小子集、固定 1 epoch，验收不是 mAP，而是：无异常、STAL on/off 都能跑、统计列非空、关闭时
结果与原路径一致、开启时小目标平均正样本数按预期增加。

## 7. 实验设计

### 7.1 冻结的公共条件

- 模型：`yolo26n.pt`；首轮不换 YOLO-Master MoE/P2 模型。
- 数据：同一 VisDrone train/val 列表和同一转换版本。
- 输入：640；若后续做 1280，只作为独立消融。
- optimizer、LR、增强、batch、epoch、patience 全部由一个实际被消费的 YAML 管理；其中 A2 固定 `patience=0`，禁用早停以保证每次完整跑满声明 epoch。
- seed：首轮 42；确认候选后使用 0/1/2 三个 seeds。
- 推理：同一 conf、IoU、max_det、end-to-end 设置。
- 选择 checkpoint 的规则预注册，例如以 val mAP50-95 最优为准，不能看完 AP_small 后临时挑 epoch。

### 7.2 最小实验矩阵

| ID | STAL | delta-k | area high | warmup | 目的 |
|---|---|---:|---:|---:|---|
| B0 | off | 0 | — | — | 严格基线 |
| S1 | on | 4 | 1024 | 5 | 主方案 |
| S2 | on | 2 | 1024 | 5 | 较保守放宽 |
| S3 | on | 6 | 1024 | 5 | 较激进放宽 |
| S4 | on | 4 | 576 | 5 | 更小面积阈值 |
| S5 | on | 4 | 1600 | 5 | 更宽面积阈值 |
| S6 | on | 4 | 1024 | 0 | 无 warmup 稳定性对照 |
| S7 | on | 4 | 1024 | 10 cosine | 曲线消融 |

先让 B0/S1 完整跑完；仅当 S1 的分配统计方向正确且 AP_small 不劣于基线时，才跑 S2–S7。这样避免把算力耗在
机制本身未生效的网格扫描上。

### 7.3 成功标准

**P0：**

- 至少一个正式 baseline（不能是 1 epoch）；
- AP_small/medium/large 和 AR_small/medium/large；
- 每 epoch 正样本演化曲线和配置/commit/checkpoint 哈希；
- 统一评测脚本与数据口径说明。

**P1：**

- STAL 可配置、默认关闭、测试通过；
- 主指标 `AP_small` 绝对提升至少 1.0 个百分点；
- 3 seeds 的均值和标准差；
- 同一验证图像上的 paired bootstrap 95% CI，建议 1000 次重采样；
- Recall/AR_small 同向改善，整体 mAP 和中大目标不存在不可接受回退；
- 正样本统计证明提升机制确实发生，而非偶然的训练波动。

建议预注册回退门槛：总体 mAP50-95 下降不超过 0.3，中目标 AP 下降不超过 0.5；大目标样本仅 130 个，波动很大，
应报告区间和样本量，不宜作为唯一否决条件。

## 8. 从 8.30 到 9.14 的实践排期

### 8.30：审计与口径冻结

- 修正 A2 的配置引用和复现说明；生成 dataset manifest。
- 安装 dev/eval 依赖，运行 TAL、配置完整性和 Agent Skill quick 测试。
- 把 1 epoch 历史结果明确标为 smoke，不再作为 baseline 数字。
- 冻结 B0 的命令、模型、数据、增强、seed 和评测协议。

**当日证据：** 环境报告、测试日志、冻结配置、数据统计表、B0 dry-run 命令。

### 8.31：P0 仪表与正式 baseline 启动

- 先实现不改变分配结果的统计探针和 `assignment_stats.csv`。
- 完成 YOLO→COCO GT 转换及分档 evaluator，小样本上人工核对坐标与类别映射。
- 运行 1 epoch instrumentation smoke；确认探针开关不会改变 loss/fg_mask。
- 启动 B0 正式训练，建议先 50 epoch；若曲线仍明显上升再延长到 100。

**当日证据：** 分档评测 JSON、首条正样本日志、B0 checkpoint 和训练日志。

### 9.1–9.2：STAL 最小实现

- 增加配置键、动态 k、one-to-many 分支控制和 warmup。
- 完成单元测试、CUDA smoke、resume smoke、on/off 等价测试。
- 用固定 batch 做机制验证：小目标 k 增加、中大目标不变、冲突率可解释。

**退出条件：** off 路径等价；on 路径统计按设计改变；无 NaN/OOM；测试全绿。

### 9.3–9.5：首轮完整对照

- 用同一训练预算运行 S1；每日保留 `last_healthy.pt`、日志和统计 CSV。
- 每 5–10 epoch 检查 AP_small、AR_small、zero-GT rate、mean IoU 和冲突率。
- 若出现训练抖动，先延长 warmup 或减小 delta-k，不同时改面积阈值和优化器。

**中止条件：** 连续多个验证点 overall/small AP 显著恶化，或冲突率/低质量正样本异常上升。

### 9.6–9.7：中期结论与 PR 草稿

- 对 B0/S1 画正样本演化、AP 曲线和面积档对照表。
- 做一次错误案例审查：密集行人、车辆群、极小框、遮挡与尾部类别。
- 根据机制证据选择最多 2–3 个消融，不做无边界全网格。
- 建立 PR 草稿，写清默认关闭、兼容性、测试和已知限制。

### 9.8–9.10：关键消融

- 优先顺序：delta-k → area threshold → warmup 曲线。
- 每个实验只改一个变量；统一从同一预训练权重开始。
- 如果动态 top-k 无收益，启动备选的面积感知 beta/代价路线，但单独编号，不与 top-k 混合。

### 9.10–9.12：多 seed 与统计证明

- 对 B0 和最佳 STAL 配置运行 seeds 0/1/2。
- 汇总 mean±std、paired delta、bootstrap 95% CI。
- 复跑最佳 checkpoint 的统一 evaluator；校验结果可由全新输出目录复现。
- 若时间不足，优先完成 B0/最佳方案的多 seed，不再扩第二数据集。

### 9.12：冻结复现包

- 冻结代码 commit、配置、数据 manifest、命令、环境、checkpoint 哈希和结果表。
- 从空结果目录做一次 quick reproduction；禁止 9.12 后再改主指标口径。
- 补齐失败实验和负结果，避免只保留成功曲线。

### 9.13–9.14：答辩与交付

- 最终 PR：STAL 模块、配置、测试、文档、默认关闭。
- 报告：主表、消融表、置信区间、正样本曲线、案例图和限制。
- 演示：同一配置一键切换 STAL on/off，展示日志和分档 evaluator。
- 明确未覆盖事项：官方 ignore 标注口径、第二数据集、seg/pose 或 P2/SAHI 联合实验。

## 9. 每日实验运行规范

每个 run 必须保存：

```text
A2/runs/<experiment_id>/
├── args.yaml
├── environment.json
├── dataset_manifest.json
├── command.txt
├── results.csv
├── assignment_stats.csv
├── area_metrics.json
├── predictions.json
├── plots/
└── weights/
```

命名建议：`b0_y26n_vd640_s42`、`s1_dk4_a1024_w5_s42`。禁止使用 `exp2/final_new` 等不可读名称。
每次实验启动前记录 `git rev-parse HEAD` 和 `git diff --stat`；正式结果应来自干净或明确保存 patch 的工作树。

## 10. 风险、监控与降级

| 风险 | 早期信号 | 处理 |
|---|---|---|
| 正样本增多但质量下降 | mean IoU/target score 下降，cls loss 上升 | 减小 delta-k；提高 area gate；加长 warmup |
| 密集 GT 冲突加剧 | conflict rate 上升、zero-GT rate 不降反升 | 限制 max-k；研究冲突感知上限 |
| 训练抖动 | loss/AP 剧烈波动或 NaN | linear warmup 5→10；先禁用 AMP 复现问题 |
| AP_small 提升但整体退化 | 中目标 AP 或整体 mAP 下滑 | 收窄 area high；仅对极小目标放宽 |
| 指标口径不兼容 | 原生 val 与离线 evaluator 数字不一致 | 冻结统一 evaluator；报告两种口径但不混用 |
| 单卡时间不足 | 50 epoch 超过 15 小时 | 先固定子集筛选机制，再只对 B0/最佳项跑全量多 seed |
| A2 与 MoE 变化混杂 | 同时更换模型 YAML/路由配置 | 主实验固定 YOLO26n；MoE/P2 放到 P2 扩展 |
| 大目标结论不稳定 | val 仅 130 个大目标 | 报样本量和 CI，不夸大 AP_large 波动 |

最低降级交付应仍包含：可开关 STAL、单元测试、正式 B0/S1、分档指标、逐 epoch 正样本统计、一个 seed 的完整
复现包，以及“统计把握不足、尚不能声称稳定提升”的诚实结论。

## 11. 最终交付清单

- [ ] 默认关闭且向后兼容的 STAL 模块与配置；
- [ ] TAL/STAL/配置/恢复/设备兼容单元测试；
- [ ] VisDrone 统一分档 evaluator 和数据 manifest；
- [ ] B0 与最佳 STAL 的完整训练配置、命令、checkpoint 哈希；
- [ ] AP_small/medium/large、AR_small/medium/large 主表；
- [ ] 每 epoch 正样本、覆盖、冲突、质量和 feature-level 曲线；
- [ ] delta-k、面积阈值、warmup 参数敏感性；
- [ ] 3 seeds 与 paired bootstrap 95% CI；
- [ ] 定性案例、失败案例和已知限制；
- [ ] 最终 PR、复现 README、答辩材料。

## 12. 结论

A2 的工程入口已经明确，数据和 GPU 也具备开工条件；真正的难点不是写出一个动态 top-k，而是建立可信的因果链：

```text
面积感知策略生效
→ 小目标候选/最终正样本覆盖发生可控变化
→ 正样本质量和冲突仍可接受
→ AP_small/AR_small 在统一口径下提升
→ 多 seed 与置信区间证明提升不是噪声
```

只要按这个顺序推进，P0 可以快速闭环，P1 也有清晰、可证伪且适合单卡完成的研究路径。
