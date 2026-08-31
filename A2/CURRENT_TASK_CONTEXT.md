# A2 当前任务上下文（有效摘要）

## 当前目标

完成 A2「STAL 式小目标自适应标签分配」研究。

- **P0 已完成**：当前 YOLO26n + 内嵌 STAL-style TAL baseline 在 VisDrone2019-DET 完整训练、分档评测和正样本统计。
- **下一目标 P1**：在该 baseline 上改进小目标分配；主指标为 `APs` 绝对提升至少 **1.0 AP point**，并用重复实验或置信区间证明不是随机波动。

`+1.0 AP point` 等于原始 0–1 数值 `+0.01`；当前基线 APs=7.31，因此 P1 门槛为 **APs >= 8.31**。

## 关键决策

1. 当前 `tal.py` 中的 YOLO26 小目标候选区域放宽逻辑就是本题 baseline 的 STAL-style 实现；**P0/P1 不需要额外证明它优于传统 TAL**。
2. STAL 只在训练期扩大极小目标的**候选区域**，不修改 GT 标注框、不改变回归目标、不参与推理。
3. 模型选择规则固定为标准 COCO `maxDets=100` 的**总体 AP 最优 checkpoint**，禁止事后按 APs 最高 checkpoint 挑选结果。
4. 面积分档正式指标使用原图像素：small `<32²`，medium `32²–96²`，large `≥96²`；同时保留 `maxDets=300` 作为 VisDrone 密集场景补充口径。
5. 训练正样本统计按增强后的 640 输入空间分档；其面积口径与正式 AP/AR 分档不同，不能逐项等同。
6. Epoch 41 起关闭 Mosaic，因此 epoch 1–40 与 epoch 41–50 的**原始 GT/正样本总数不可直接比较**；优先比较 `positives-per-GT` 和 `zero-GT rate`。

## 核心约束

- 仓库：`D:\coding\YOLO-Master`
- 环境：Conda `yolo_master`；GPU：RTX 5060 Laptop GPU。
- 数据：`D:\coding\datasets\VisDrone`；训练配置：`A2/configs/visdrone.yaml`。
- 复现实验固定：YOLO26n、imgsz=640、batch=2、AdamW、seed=42、deterministic=True、50 epoch、`patience=0`（禁用早停，保证跑满声明轮数）。
- P1/P2 必须复用上述模型、数据划分、训练时长、评测脚本和 checkpoint 选择规则，除非实验设计明确声明唯一改动。
- 保留所有现有 P0 权重、日志和输出；不覆盖用户已有改动或既有实验。

## 当前进度

### P0：已完成

- 50/50 epoch 完成，`epoch0.pt` 至 `epoch49.pt`、`best.pt`、`last.pt` 均已保存。
- 每 epoch 记录 O2M/O2O 的 small/medium/large GT 数、正样本数、`positives-per-GT` 和 `zero-GT rate`。
- 全部 50 个 checkpoint 已完成小/中/大目标 COCO 分档评测。
- 标准 COCO `maxDets=100` 最终结果：

| AP | AP50 | APs | APm | APl | ARs | ARm | ARl |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15.19 | 27.26 | 7.31 | 23.23 | 41.98 | 18.56 | 41.38 | 60.87 |

- Epoch 50 O2M：small 3.7259 positives/GT、8.88% zero-GT；medium 9.9389、0.02%；large 9.9760、0.10%。
- Epoch 50 O2O：small 0.9221 positives/GT、7.79% zero-GT；medium 0.9999、0.01%；large 1.0000、0.00%。

### 已落地的代码与评测工具

- `ultralytics/utils/loss.py`：默认关闭的训练期正样本 telemetry。
- `ultralytics/engine/trainer.py`：每 epoch 聚合并写入 assignment metrics。
- `ultralytics/cfg/default.yaml`、`ultralytics/cfg/__init__.py`：telemetry 配置注入。
- `A2/scripts/evaluate_visdrone_area.py`：最终预测 JSON 的原图面积 COCO 评测。
- `A2/scripts/evaluate_p0_checkpoints.py`：全 checkpoint 分档评测。
- `A2/scripts/build_p0_artifacts.py`：P0曲线与报告生成。

## P0 证据入口

- 报告：`A2/runs/p0_y26n_vd640_s42_50e/P0_FINAL_REPORT.md`
- 工作簿：`A2/runs/p0_y26n_vd640_s42_50e/outputs/01a05285-db0e-70a2-b368-c05430fa5704/P0_RESULTS.xlsx`
- 最终面积指标：`A2/runs/p0_y26n_vd640_s42_50e/p0_area_metrics.json`
- 50轮面积指标：`A2/runs/p0_y26n_vd640_s42_50e/p0_checkpoint_area_metrics.json`
- 50轮正样本记录：`A2/runs/p0_y26n_vd640_s42_50e/p0_epoch_assignment_records.json`

## 未解决问题

1. P1 的具体 STAL 改进尚未确定和实现。
2. 需要将现有硬编码小目标候选放宽逻辑变为可配置、可消融的实验模块。
3. 需要选择并预注册 P1 的唯一主要改动，避免同时改变多个变量。
4. P1 首轮成功后，需要至少多 seed 或按图像配对 bootstrap 置信区间，验证 APs 增益稳定性。

## 下一步行动

1. 设计并实现可配置的面积感知 STAL 变体；优先只改候选区域阈值/放宽幅度，保留 TAL 的评分、Top-k和冲突处理。
2. 加单元测试：默认配置严格复现当前 baseline；开启新配置只改变候选资格，不改变原始 GT 回归目标。
3. 用 seed=42 跑一次 50 epoch P1 on/off 对照，按标准 COCO `maxDets=100` 总体 AP 最优 checkpoint 报告 APs/APm/APl 和正样本曲线。
4. 若 APs 达到或接近 8.31，再跑额外 seeds / bootstrap，形成 P1 结论。
