# YOLO26n VisDrone 默认 TAL/STAL 基线提前终止报告

## 运行信息

- 实验 ID：`b0_y26n_vd640_s42`
- 模型：`yolo26n.pt`
- 数据集：VisDrone2019-DET
- 输入尺寸：640
- batch：2
- seed：42
- 优化器：AdamW
- 计划训练：50 epochs
- 实际完整完成：4 epochs
- 终止方式：按用户要求手动终止
- 标签分配：仓库默认 `TaskAlignedAssigner`，包含 YOLO26 内置的小目标候选框扩张逻辑（默认 TAL/STAL）

## 已完成 epoch 指标

| epoch | box loss | cls loss | dfl loss | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.5993 | 5.1021 | 0.00534 | 0.05677 | 0.01271 | 0.00043 | 0.00018 |
| 2 | 2.5013 | 2.6720 | 0.00388 | 0.22864 | 0.16059 | 0.09463 | 0.04743 |
| 3 | 2.4074 | 2.2146 | 0.00361 | 0.22478 | 0.19809 | 0.12857 | 0.06671 |
| 4 | 2.3967 | 2.1130 | 0.00354 | 0.24665 | 0.21322 | 0.15070 | 0.07809 |

第 4 epoch 是现有记录中的最佳结果：

- Precision：24.665%
- Recall：21.322%
- mAP50：15.070%
- mAP50-95：7.809%

从 epoch 1 到 epoch 4，分类损失由 5.1021 降至 2.1130，Recall 由 1.271% 上升至 21.322%，
mAP50-95 由 0.018% 上升至 7.809%。曲线仍在明显改善，尚未收敛，因此不能把第 4 epoch 数字作为正式最终性能。

## 已保存产物

- `args.yaml`
- `results.csv`
- `weights/best.pt`
- `weights/last.pt`
- `weights/last_healthy.pt`
- `weights/epoch0.pt`
- 标签及训练 batch 可视化

## 当前结论与限制

本次运行证明了 YOLO26n、VisDrone 数据和仓库默认 TAL/STAL 训练链路能够持续运行并产生有效的上升指标，
比原有单 epoch smoke 提供了更有意义的连通性证据。

但它还不能完成 P0，原因如下：

1. 只完成 4 epochs，训练尚未收敛；
2. 当前验证结果只有总体 P/R/mAP，没有 AP_small、AP_medium、AP_large；
3. 当前代码没有逐 epoch 保存正样本分配统计；
4. 因此不能据此判断小目标正样本变化是否改善 AP_small/Recall_small；
5. 这些结果只能称为“提前终止的默认基线初步结果”，不能称为正式 P0 或 P1 结论。
