"""Build P0 epoch records, curves, and the final Markdown report from completed experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SIZES = ("small", "medium", "large")
COLORS = {"small": "#d62728", "medium": "#1f77b4", "large": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Ultralytics results.csv.")
    parser.add_argument("--area-final", type=Path, required=True, help="Final area metrics JSON.")
    parser.add_argument("--area-epochs", type=Path, required=True, help="Per-checkpoint area metrics JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_training_rows(path: Path) -> list[dict]:
    """Read numeric epoch rows while preserving the original column names."""
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [{key: int(value) if key == "epoch" else float(value) for key, value in row.items()} for row in rows]


def plot_assignment(rows: list[dict], output: Path) -> None:
    """Plot one-to-many and one-to-one assignment evolution by area bin."""
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panels = (
        (axes[0, 0], "o2m", "pos_per_gt", "O2M positives per GT", False),
        (axes[0, 1], "o2m", "zero_gt_rate", "O2M zero-GT rate", True),
        (axes[1, 0], "o2o", "pos_per_gt", "O2O positives per GT", False),
        (axes[1, 1], "o2o", "zero_gt_rate", "O2O zero-GT rate", True),
    )
    for axis, branch, metric, title, percentage in panels:
        for size in SIZES:
            values = [row[f"assign/{branch}/{metric}_{size}"] for row in rows]
            if percentage:
                values = [value * 100 for value in values]
            axis.plot(epochs, values, label=size, color=COLORS[size], linewidth=1.8)
        axis.axvline(40.5, color="#666666", linestyle="--", linewidth=1, label="Mosaic off" if branch == "o2m" else None)
        axis.set_title(title)
        axis.set_ylabel("%" if percentage else "count")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes[1]:
        axis.set_xlabel("Epoch")
    figure.suptitle("P0 TAL/STAL assignment evolution (training-input area bins)", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_area(records: list[dict], output: Path) -> None:
    """Plot per-checkpoint area AP/AR and overall COCO metrics."""
    epochs = [record["epoch"] for record in records]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for size in SIZES:
        ap = [record["coco_max_dets_100"][f"AP_{size}"] * 100 for record in records]
        ar = [record["coco_max_dets_100"][f"AR_{size}"] * 100 for record in records]
        axes[0, 0].plot(epochs, ap, label=size, color=COLORS[size], linewidth=1.8)
        axes[0, 1].plot(epochs, ar, label=size, color=COLORS[size], linewidth=1.8)
    axes[0, 0].set_title("COCO AP by area (maxDets=100)")
    axes[0, 1].set_title("COCO AR by area (maxDets=100)")
    for axis in axes[0]:
        axis.set_ylabel("AP/AR points")
        axis.grid(alpha=0.25)
        axis.legend()

    axes[1, 0].plot(
        epochs,
        [record["coco_max_dets_100"]["AP_all"] * 100 for record in records],
        label="AP50-95",
        color="#6f42c1",
    )
    axes[1, 0].plot(
        epochs,
        [record["coco_max_dets_100"]["AP_50"] * 100 for record in records],
        label="AP50",
        color="#ff7f0e",
    )
    axes[1, 0].set_title("Overall COCO AP")
    axes[1, 0].set_ylabel("AP points")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(
        epochs,
        [record["coco_max_dets_100"]["AP_small"] * 100 for record in records],
        label="APs",
        color=COLORS["small"],
    )
    axes[1, 1].plot(
        epochs,
        [record["coco_max_dets_100"]["AR_small"] * 100 for record in records],
        label="ARs",
        color="#9467bd",
    )
    axes[1, 1].set_title("Small-object AP and AR")
    axes[1, 1].set_ylabel("points")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    for axis in axes[1]:
        axis.set_xlabel("Epoch")
    figure.suptitle("P0 VisDrone area-stratified checkpoint evaluation", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(output: Path, rows: list[dict], area_final: dict, area_records: list[dict]) -> None:
    """Write the final P0 report with metric definitions and evidence paths."""
    last = rows[-1]
    standard = area_final["metrics"]["100"]
    dense = area_final["metrics"]["300"]
    best_small = max(area_records, key=lambda item: item["coco_max_dets_100"]["AP_small"])
    best_all = max(area_records, key=lambda item: item["coco_max_dets_100"]["AP_all"])
    lines = [
        "# A2 P0 实验报告：YOLO26n + 当前 STAL-style baseline",
        "",
        "## 结论",
        "",
        "当前带内嵌 STAL-style 分配的 YOLO26n 即本项目 baseline；P0 不要求再证明其优于传统 TAL。",
        "50 epoch 训练、逐 epoch checkpoint、逐 epoch 正样本统计和小/中/大目标分档评测均已完成。",
        "",
        "## AP 单位",
        "",
        "代码内部以 0–1 保存 AP，报告中乘以 100 写成 AP points。`APs +1.0` 等价于内部数值 `+0.01`，",
        "例如从 7.31 AP 提升到 8.31 AP。",
        "",
        "## 面积分档结果",
        "",
        "面积使用原图像素：small `<32²`，medium `32²–96²`，large `≥96²`。",
        "",
        "| 协议 | AP | AP50 | AP75 | APs | APm | APl | ARs | ARm | ARl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| COCO maxDets=100 | {AP_all:.2f} | {AP_50:.2f} | {AP_75:.2f} | {AP_small:.2f} | {AP_medium:.2f} | "
        "{AP_large:.2f} | {AR_small:.2f} | {AR_medium:.2f} | {AR_large:.2f} |".format(
            **{key: value * 100 for key, value in standard.items()}
        ),
        "| Dense maxDets=300 | {AP_all:.2f} | {AP_50:.2f} | {AP_75:.2f} | {AP_small:.2f} | {AP_medium:.2f} | "
        "{AP_large:.2f} | {AR_small:.2f} | {AR_medium:.2f} | {AR_large:.2f} |".format(
            **{key: value * 100 for key, value in dense.items()}
        ),
        "",
        f"标准口径最佳总体 AP 出现在 epoch {best_all['epoch']}，为 "
        f"{best_all['coco_max_dets_100']['AP_all'] * 100:.2f}；最佳 APs 出现在 epoch {best_small['epoch']}，为 "
        f"{best_small['coco_max_dets_100']['AP_small'] * 100:.2f}。",
        "P1 必须按预先固定的 checkpoint 选择规则比较，不能事后只挑 APs 最高的 epoch。",
        "",
        "## Epoch 50 正样本统计",
        "",
        "| 分支/档位 | positives per GT | zero-GT rate |",
        "|---|---:|---:|",
    ]
    for branch in ("o2m", "o2o"):
        for size in SIZES:
            lines.append(
                f"| {branch}/{size} | {last[f'assign/{branch}/pos_per_gt_{size}']:.4f} | "
                f"{last[f'assign/{branch}/zero_gt_rate_{size}'] * 100:.2f}% |"
            )
    lines.extend(
        (
            "",
            "小目标分配明显弱于中、大目标。注意 epoch 41 起关闭 Mosaic，原始 GT 数量和目标尺度分布发生改变，",
            "因此原始计数不能跨 epoch 40/41 直接比较，应优先查看 positives-per-GT 和 zero-GT rate。",
            "",
            "## 总体训练结果",
            "",
            f"- Epoch 50 Precision：{last['metrics/precision(B)']:.4f}",
            f"- Epoch 50 Recall：{last['metrics/recall(B)']:.4f}",
            f"- Epoch 50 Ultralytics mAP50：{last['metrics/mAP50(B)']:.4f}",
            f"- Epoch 50 Ultralytics mAP50-95：{last['metrics/mAP50-95(B)']:.4f}",
            "",
            "## P0 验收状态",
            "",
            "- [x] VisDrone2019-DET baseline 完成 50 epoch。",
            "- [x] 保存 50 个 epoch checkpoint、best.pt 和 last.pt。",
            "- [x] 每 epoch 记录 O2M/O2O 小、中、大目标正样本和零分配统计。",
            "- [x] 输出标准 COCO 和密集场景补充口径的 APs/APm/APl、ARs/ARm/ARl。",
            "- [x] 输出逐 checkpoint 面积分档演化记录和曲线。",
            "- [x] 固定后续 P1/P2 评测口径。",
            "",
            "## 限制",
            "",
            "训练统计面积在 640 训练输入及增强后计算；正式 AP/AR 面积分档在原图像素计算，两者用途不同，不能逐项等同。",
            "本次只有 seed=42，P1 的提升仍需多 seed 或置信区间证明不是噪声。",
            "",
        )
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate all derived P0 artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_training_rows(args.results)
    area_final = json.loads(args.area_final.read_text(encoding="utf-8"))
    area_epochs = json.loads(args.area_epochs.read_text(encoding="utf-8"))
    records = area_epochs["records"]
    if len(rows) != 50 or len(records) != 50:
        raise ValueError(f"Expected 50 training and area records, got {len(rows)} and {len(records)}")

    epoch_records = {
        "source": str(args.results.resolve()),
        "area_bin_note": "Assignment bins use augmented 640 training-input pixels, not original-image COCO areas.",
        "records": rows,
    }
    (args.output_dir / "p0_epoch_assignment_records.json").write_text(
        json.dumps(epoch_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_assignment(rows, args.output_dir / "p0_assignment_evolution.png")
    plot_area(records, args.output_dir / "p0_area_metric_evolution.png")
    write_report(args.output_dir / "P0_FINAL_REPORT.md", rows, area_final, records)
    print(f"P0 artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
