#!/usr/bin/env python3
"""Build the checked-in Issue #52 report dataset and figures.

The source tables are the compact VisDrone results imported from upstream PR
#85.  This script keeps the report assets reproducible instead of checking in
figures that cannot be regenerated from the accompanying CSV files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLES = {
    "pruning-threshold-results.csv": ROOT / "scripts/issue52_pruning_results.csv",
    "alternative-pruning-results.csv": ROOT / "scripts/issue52_alternative_pruning_results.csv",
    "dynamic-schedule-results.csv": ROOT / "scripts/issue52_dynamic_schedule_results.csv",
    "expert-usage-gini.csv": ROOT / "scripts/issue52_expert_usage_gini.csv",
    "per-layer-experts.csv": ROOT / "scripts/issue52_per_layer_experts.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def copy_tables(output: Path) -> None:
    for name, source in SOURCE_TABLES.items():
        shutil.copyfile(source, output / name)


def write_pareto(output: Path, pruning: list[dict[str, str]], alternatives: list[dict[str, str]]) -> list[dict[str, str]]:
    points = [
        {
            "label": f"t={row['threshold']} {row['stage']}",
            "source": "threshold_sweep",
            "structurally_pruned": "false",
            "mAP50-95": row["mAP50-95"],
            "latency_mean_ms": row["latency_mean_ms"],
        }
        for row in pruning
    ]
    points.extend(
        {
            "label": f"{row['variant']} {row['stage']}",
            "source": "fixed_budget_probe",
            "structurally_pruned": str(row["retained_experts"] == "2/2").lower(),
            "mAP50-95": row["mAP50-95"],
            "latency_mean_ms": row["latency_mean_ms"],
        }
        for row in alternatives
        if row["variant"] == "weighted_top2"
    )
    ranked = sorted(points, key=lambda row: (number(row, "latency_mean_ms"), -number(row, "mAP50-95")))
    best_map = -1.0
    for row in ranked:
        is_pareto = number(row, "mAP50-95") > best_map
        row["pareto"] = str(is_pareto).lower()
        if is_pareto:
            best_map = number(row, "mAP50-95")
    fields = ("label", "source", "structurally_pruned", "mAP50-95", "latency_mean_ms", "pareto")
    with (output / "pareto-accuracy-latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ranked)
    return ranked


def build_plots(data_output: Path, figure_output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_output.mkdir(parents=True, exist_ok=True)
    pruning = read_csv(data_output / "pruning-threshold-results.csv")
    alternatives = read_csv(data_output / "alternative-pruning-results.csv")
    dynamic = read_csv(data_output / "dynamic-schedule-results.csv")
    layers = read_csv(data_output / "per-layer-experts.csv")
    pareto = write_pareto(data_output, pruning, alternatives)

    colors = {"direct": "#2878B5", "LoRA10": "#F28E2B"}
    metric_specs = (
        ("mAP50-95", "mAP50-95", "threshold-map.png"),
        ("moe_gflops", "MoE GFLOPs", "threshold-gflops.png"),
        ("latency_mean_ms", "Latency (ms)", "threshold-latency.png"),
    )
    for key, ylabel, filename in metric_specs:
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        for stage in ("direct", "LoRA10"):
            group = sorted((row for row in pruning if row["stage"] == stage), key=lambda row: number(row, "threshold"))
            axis.plot(
                [number(row, "threshold") for row in group],
                [number(row, key) for row in group],
                marker="o",
                linewidth=2,
                label=stage,
                color=colors[stage],
            )
        axis.set(xlabel="Pruning threshold", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_output / filename, dpi=180)
        plt.close(fig)

    fig = plt.figure(figsize=(8.2, 6.2))
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        [number(row, "threshold") for row in pruning],
        [number(row, "moe_gflops") for row in pruning],
        [number(row, "latency_mean_ms") for row in pruning],
        c=[number(row, "mAP50-95") for row in pruning],
        cmap="viridis",
        s=55,
    )
    axis.set(xlabel="Threshold", ylabel="MoE GFLOPs", zlabel="Latency (ms)")
    fig.colorbar(scatter, ax=axis, label="mAP50-95", shrink=0.72)
    fig.tight_layout()
    fig.savefig(figure_output / "threshold-map-flops-latency-3d.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    for row in pareto:
        x, y = number(row, "latency_mean_ms"), number(row, "mAP50-95")
        marker = "s" if row["structurally_pruned"] == "true" else "o"
        axis.scatter(x, y, marker=marker, s=72, alpha=0.8)
        axis.annotate(row["label"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7)
    front = [row for row in pareto if row["pareto"] == "true"]
    axis.plot(
        [number(row, "latency_mean_ms") for row in front],
        [number(row, "mAP50-95") for row in front],
        color="#D62728",
        linewidth=1.8,
        label="Pareto front",
    )
    axis.set(xlabel="Latency (ms)", ylabel="mAP50-95", title="No quality-qualified structural sweet spot observed")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_output / "pareto-accuracy-latency.png", dpi=180)
    plt.close(fig)

    labels = [row["variant"].replace("_", "\n") for row in dynamic]
    x = np.arange(len(dynamic))
    width = 0.34
    fig, left = plt.subplots(figsize=(7.4, 4.8))
    left.bar(x - width / 2, [number(row, "final_mAP50-95") for row in dynamic], width, label="Final mAP50-95")
    left.bar(x + width / 2, [number(row, "best_mAP50-95") for row in dynamic], width, label="Best mAP50-95")
    left.set_ylabel("mAP50-95")
    left.set_xticks(x, labels)
    left.set_ylim(0.17, 0.18)
    left.grid(axis="y", alpha=0.2)
    right = left.twinx()
    right.plot(x, [number(row, "convergence_ratio") for row in dynamic], color="#D62728", marker="D", label="Convergence ratio")
    right.set_ylabel("Epoch ratio to 95% target")
    right.set_ylim(0.85, 1.02)
    handles_l, labels_l = left.get_legend_handles_labels()
    handles_r, labels_r = right.get_legend_handles_labels()
    left.legend(handles_l + handles_r, labels_l + labels_r, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_output / "dynamic-schedule-comparison.png", dpi=180)
    plt.close(fig)

    thresholds = sorted({number(row, "threshold") for row in layers})
    layer_names = sorted({row["layer_name"] for row in layers}, key=lambda value: int(value.split(".")[-1]))
    matrix = np.array(
        [
            [
                number(next(row for row in layers if number(row, "threshold") == threshold and row["layer_name"] == layer), "retained_experts")
                for layer in layer_names
            ]
            for threshold in thresholds
        ]
    )
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=3, aspect="auto")
    axis.set_xticks(range(len(layer_names)), layer_names)
    axis.set_yticks(range(len(thresholds)), [f"{value:.2f}" for value in thresholds])
    axis.set(xlabel="MoE layer", ylabel="Threshold", title="Retained experts (all default points remain 3/3)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{int(matrix[i, j])}/3", ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, label="Retained experts")
    fig.tight_layout()
    fig.savefig(figure_output / "per-layer-retained-experts.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-output", type=Path, default=ROOT / "reports/moe-pruning")
    parser.add_argument("--figure-output", type=Path, default=ROOT / "reports/issue52-figs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_output = args.data_output.resolve()
    figure_output = args.figure_output.resolve()
    data_output.mkdir(parents=True, exist_ok=True)
    copy_tables(data_output)
    build_plots(data_output, figure_output)
    print(f"Issue #52 CSV files written to {data_output}")
    print(f"Issue #52 figures written to {figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
