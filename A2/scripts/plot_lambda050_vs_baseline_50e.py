"""Plot APs and APs deltas for the completed lambda=0.50 and fixed-K=10 subset runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"D:\coding\YOLO-Master")
BASELINE = ROOT / "A2" / "runs" / "p1_fixedk10_baseline_vd10pct_s42_b2_w0" / "checkpoint_area_metrics_50e.json"
DYNAMIC = (
    ROOT
    / "A2"
    / "runs"
    / "p1_dynamic_topk_lambda_precheck_vd10pct_s42_b2_w0"
    / "lambda_0p50"
    / "checkpoint_area_metrics_50e.json"
)
OUTPUT = ROOT / "A2" / "runs" / "p1_fixedk10_baseline_vd10pct_s42_b2_w0" / "lambda_0p50_vs_baseline_50e_aps.png"


def load_aps(path: Path) -> list[float]:
    """Load APs as percentage points from exactly fifty checkpoint records."""
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    if [record["epoch"] for record in records] != list(range(1, 51)):
        raise ValueError(f"Expected epoch 1-50 records in {path}")
    return [100.0 * record["coco_max_dets_100"]["AP_small"] for record in records]


def main() -> None:
    """Write a two-panel 50-epoch comparison image."""
    baseline, dynamic = load_aps(BASELINE), load_aps(DYNAMIC)
    epochs = list(range(1, 51))
    deltas = [candidate - reference for candidate, reference in zip(dynamic, baseline)]
    figure, (aps_axis, delta_axis) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=(3, 2))

    aps_axis.plot(epochs, baseline, color="black", linewidth=2.5, label="Fixed K=10 baseline")
    aps_axis.plot(epochs, dynamic, color="#1f77b4", linewidth=2.2, label="Dynamic TopK λ=0.50")
    aps_axis.set(title="VisDrone 10% subset: APs over 50 epochs", ylabel="APs (points)")
    aps_axis.grid(alpha=0.3)
    aps_axis.legend()

    delta_axis.plot(epochs, deltas, color="#1f77b4", linewidth=1.8, label="λ=0.50 − baseline")
    delta_axis.axhline(0, color="black", linewidth=1.2, linestyle="--", label="Parity")
    delta_axis.axhline(1, color="#2ca02c", linewidth=1.2, linestyle="--", label="+1.0 point target")
    delta_axis.fill_between(epochs, deltas, 0, where=[value >= 0 for value in deltas], color="#2ca02c", alpha=0.16)
    delta_axis.fill_between(epochs, deltas, 0, where=[value < 0 for value in deltas], color="#d62728", alpha=0.16)
    delta_axis.set(xlabel="Training epoch", ylabel="Δ APs (points)")
    delta_axis.set_xticks(list(range(0, 51, 5)))
    delta_axis.grid(alpha=0.3)
    delta_axis.legend(ncol=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
