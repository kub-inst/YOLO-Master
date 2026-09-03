"""Plot matched A2OR baseline and DTK lambda=0.50 APs across 50 epochs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"D:\coding\YOLO-Master")
RUNS = ROOT / "A2OR" / "runs"
BASELINE = RUNS / "baseline_fixedk10_vd10pct_s42_20e_b4_w1" / "checkpoint_area_metrics_50e.json"
DTK = RUNS / "dtk_lambda0p5_vd10pct_s42_20e_b4_w1" / "checkpoint_area_metrics_50e.json"
OUTPUT = RUNS / "baseline_fixedk10_vd10pct_s42_20e_b4_w1" / "lambda0p5_vs_baseline_50e_aps.png"


def load_aps(path: Path) -> tuple[list[int], list[float]]:
    """Load the primary COCO maxDets=100 APs series, in percentage points."""
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    return (
        [record["epoch"] for record in records],
        [100.0 * record["coco_max_dets_100"]["AP_small"] for record in records],
    )


def main() -> None:
    """Render APs trajectories and their matched per-epoch difference."""
    epochs, baseline = load_aps(BASELINE)
    dtk_epochs, dtk = load_aps(DTK)
    if epochs != dtk_epochs:
        raise ValueError("Baseline and lambda=0.50 checkpoint epochs do not align")
    delta = [candidate - control for candidate, control in zip(dtk, baseline)]

    figure, (aps_axis, delta_axis) = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": (2.2, 1)}
    )
    figure.suptitle("A2OR VisDrone 10%: DTK lambda=0.50 vs Fixed K=10", fontsize=14, fontweight="bold")

    aps_axis.plot(epochs, baseline, color="#222222", linewidth=2.2, label="Fixed K=10 baseline")
    aps_axis.plot(epochs, dtk, color="#1565c0", linewidth=2.2, label="DTK lambda=0.50")
    aps_axis.scatter([epochs[-1]], [baseline[-1]], color="#222222", zorder=3)
    aps_axis.scatter([epochs[-1]], [dtk[-1]], color="#1565c0", zorder=3)
    aps_axis.annotate(f"E50: {baseline[-1]:.3f}", (epochs[-1], baseline[-1]), xytext=(-92, -17), textcoords="offset points")
    aps_axis.annotate(f"E50: {dtk[-1]:.3f}", (epochs[-1], dtk[-1]), xytext=(-92, 10), textcoords="offset points", color="#1565c0")
    aps_axis.set_ylabel("APs (points)")
    aps_axis.grid(alpha=0.25)
    aps_axis.legend(loc="lower right", frameon=True)

    colors = ["#1565c0" if value >= 0 else "#c62828" for value in delta]
    delta_axis.axhline(0.0, color="#333333", linewidth=1.1)
    delta_axis.bar(epochs, delta, color=colors, width=0.82, alpha=0.9, label="DTK - baseline")
    delta_axis.plot(epochs, delta, color="#1565c0", linewidth=1.15, alpha=0.75)
    delta_axis.annotate(
        f"E50: {delta[-1]:+.3f}",
        (epochs[-1], delta[-1]),
        xytext=(-93, 10 if delta[-1] >= 0 else -20),
        textcoords="offset points",
        color="#1565c0" if delta[-1] >= 0 else "#c62828",
    )
    delta_axis.set(xlabel="Training epoch", ylabel="Delta APs\n(points)", xlim=(1, 50))
    delta_axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
