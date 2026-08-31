"""Plot Dynamic TopK and P0 baseline APs for aligned first-ten-epoch checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse metric inputs and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="P0 checkpoint area metrics JSON.")
    parser.add_argument("--dynamic-topk", type=Path, required=True, help="Dynamic TopK checkpoint area metrics JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    return parser.parse_args()


def extract_aps(path: Path) -> tuple[list[int], list[float]]:
    """Read standard COCO maxDets=100 APs, expressed in percentage points."""
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    records = [record for record in records if record["epoch"] <= 10]
    epochs = [record["epoch"] for record in records]
    aps = [100.0 * record["coco_max_dets_100"]["AP_small"] for record in records]
    if epochs != list(range(1, 11)):
        raise ValueError(f"Expected epochs 1-10 in {path}, got {epochs}")
    return epochs, aps


def main() -> None:
    """Render an aligned APs curve and Dynamic TopK minus baseline delta curve."""
    args = parse_args()
    epochs, baseline = extract_aps(args.baseline)
    dynamic_epochs, dynamic = extract_aps(args.dynamic_topk)
    if dynamic_epochs != epochs:
        raise ValueError("Baseline and Dynamic TopK epochs do not align")
    delta = [candidate - reference for candidate, reference in zip(dynamic, baseline)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, (axis_aps, axis_delta) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=(2.2, 1))
    figure.suptitle("VisDrone first-10-epoch APs: Dynamic TopK vs P0 baseline", fontsize=14, fontweight="bold")

    axis_aps.plot(epochs, baseline, marker="o", linewidth=2, label="P0 baseline (fixed K=10)", color="#4c78a8")
    axis_aps.plot(epochs, dynamic, marker="o", linewidth=2, label="Dynamic TopK (small GT, α=0.8)", color="#f58518")
    axis_aps.set_ylabel("APs (points, COCO maxDets=100)")
    axis_aps.grid(axis="y", alpha=0.3)
    axis_aps.legend(loc="upper left")
    axis_aps.annotate(
        f"Epoch 10: {dynamic[-1]:.2f} vs {baseline[-1]:.2f}",
        xy=(10, dynamic[-1]),
        xytext=(-175, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )

    axis_delta.axhline(0, color="#555555", linewidth=1)
    axis_delta.plot(epochs, delta, marker="o", linewidth=2, color="#54a24b")
    axis_delta.fill_between(epochs, 0, delta, color="#54a24b", alpha=0.2)
    axis_delta.set_xlabel("Training epoch")
    axis_delta.set_ylabel("Δ APs\n(points)")
    axis_delta.set_xticks(epochs)
    axis_delta.grid(axis="y", alpha=0.3)
    axis_delta.annotate(f"{delta[-1]:+.2f}", xy=(10, delta[-1]), xytext=(-35, 12), textcoords="offset points")

    figure.tight_layout()
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
