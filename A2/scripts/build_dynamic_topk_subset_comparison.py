"""Build matched-subset APs tables and plots for fixed K=10 versus Dynamic TopK lambdas."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"D:\coding\YOLO-Master")
BASELINE_DIR = ROOT / "A2" / "runs" / "p1_fixedk10_baseline_vd10pct_s42_b2_w0"
SWEEP_SUMMARY = ROOT / "A2" / "runs" / "p1_dynamic_topk_lambda_precheck_vd10pct_s42_b2_w0" / "sweep_summary.json"
THRESHOLD_POINTS = 1.0


def load_inputs() -> tuple[dict, list[dict]]:
    """Load the completed matched baseline and lambda precheck summaries."""
    baseline = json.loads((BASELINE_DIR / "baseline_summary.json").read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP_SUMMARY.read_text(encoding="utf-8"))
    if len(baseline["aps"]) != 10 or any(len(entry["aps"]) != 10 for entry in sweep):
        raise ValueError("Expected exactly ten APs values for baseline and every lambda")
    return baseline, sweep


def write_epoch_csv(baseline: dict, sweep: list[dict], output: Path) -> None:
    """Write APs and point deltas for every epoch and lambda."""
    fields = ["epoch", "baseline_aps"]
    for entry in sweep:
        label = f"lambda_{entry['lambda']:.2f}".replace(".", "p")
        fields.extend((f"{label}_aps", f"{label}_delta_aps"))
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch, baseline_aps in enumerate(baseline["aps"], start=1):
            row = {"epoch": epoch, "baseline_aps": baseline_aps}
            for entry in sweep:
                label = f"lambda_{entry['lambda']:.2f}".replace(".", "p")
                candidate = entry["aps"][epoch - 1]
                row[f"{label}_aps"] = candidate
                row[f"{label}_delta_aps"] = candidate - baseline_aps
            writer.writerow(row)


def summarize(baseline: dict, sweep: list[dict]) -> list[dict]:
    """Summarize mean, final-epoch, and strict all-epoch threshold behavior."""
    summaries = []
    for entry in sweep:
        deltas = [candidate - reference for candidate, reference in zip(entry["aps"], baseline["aps"])]
        summaries.append(
            {
                "lambda": entry["lambda"],
                "mean_aps": entry["mean_aps"],
                "mean_delta_aps": entry["mean_aps"] - baseline["mean_aps"],
                "epoch10_aps": entry["aps"][-1],
                "epoch10_delta_aps": deltas[-1],
                "minimum_epoch_delta_aps": min(deltas),
                "epochs_ahead": sum(delta > 0 for delta in deltas),
                "all_epochs_at_least_plus_1": all(delta >= THRESHOLD_POINTS for delta in deltas),
                "deltas": deltas,
            }
        )
    return summaries


def plot(baseline: dict, sweep: list[dict], output: Path) -> None:
    """Plot all matched-subset APs trajectories."""
    figure, axis = plt.subplots(figsize=(11, 6))
    epochs = range(1, 11)
    axis.plot(epochs, baseline["aps"], marker="o", linewidth=3, color="black", label="Fixed K=10 baseline")
    for entry in sweep:
        axis.plot(epochs, entry["aps"], marker="o", linewidth=1.8, label=f"Dynamic TopK λ={entry['lambda']:.2f}")
    axis.set(title="VisDrone 10%: Fixed K=10 vs Dynamic TopK", xlabel="Training epoch", ylabel="APs (points)")
    axis.set_xticks(list(epochs))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")


def plot_relative_difference(baseline: dict, sweep: list[dict], output: Path) -> None:
    """Plot each lambda's relative APs difference from the fixed-K=10 baseline."""
    figure, axis = plt.subplots(figsize=(11, 6))
    epochs = range(1, 11)
    for entry in sweep:
        differences = [
            100.0 * (candidate - reference) / reference
            for candidate, reference in zip(entry["aps"], baseline["aps"])
        ]
        axis.plot(epochs, differences, marker="o", linewidth=2, label=f"Dynamic TopK λ={entry['lambda']:.2f}")
    axis.axhline(0.0, color="black", linewidth=1.2, linestyle="--", label="Fixed K=10 baseline")
    axis.set(
        title="VisDrone 10%: APs difference relative to fixed K=10 baseline",
        xlabel="Training epoch",
        ylabel="Relative APs difference (%)",
    )
    axis.set_xticks(list(epochs))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")


def write_report(baseline: dict, summaries: list[dict], output: Path) -> None:
    """Write a concise auditable interpretation beside the numeric artifacts."""
    qualified = [entry for entry in summaries if entry["all_epochs_at_least_plus_1"]]
    lines = [
        "# Matched 10% VisDrone: Fixed K=10 vs Dynamic TopK",
        "",
        "All variants use seed=42, 648 training images, 55 validation images, 10 epochs, batch=2, imgsz=640, and workers=0.",
        "",
        f"Fixed K=10 baseline mean APs: **{baseline['mean_aps']:.3f}** points; epoch-10 APs: **{baseline['aps'][-1]:.3f}** points.",
        "",
        "| Lambda | Mean APs | Mean delta | Epoch-10 delta | Best per-epoch delta | Epochs ahead | All epochs >= +1.0? |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for entry in summaries:
        lines.append(
            f"| {entry['lambda']:.2f} | {entry['mean_aps']:.3f} | {entry['mean_delta_aps']:+.3f} | "
            f"{entry['epoch10_delta_aps']:+.3f} | {max(entry['deltas']):+.3f} | {entry['epochs_ahead']}/10 | "
            f"{'yes' if entry['all_epochs_at_least_plus_1'] else 'no'} |"
        )
    lines.extend(
        (
            "",
            "## Strict +1.0-point stability check",
            "",
            "Qualified lambdas: " + (", ".join(f"λ={entry['lambda']:.2f}" for entry in qualified) if qualified else "none") + ".",
            "",
            "The criterion requires delta APs >= +1.0 point at every epoch 1-10, not only at the best checkpoint.",
        )
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate all comparison artifacts."""
    baseline, sweep = load_inputs()
    summaries = summarize(baseline, sweep)
    write_epoch_csv(baseline, sweep, BASELINE_DIR / "dynamic_topk_epoch_aps_comparison.csv")
    (BASELINE_DIR / "dynamic_topk_comparison_summary.json").write_text(
        json.dumps({"baseline": baseline, "variants": summaries, "strict_threshold_points": THRESHOLD_POINTS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot(baseline, sweep, BASELINE_DIR / "dynamic_topk_vs_baseline_aps.png")
    plot_relative_difference(baseline, sweep, BASELINE_DIR / "dynamic_topk_vs_baseline_relative_pct.png")
    write_report(baseline, summaries, BASELINE_DIR / "DYNAMIC_TOPK_COMPARISON.md")


if __name__ == "__main__":
    main()
