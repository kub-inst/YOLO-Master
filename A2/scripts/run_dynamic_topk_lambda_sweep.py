"""Run a first-ten-epoch Dynamic TopK lambda sweep against a matched fixed-K control."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO


ROOT = Path(r"D:\coding\YOLO-Master")
RUNS_ROOT = ROOT / "A2" / "runs" / "p1_dynamic_topk_lambda_sweep_b2_w0"
MODEL = ROOT / "yolo26n.pt"
DATA = ROOT / "A2" / "configs" / "visdrone.yaml"
IMAGES = Path(r"D:\coding\datasets\VisDrone\images\val")
LABELS = Path(r"D:\coding\datasets\VisDrone\labels\val")
EVALUATOR = ROOT / "A2" / "scripts" / "evaluate_p0_checkpoints.py"
P0_BASELINE_METRICS = ROOT / "A2" / "runs" / "p0_y26n_vd640_s42_50e" / "p0_checkpoint_area_metrics.json"
LAMBDAS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
TARGET_MEAN_DELTA_APS = 1.0


def variant_name(lambda_value: float) -> str:
    """Return a stable output-directory name for one lambda candidate."""
    return f"lambda_{lambda_value:.2f}".replace(".", "p")


def train_variant(lambda_value: float) -> Path:
    """Train one ten-epoch candidate with all non-TopK settings held constant."""
    name = variant_name(lambda_value)
    print(f"\n{'=' * 88}\nTraining {name}\n{'=' * 88}", flush=True)
    model = YOLO(str(MODEL))
    model.train(
        data=str(DATA),
        epochs=10,
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        close_mosaic=10,
        patience=0,
        assignment_stats=True,
        assignment_small_area=1024.0,
        assignment_medium_area=9216.0,
        tal_dynamic_topk_small=True,
        tal_dynamic_topk_lambda=lambda_value,
        save=True,
        save_period=1,
        save_json=True,
        val=True,
        plots=True,
        project=str(RUNS_ROOT),
        name=name,
        exist_ok=False,
    )
    return RUNS_ROOT / name


def evaluate_variant(run_dir: Path) -> Path:
    """Evaluate all first-ten-epoch checkpoints with the project-standard APs protocol."""
    output = run_dir / "checkpoint_area_metrics.json"
    command = [
        sys.executable,
        str(EVALUATOR),
        "--weights",
        str(run_dir / "weights"),
        "--data",
        str(DATA),
        "--images",
        str(IMAGES),
        "--labels",
        str(LABELS),
        "--output",
        str(output),
        "--imgsz",
        "640",
        "--batch",
        "2",
        "--device",
        "0",
        "--workers",
        "0",
        "--start-epoch",
        "1",
        "--end-epoch",
        "10",
    ]
    print(f"\nEvaluating {run_dir.name} APs with workers=0 for Windows-safe offline evaluation.", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    return output


def aps_series(metrics_path: Path) -> list[float]:
    """Load standard COCO APs as percentage points for epochs one through ten."""
    records = json.loads(metrics_path.read_text(encoding="utf-8"))["records"]
    records = [record for record in records if record["epoch"] <= 10]
    if [record["epoch"] for record in records] != list(range(1, 11)):
        raise ValueError(f"Expected APs metrics for epochs 1-10: {metrics_path}")
    return [100.0 * record["coco_max_dets_100"]["AP_small"] for record in records]


def write_summary(entries: list[dict]) -> None:
    """Persist partial or final sweep summaries after each candidate."""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    (RUNS_ROOT / "sweep_summary.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RUNS_ROOT / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("variant", "lambda", "mean_aps", "mean_delta_aps", "status"))
        writer.writeheader()
        writer.writerows({key: entry[key] for key in writer.fieldnames} for entry in entries)


def plot_all(entries: list[dict]) -> None:
    """Visualize every completed group when no lambda reaches the stopping threshold."""
    figure, axis = plt.subplots(figsize=(11, 6))
    epochs = range(1, 11)
    for entry in entries:
        label = "P0 baseline (fixed K=10)" if entry["lambda"] is None else f"Dynamic TopK λ={entry['lambda']:.2f}"
        axis.plot(epochs, entry["aps"], marker="o", linewidth=2, label=label)
    axis.set_title("Dynamic TopK lambda sweep: APs over the first 10 epochs")
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("APs (points, COCO maxDets=100)")
    axis.set_xticks(list(epochs))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(RUNS_ROOT / "lambda_sweep_aps.png", dpi=180, bbox_inches="tight")


def main() -> None:
    """Run lambda candidates against the existing matched P0 baseline and stop when the target is met."""
    entries: list[dict] = []
    baseline_aps = aps_series(P0_BASELINE_METRICS)
    entries.append(
        {
            "variant": "P0 baseline (fixed K=10)",
            "lambda": None,
            "mean_aps": sum(baseline_aps) / len(baseline_aps),
            "mean_delta_aps": 0.0,
            "status": "control",
            "aps": baseline_aps,
        }
    )
    write_summary(entries)

    for lambda_value in LAMBDAS:
        metrics = evaluate_variant(train_variant(lambda_value))
        candidate_aps = aps_series(metrics)
        mean_delta = sum(candidate - baseline for candidate, baseline in zip(candidate_aps, baseline_aps)) / len(baseline_aps)
        entry = {
            "variant": variant_name(lambda_value),
            "lambda": lambda_value,
            "mean_aps": sum(candidate_aps) / len(candidate_aps),
            "mean_delta_aps": mean_delta,
            "status": "threshold_met" if mean_delta > TARGET_MEAN_DELTA_APS else "completed",
            "aps": candidate_aps,
        }
        entries.append(entry)
        write_summary(entries)
        print(f"{entry['variant']}: mean ΔAPs={mean_delta:+.3f} points", flush=True)
        if mean_delta > TARGET_MEAN_DELTA_APS:
            print(f"Stopping: {entry['variant']} exceeded +{TARGET_MEAN_DELTA_APS:.2f} mean ΔAPs.", flush=True)
            return

    plot_all(entries)
    print("No lambda exceeded the threshold; saved lambda_sweep_aps.png.", flush=True)


if __name__ == "__main__":
    main()
