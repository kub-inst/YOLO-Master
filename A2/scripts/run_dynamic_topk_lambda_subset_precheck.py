"""Run a reproducible 10% VisDrone Dynamic TopK lambda precheck without a baseline gate."""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO


ROOT = Path(r"D:\coding\YOLO-Master")
DATASET_ROOT = Path(r"D:\coding\datasets\VisDrone")
RUNS_ROOT = ROOT / "A2" / "runs" / "p1_dynamic_topk_lambda_precheck_vd10pct_s42_b2_w0"
SUBSET_ROOT = ROOT / "A2" / "subsets" / "visdrone_10pct_seed42"
MODEL = ROOT / "yolo26n.pt"
EVALUATOR = ROOT / "A2" / "scripts" / "evaluate_p0_checkpoints.py"
LAMBDAS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
SEED = 42
FRACTION = 0.10
NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def variant_name(lambda_value: float) -> str:
    """Return a stable run-directory name for one lambda value."""
    return f"lambda_{lambda_value:.2f}".replace(".", "p")


def sample_images(split: str) -> list[Path]:
    """Select exactly 10% of one split without replacement using the registered seed."""
    source = DATASET_ROOT / "images" / split
    images = sorted(path.resolve() for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images in {source}")
    count = math.ceil(len(images) * FRACTION)
    return sorted(random.Random(f"{SEED}:{split}").sample(images, count))


def write_subset() -> Path:
    """Create deterministic image lists and the matching Ultralytics dataset YAML."""
    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)
    train_images, val_images = sample_images("train"), sample_images("val")
    train_list, val_list = SUBSET_ROOT / "train.txt", SUBSET_ROOT / "val.txt"
    train_list.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(map(str, val_images)) + "\n", encoding="utf-8")
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(NAMES))
    data = SUBSET_ROOT / "visdrone_10pct_seed42.yaml"
    data.write_text(f"path: {DATASET_ROOT.as_posix()}\ntrain: {train_list.as_posix()}\nval: {val_list.as_posix()}\nnames:\n{names}\n", encoding="utf-8")
    manifest = {
        "seed": SEED,
        "fraction": FRACTION,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_list": str(train_list),
        "val_list": str(val_list),
    }
    (SUBSET_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared fixed VisDrone subset: {len(train_images)} train, {len(val_images)} val images.", flush=True)
    return data


def train_variant(data: Path, lambda_value: float) -> Path:
    """Train one lambda candidate; only lambda varies across the precheck."""
    name = variant_name(lambda_value)
    print(f"\n{'=' * 88}\nTraining {name}\n{'=' * 88}", flush=True)
    YOLO(str(MODEL)).train(
        data=str(data),
        epochs=10,
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        seed=SEED,
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


def evaluate_variant(data: Path, run_dir: Path) -> Path:
    """Evaluate every checkpoint only against the registered validation subset."""
    output = run_dir / "checkpoint_area_metrics.json"
    command = [
        sys.executable,
        str(EVALUATOR),
        "--weights",
        str(run_dir / "weights"),
        "--data",
        str(data),
        "--images",
        str(DATASET_ROOT / "images" / "val"),
        "--labels",
        str(DATASET_ROOT / "labels" / "val"),
        "--image-list",
        str(SUBSET_ROOT / "val.txt"),
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
    subprocess.run(command, check=True, cwd=ROOT)
    return output


def aps_series(metrics_path: Path) -> list[float]:
    """Read APs as percentage points for the ten saved epochs."""
    records = json.loads(metrics_path.read_text(encoding="utf-8"))["records"]
    if [record["epoch"] for record in records] != list(range(1, 11)):
        raise ValueError(f"Expected epochs 1-10 in {metrics_path}")
    return [100.0 * record["coco_max_dets_100"]["AP_small"] for record in records]


def write_summary(entries: list[dict]) -> None:
    """Persist completed precheck results after each candidate."""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    (RUNS_ROOT / "sweep_summary.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RUNS_ROOT / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("variant", "lambda", "mean_aps", "best_aps", "status"))
        writer.writeheader()
        writer.writerows({key: entry[key] for key in writer.fieldnames} for entry in entries)


def plot_all(entries: list[dict]) -> None:
    """Plot APs trajectories for the lambda-only precheck."""
    figure, axis = plt.subplots(figsize=(11, 6))
    for entry in entries:
        axis.plot(range(1, 11), entry["aps"], marker="o", linewidth=2, label=f"Dynamic TopK λ={entry['lambda']:.2f}")
    axis.set(title="VisDrone 10% precheck: Dynamic TopK lambda APs", xlabel="Training epoch", ylabel="APs (points)")
    axis.set_xticks(range(1, 11))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(RUNS_ROOT / "lambda_precheck_aps.png", dpi=180, bbox_inches="tight")


def main() -> None:
    """Run all six lambda candidates; this precheck intentionally has no baseline or early-stop gate."""
    if RUNS_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing run root: {RUNS_ROOT}")
    data = write_subset()
    entries = []
    for lambda_value in LAMBDAS:
        metrics = evaluate_variant(data, train_variant(data, lambda_value))
        aps = aps_series(metrics)
        entries.append(
            {"variant": variant_name(lambda_value), "lambda": lambda_value, "mean_aps": sum(aps) / len(aps), "best_aps": max(aps), "status": "completed", "aps": aps}
        )
        write_summary(entries)
        print(f"{entries[-1]['variant']}: mean APs={entries[-1]['mean_aps']:.3f} points", flush=True)
    plot_all(entries)
    print("Precheck complete; saved lambda_precheck_aps.png.", flush=True)


if __name__ == "__main__":
    main()
