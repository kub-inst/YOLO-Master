"""Train and evaluate the fixed-K=10 baseline on the registered VisDrone 10% subset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(r"D:\coding\YOLO-Master")
DATASET_ROOT = Path(r"D:\coding\datasets\VisDrone")
SUBSET_ROOT = ROOT / "A2" / "subsets" / "visdrone_10pct_seed42"
DATA = SUBSET_ROOT / "visdrone_10pct_seed42.yaml"
RUN_DIR = ROOT / "A2" / "runs" / "p1_fixedk10_baseline_vd10pct_s42_b2_w0"
EVALUATOR = ROOT / "A2" / "scripts" / "evaluate_p0_checkpoints.py"


def train() -> None:
    """Run the fixed TopK control with every non-TopK setting matched to the lambda precheck."""
    YOLO(str(ROOT / "yolo26n.pt")).train(
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
        tal_dynamic_topk_small=False,
        save=True,
        save_period=1,
        save_json=True,
        val=True,
        plots=True,
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        exist_ok=False,
    )


def evaluate() -> Path:
    """Evaluate all ten checkpoints using the exact same 55-image validation subset."""
    output = RUN_DIR / "checkpoint_area_metrics.json"
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--weights",
            str(RUN_DIR / "weights"),
            "--data",
            str(DATA),
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
        ],
        check=True,
        cwd=ROOT,
    )
    return output


def main() -> None:
    """Refuse overwrite, then create an auditable matched baseline result."""
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing baseline run: {RUN_DIR}")
    if not DATA.is_file() or not (SUBSET_ROOT / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing registered subset under {SUBSET_ROOT}")
    train()
    metrics = json.loads(evaluate().read_text(encoding="utf-8"))["records"]
    aps = [100.0 * record["coco_max_dets_100"]["AP_small"] for record in metrics]
    (RUN_DIR / "baseline_summary.json").write_text(
        json.dumps(
            {
                "variant": "fixed_K10_baseline",
                "dynamic_topk_small": False,
                "mean_aps": sum(aps) / len(aps),
                "best_aps": max(aps),
                "aps": aps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed fixed K=10 baseline: mean APs={sum(aps) / len(aps):.3f} points.", flush=True)


if __name__ == "__main__":
    main()
