"""Resume four completed A2OR 20-epoch DTK runs to a common 50-epoch endpoint.

Each child owns one CUDA context so a Windows driver/context failure cannot leak
from one variant into the next. Existing 20-epoch APs artifacts are preserved;
the 50-epoch evaluation is written to a separate file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\coding\YOLO-Master")
A2OR = ROOT / "A2OR"
RUNS = A2OR / "runs"
SUBSET = ROOT / "A2" / "subsets" / "visdrone_10pct_seed42"
DATA = SUBSET / "visdrone_10pct_seed42.yaml"
DATASET_ROOT = Path(r"D:\coding\datasets\VisDrone")
EVALUATOR = ROOT / "A2" / "scripts" / "evaluate_p0_checkpoints.py"
TARGET_EPOCHS = 50
VARIANTS = (
    "baseline_fixedk10_vd10pct_s42_20e_b4_w1",
    "dtk_lambda0p5_vd10pct_s42_20e_b4_w1",
    "dtk_lambda0p55_vd10pct_s42_20e_b4_w1",
    "dtk_lambda0p6_vd10pct_s42_20e_b4_w1",
)


def parse_args() -> argparse.Namespace:
    """Choose a single child variant or the sequential parent mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, help="Resume one variant in this process only.")
    return parser.parse_args()


def continue_variant(name: str) -> None:
    """Resume one healthy epoch-20 checkpoint and score all 50 checkpoints."""
    from ultralytics import YOLO

    run_dir = RUNS / name
    healthy = run_dir / "weights" / "last_healthy.pt"
    epoch19 = run_dir / "weights" / "epoch19.pt"
    if not healthy.is_file() or not epoch19.is_file():
        raise FileNotFoundError(f"{name} lacks a resumable 20-epoch checkpoint: {run_dir}")

    print(f"\n{'=' * 88}\nResuming {name}: total epochs 20 -> {TARGET_EPOCHS}\n{'=' * 88}", flush=True)
    YOLO(str(healthy)).train(resume=str(healthy), epochs=TARGET_EPOCHS, patience=0, save_period=1)

    output = run_dir / f"checkpoint_area_metrics_{TARGET_EPOCHS}e.json"
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--weights",
            str(run_dir / "weights"),
            "--data",
            str(DATA),
            "--images",
            str(DATASET_ROOT / "images" / "val"),
            "--labels",
            str(DATASET_ROOT / "labels" / "val"),
            "--image-list",
            str(SUBSET / "val.txt"),
            "--output",
            str(output),
            "--imgsz",
            "800",
            "--batch",
            "4",
            "--device",
            "0",
            "--workers",
            "1",
            "--start-epoch",
            "1",
            "--end-epoch",
            str(TARGET_EPOCHS),
        ],
        check=True,
        cwd=ROOT,
    )
    print(f"Completed {name}; wrote {output.name}.", flush=True)


def main() -> None:
    """Run the four matched variants sequentially in isolated child interpreters."""
    os.chdir(ROOT)
    os.environ["YOLO_CONFIG_DIR"] = str(A2OR / ".ultralytics_config")
    args = parse_args()
    if args.variant:
        continue_variant(args.variant)
        return
    for name in VARIANTS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--variant", name], check=True, cwd=ROOT)
    print("All requested A2OR baseline/DTK variants reached 50 total epochs.", flush=True)


if __name__ == "__main__":
    main()
