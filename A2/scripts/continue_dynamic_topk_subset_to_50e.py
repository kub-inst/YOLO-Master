"""Continue the matched 10% VisDrone baseline and Dynamic TopK runs from epoch 10 to epoch 50.

The parent process invokes one child Python interpreter per variant. This releases the CUDA context between variants,
which is intentional after the earlier Windows CUDA-context reset during the lambda sweep.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\coding\YOLO-Master")
DATASET_ROOT = Path(r"D:\coding\datasets\VisDrone")
SUBSET_ROOT = ROOT / "A2" / "subsets" / "visdrone_10pct_seed42"
DATA = SUBSET_ROOT / "visdrone_10pct_seed42.yaml"
EVALUATOR = ROOT / "A2" / "scripts" / "evaluate_p0_checkpoints.py"
TARGET_EPOCHS = 50
VARIANTS = {
    "baseline": ROOT / "A2" / "runs" / "p1_fixedk10_baseline_vd10pct_s42_b2_w0",
    **{
        f"lambda_{value:.2f}".replace(".", "p"): ROOT
        / "A2"
        / "runs"
        / "p1_dynamic_topk_lambda_precheck_vd10pct_s42_b2_w0"
        / f"lambda_{value:.2f}".replace(".", "p")
        for value in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    },
}


def parse_args() -> argparse.Namespace:
    """Select the all-variant parent mode or one isolated child variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), help="Run only one variant in this process.")
    return parser.parse_args()


def continue_variant(name: str) -> None:
    """Resume one saved epoch-10 checkpoint and evaluate all 50 checkpoints on the fixed subset."""
    from ultralytics import YOLO

    run_dir = VARIANTS[name]
    # Ultralytics strips optimizer state from last.pt after a normally completed run. The healthy checkpoint retains
    # optimizer, scheduler, scaler, and epoch state required for a true continuation.
    last = run_dir / "weights" / "last_healthy.pt"
    epoch9 = run_dir / "weights" / "epoch9.pt"
    if not last.is_file() or not epoch9.is_file():
        raise FileNotFoundError(f"{name} must have a resumable epoch-10 checkpoint before continuation: {run_dir}")

    print(f"\n{'=' * 88}\nContinuing {name}: epochs 11-{TARGET_EPOCHS}\n{'=' * 88}", flush=True)
    YOLO(str(last)).train(resume=str(last), epochs=TARGET_EPOCHS, patience=0, save_period=1)

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
            str(TARGET_EPOCHS),
        ],
        check=True,
        cwd=ROOT,
    )
    print(f"Completed {name}; wrote {output.name}.", flush=True)


def main() -> None:
    """Run isolated continuation children sequentially, allowing the GPU context to be freed every time."""
    args = parse_args()
    if args.variant:
        continue_variant(args.variant)
        return
    for name in VARIANTS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--variant", name], check=True, cwd=ROOT)
    print("All baseline and Dynamic TopK variants reached 50 total epochs.", flush=True)


if __name__ == "__main__":
    main()
