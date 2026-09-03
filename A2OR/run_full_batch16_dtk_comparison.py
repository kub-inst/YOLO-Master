"""Run the requested full-VisDrone batch-4 baseline/DTK comparison sequentially."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\coding\YOLO-Master")
A2OR = ROOT / "A2OR"
RUNS = A2OR / "runs"
DATA = A2OR / "visdrone_full.yaml"
DATASET_ROOT = Path(r"D:\coding\datasets\VisDrone")
MODEL = ROOT / "ultralytics" / "cfg" / "models" / "master" / "v0_1" / "det" / "yolo-master-n.yaml"
EVALUATOR = ROOT / "A2" / "scripts" / "evaluate_p0_checkpoints.py"
VARIANTS = (
    ("full_baseline_vd100pct_s0_50e_b4_w4", 50, False, 0.0),
    ("full_dtk_lambda0p5_vd100pct_s0_20e_b4_w4", 20, True, 0.50),
    ("full_dtk_lambda0p55_vd100pct_s0_20e_b4_w4", 20, True, 0.55),
    ("full_dtk_lambda0p6_vd100pct_s0_20e_b4_w4", 20, True, 0.60),
)


class Tee:
    """Mirror training output to the visible console and a durable per-run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    """Select the sequential parent mode or one isolated child variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(item[0] for item in VARIANTS), help="Run one variant only.")
    parser.add_argument("--lambda-only", action="store_true", help="Run the three DTK lambda variants, skipping baseline.")
    return parser.parse_args()


def get_variant(name: str) -> tuple[str, int, bool, float]:
    """Return one exact named variant definition."""
    return next(item for item in VARIANTS if item[0] == name)


def train_variant(name: str) -> None:
    """Train, save every checkpoint, then score every checkpoint with full-val APs."""
    from ultralytics import YOLO

    _, epochs, dynamic, lambda_value = get_variant(name)
    run_dir = RUNS / name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {run_dir}")

    log_path = A2OR / f"{name}_console.log"
    config = {
        "data": str(DATA),
        "epochs": epochs,
        "patience": 0,
        "batch": 4,
        "workers": 4,
        "imgsz": 800,
        "device": 0,
        "seed": 0,
        "deterministic": True,
        "optimizer": "auto",
        "cache": False,
        "val": True,
        "split": "val",
        "fraction": 1.0,
        "nbs": 64,
        "close_mosaic": 10,
        "tal_topk": 10,
        "tal_alpha": 0.5,
        "tal_beta": 6.0,
        "tal_dynamic_topk_small": dynamic,
        "tal_dynamic_topk_lambda": lambda_value,
        "assignment_stats": True,
        "assignment_small_area": 1024.0,
        "assignment_medium_area": 9216.0,
        "save": True,
        "save_period": 1,
        "plots": True,
        "project": str(RUNS),
        "name": name,
        "exist_ok": False,
        "pretrained": True,
    }
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, log)), contextlib.redirect_stderr(Tee(sys.__stderr__, log)):
            print(f"\n=== START {name} ===", flush=True)
            print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
            YOLO(str(MODEL)).train(**config)
            output = run_dir / "checkpoint_area_metrics.json"
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
                    "--output",
                    str(output),
                    "--imgsz",
                    "800",
                    "--batch",
                    "4",
                    "--device",
                    "0",
                    "--workers",
                    "4",
                    "--start-epoch",
                    "1",
                    "--end-epoch",
                    str(epochs),
                ],
                check=True,
                cwd=ROOT,
            )
            print(f"=== DONE {name}; wrote {output.name} ===", flush=True)


def main() -> None:
    """Run each variant in a separate process so CUDA state does not carry over."""
    os.chdir(ROOT)
    os.environ["YOLO_CONFIG_DIR"] = str(A2OR / ".ultralytics_config")
    args = parse_args()
    if args.variant:
        train_variant(args.variant)
        return
    variants = (item for item in VARIANTS if item[2]) if args.lambda_only else VARIANTS
    for name, _, _, _ in variants:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--variant", name], check=True, cwd=ROOT)
    print("All requested full-VisDrone batch-4 baseline/DTK runs completed.", flush=True)


if __name__ == "__main__":
    main()
