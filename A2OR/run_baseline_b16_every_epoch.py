"""Train a fresh A2OR batch-16 baseline and save a checkpoint every epoch."""

import os
import sys
from pathlib import Path

import torch


GPU_INDEX = 0
WORKSPACE = Path(__file__).resolve().parents[1]
RUN_NAME = "baseline_b16_nbs64_120e_every_epoch"
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE / "A2OR/.ultralytics_config"))
os.environ.setdefault("TQDM_ASCII", "1")
sys.path.insert(0, str(WORKSPACE))


def main():
    """Train a new batch-16 run; do not silently fall back to another batch size."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested A2OR baseline run.")

    from ultralytics import YOLO

    model = YOLO(WORKSPACE / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")
    if "--preflight" in sys.argv:
        print("Preflight succeeded: model and CUDA are ready.", flush=True)
        return

    model.train(
        data=WORKSPACE / "A2OR/visdrone_full.yaml",
        epochs=120,
        patience=0,
        batch=16,
        nbs=64,
        workers=0,
        imgsz=800,
        device=GPU_INDEX,
        val=True,
        split="val",
        fraction=1.0,
        save_period=1,
        project=WORKSPACE / "A2OR/runs",
        name=RUN_NAME,
    )


if __name__ == "__main__":
    main()
