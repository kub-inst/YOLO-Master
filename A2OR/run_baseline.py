"""Run a fresh 120-epoch A2OR batch-4 baseline without a GPU memory cap."""

import os
import sys
from pathlib import Path

import torch


GPU_INDEX = 0
WORKSPACE = Path(__file__).resolve().parents[1]
RUN_NAME = "baseline_b4_nbs64_120e"
RESUME_CHECKPOINT = WORKSPACE / "A2OR/runs" / RUN_NAME / "weights/last_healthy.pt"
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE / "A2OR/.ultralytics_config"))
# The visible Windows PowerShell console may not render tqdm's Unicode blocks correctly.
os.environ.setdefault("TQDM_ASCII", "1")
sys.path.insert(0, str(WORKSPACE))


def main():
    """Train the A2OR constrained-memory baseline from scratch."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested A2OR baseline run.")

    from ultralytics import YOLO

    if "--preflight" in sys.argv:
        YOLO(WORKSPACE / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")
        print("Preflight succeeded: model and CUDA are ready.", flush=True)
        return

    if "--resume" in sys.argv:
        if not RESUME_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CHECKPOINT}")
        model = YOLO(RESUME_CHECKPOINT)
        model.train(resume=str(RESUME_CHECKPOINT), epochs=120, device=GPU_INDEX, workers=0)
        return

    model = YOLO(WORKSPACE / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")
    model.train(
        data=WORKSPACE / "A2OR/visdrone_full.yaml",
        epochs=120,
        patience=0,
        batch=4,
        nbs=64,
        workers=0,
        imgsz=800,
        device=GPU_INDEX,
        val=True,
        split="val",
        fraction=1.0,
        project=WORKSPACE / "A2OR/runs",
        name=RUN_NAME,
    )


if __name__ == "__main__":
    main()
