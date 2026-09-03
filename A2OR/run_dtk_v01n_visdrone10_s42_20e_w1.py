"""Launch the confirmed A2OR v0.1-N Dynamic TopK experiment with a 6 GiB CUDA allocator cap."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(r"D:\coding\YOLO-Master")
A2OR = ROOT / "A2OR"
CONFIG = A2OR / "dtk_v01n_visdrone10_s42_20e_w1.yaml"
RUN_DIR = A2OR / "runs" / "dtk_lambda0p80_vd10pct_s42_20e_w1"
LOG = A2OR / "dtk_lambda0p80_vd10pct_s42_20e_w1_console.log"
SUBSET = ROOT / "A2" / "subsets" / "visdrone_10pct_seed42"
EXPECTED = {
    "train.txt": (648, "2e75d7df777b9cd8b43c1b7dfe5e951b0c9628a9d24acda9fa99d57c5d60af10"),
    "val.txt": (55, "fa94c59f6b5b202f6ba53b684403ec1b29a3cc27becf57da373c2f5e49099d8d"),
}
GPU_LIMIT_GIB = 6.0


class Tee:
    """Write console output to both the visible terminal and an audit log."""

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


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs() -> dict:
    """Validate immutable inputs and refuse accidental output reuse."""
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to start because the output directory already exists: {RUN_DIR}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for filename, (expected_count, expected_hash) in EXPECTED.items():
        path = SUBSET / filename
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_hash = sha256(path)
        if len(lines) != expected_count or actual_hash.lower() != expected_hash:
            raise RuntimeError(
                f"Subset verification failed for {path}: count={len(lines)}, sha256={actual_hash}"
            )
    return config


def main() -> None:
    """Apply the memory cap, print the effective protocol, and start training."""
    os.chdir(ROOT)
    config_dir = A2OR / ".ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

        config = validate_inputs()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 is unavailable; refusing to fall back to CPU.")
        properties = torch.cuda.get_device_properties(0)
        limit_bytes = int(GPU_LIMIT_GIB * 1024**3)
        fraction = min(1.0, limit_bytes / properties.total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)

        print("A2OR DTK launch preflight passed.", flush=True)
        print(f"model={config['model']}", flush=True)
        print(f"data={config['data']} (648 train / 55 val; registered seed-42 subset)", flush=True)
        print(
            f"epochs={config['epochs']} imgsz={config['imgsz']} batch={config['batch']} "
            f"workers={config['workers']} prefetch_factor=4",
            flush=True,
        )
        print(
            f"DTK small_only={config['tal_dynamic_topk_small']} lambda={config['tal_dynamic_topk_lambda']} "
            f"training_seed={config['seed']}",
            flush=True,
        )
        print(
            f"GPU={properties.name} total={properties.total_memory / 1024**3:.2f} GiB "
            f"allocator_cap={GPU_LIMIT_GIB:.2f} GiB (fraction={fraction:.6f})",
            flush=True,
        )
        print(f"run_dir={RUN_DIR}", flush=True)
        print(f"console_log={LOG}", flush=True)

        from ultralytics import YOLO

        model_path = config.pop("model")
        config.pop("task", None)
        config.pop("mode", None)
        YOLO(model_path).train(**config)


if __name__ == "__main__":
    main()
