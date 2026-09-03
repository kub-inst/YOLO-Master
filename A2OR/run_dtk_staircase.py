"""Run the matched A2OR fixed-TopK baseline and Dynamic-TopK lambda staircase.

Each variant is trained independently under the actual batch-4, 6-GiB-cap protocol
used by the completed A2OR DTK lambda=0.8 pilot.  Every epoch checkpoint is then
scored with the registered VisDrone subset using COCO AP/APs at IoU=.50:.95.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(r"D:\coding\YOLO-Master")
A2OR = ROOT / "A2OR"
DATASET_ROOT = Path(r"D:\coding\datasets\VisDrone")
SUBSET = ROOT / "A2" / "subsets" / "visdrone_10pct_seed42"
DATA = SUBSET / "visdrone_10pct_seed42.yaml"
BASE_CONFIG = A2OR / "dtk_v01n_visdrone10_s42_20e_w1.yaml"
EVALUATOR = ROOT / "A2" / "scripts" / "evaluate_p0_checkpoints.py"
RUNS = A2OR / "runs"
GPU_LIMIT_GIB = 6.0
EXPECTED = {
    "train.txt": (648, "2e75d7df777b9cd8b43c1b7dfe5e951b0c9628a9d24acda9fa99d57c5d60af10"),
    "val.txt": (55, "fa94c59f6b5b202f6ba53b684403ec1b29a3cc27becf57da373c2f5e49099d8d"),
}
VARIANTS = (("baseline_fixedk10", False, None),) + tuple(
    (f"dtk_lambda{str(value).replace('.', 'p')}", True, value) for value in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
)


class Tee:
    """Mirror stdout and stderr to a per-variant audit log."""

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


def preflight() -> None:
    """Refuse accidental output reuse and validate immutable subset inputs."""
    for variant, _, _ in VARIANTS:
        run_dir = RUNS / f"{variant}_vd10pct_s42_20e_b4_w1"
        if run_dir.exists():
            resumable = (
                variant == "baseline_fixedk10"
                and (run_dir / "weights" / "last.pt").is_file()
                and len((run_dir / "results.csv").read_text(encoding="utf-8").splitlines()) < 21
            )
            if not resumable:
                raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    for filename, (expected_count, expected_hash) in EXPECTED.items():
        path = SUBSET / filename
        actual_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_hash = sha256(path)
        if len(actual_lines) != expected_count or actual_hash.lower() != expected_hash:
            raise RuntimeError(f"Subset verification failed for {path}: {len(actual_lines)} lines, {actual_hash}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is unavailable; refusing CPU execution.")


def build_config(variant: str, dynamic: bool, lambda_value: float | None) -> dict:
    """Build the exact matched config for one independent training run."""
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    run_name = f"{variant}_vd10pct_s42_20e_b4_w1"
    config.update(
        batch=4,
        data=str(DATA),
        name=run_name,
        project=str(RUNS),
        tal_dynamic_topk_small=dynamic,
        tal_dynamic_topk_lambda=0.0 if lambda_value is None else lambda_value,
    )
    return config


def run_variant(variant: str, dynamic: bool, lambda_value: float | None) -> dict:
    """Train and score one variant, returning its durable manifest record."""
    config = build_config(variant, dynamic, lambda_value)
    run_dir = RUNS / config["name"]
    resume_checkpoint = run_dir / "weights" / "last.pt"
    resume = variant == "baseline_fixedk10" and resume_checkpoint.is_file()
    config_path = A2OR / f"{config['name']}.yaml"
    log_path = A2OR / f"{config['name']}_console.log"
    yaml.safe_dump(config, config_path.open("w", encoding="utf-8"), allow_unicode=True, sort_keys=False)

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, log)), contextlib.redirect_stderr(Tee(sys.__stderr__, log)):
            print(f"\n=== START {config['name']} ===", flush=True)
            print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
            from ultralytics import YOLO

            model_path = str(resume_checkpoint) if resume else config.pop("model")
            config.pop("task", None)
            config.pop("mode", None)
            if resume:
                config["resume"] = True
                print(f"Resuming baseline from {resume_checkpoint}", flush=True)
            YOLO(model_path).train(**config)
            metrics_path = run_dir / "checkpoint_area_metrics.json"
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
                    str(metrics_path),
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
                    "20",
                ],
                check=True,
                cwd=ROOT,
            )
            records = json.loads(metrics_path.read_text(encoding="utf-8"))["records"]
            best = max(records, key=lambda record: record["coco_max_dets_100"]["AP_small"])
            print(
                f"=== DONE {config['name']}; best APs={best['coco_max_dets_100']['AP_small']:.4f} "
                f"at epoch {best['epoch']} ===",
                flush=True,
            )
    return {
        "variant": variant,
        "dynamic_topk_small": dynamic,
        "lambda": lambda_value,
        "run_dir": str(run_dir),
        "config": str(config_path),
        "log": str(log_path),
        "area_metrics": str(run_dir / "checkpoint_area_metrics.json"),
    }


def main() -> None:
    """Launch the whole matched staircase sequentially and preserve progress."""
    os.chdir(ROOT)
    os.environ["YOLO_CONFIG_DIR"] = str(A2OR / ".ultralytics_config")
    preflight()
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_LIMIT_GIB * 1024**3 / properties.total_memory), 0)
    manifest_path = A2OR / "dtk_staircase_manifest.json"
    manifest = {"protocol": "A2OR matched batch-4 DTK staircase", "variants": [], "failures": []}
    for variant, dynamic, lambda_value in VARIANTS:
        try:
            manifest["variants"].append(run_variant(variant, dynamic, lambda_value))
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            torch.cuda.empty_cache()
        except Exception as error:
            manifest["failures"].append({"variant": variant, "error": repr(error)})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
