"""Evaluate every P0 checkpoint with consistent VisDrone area-stratified COCO metrics."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from faster_coco_eval import COCO
from ultralytics import YOLO

from evaluate_visdrone_area import evaluate, remap_predictions, yolo_to_coco


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Directory containing epochN.pt files.")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--images", type=Path, required=True, help="Validation image directory.")
    parser.add_argument("--labels", type=Path, required=True, help="Validation label directory.")
    parser.add_argument(
        "--image-list",
        type=Path,
        help="Optional newline-delimited validation-image subset. Defaults to every image in --images.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON updated after every checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=1, help="One-based first training epoch to evaluate.")
    parser.add_argument("--end-epoch", type=int, help="One-based final training epoch to evaluate.")
    return parser.parse_args()


def checkpoint_epoch(path: Path) -> int:
    """Map epoch0.pt to the human-readable training epoch 1."""
    match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1)) + 1


def main() -> None:
    """Run validation and area evaluation checkpoint by checkpoint."""
    args = parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(args.weights.glob("epoch*.pt"), key=checkpoint_epoch)
    checkpoints = [
        path
        for path in checkpoints
        if checkpoint_epoch(path) >= args.start_epoch
        and (args.end_epoch is None or checkpoint_epoch(path) <= args.end_epoch)
    ]
    if not checkpoints:
        raise FileNotFoundError("No matching epoch checkpoints")

    dataset, filename_to_id = yolo_to_coco(args.images, args.labels, args.image_list)
    coco_gt = COCO(dataset, print_function=lambda *_: None)
    records = []
    with tempfile.TemporaryDirectory(prefix="p0_area_eval_", dir=args.output.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for index, checkpoint in enumerate(checkpoints, start=1):
            epoch = checkpoint_epoch(checkpoint)
            print(f"[{index}/{len(checkpoints)}] evaluating training epoch {epoch}: {checkpoint.name}", flush=True)
            model = YOLO(str(checkpoint))
            validation = model.val(
                data=str(args.data),
                split="val",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                plots=False,
                save_json=True,
                project=str(temporary_root),
                name=checkpoint.stem,
                exist_ok=True,
                verbose=False,
                max_det=300,
            )
            predictions_path = Path(validation.save_dir) / "predictions.json"
            predictions = remap_predictions(predictions_path, filename_to_id)
            standard = evaluate(coco_gt, predictions, 100, verbose=False)
            dense = evaluate(coco_gt, predictions, 300, verbose=False)
            records.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint.resolve()),
                    "ultralytics": {key: float(value) for key, value in validation.results_dict.items()},
                    "coco_max_dets_100": standard,
                    "dense_max_dets_300": dense,
                }
            )
            args.output.write_text(
                json.dumps(
                    {
                        "protocol": {
                            "area_ranges": {"small": [0, 32**2], "medium": [32**2, 96**2], "large": [96**2, 1e10]},
                            "coordinates": "original_image_pixels",
                            "primary": "COCO maxDets=100",
                            "supplemental": "VisDrone-dense maxDets=300",
                        },
                        "records": records,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(f"Saved {len(records)} checkpoint records to {args.output}")


if __name__ == "__main__":
    main()
