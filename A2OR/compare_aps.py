"""Compare two VisDrone checkpoints with COCO small-object AP using no A2 imports."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SETTINGS_PATH = Path(__file__).resolve().parent / ".runtime_data" / "compare_aps_settings.json"
VISDRONE_NAMES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compare", "set"), default="compare")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline checkpoint, e.g. weights/epoch29.pt.")
    parser.add_argument("--exp", type=Path, default=None, help="Experiment checkpoint from the same training epoch.")
    parser.add_argument("--data", type=Path, default=None, help="Ultralytics dataset YAML.")
    parser.add_argument("--images", type=Path, default=None, help="Validation images directory.")
    parser.add_argument("--labels", type=Path, default=None, help="Validation YOLO labels directory.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None, help="Evaluation batch size; does not change the metric.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None, help="Maximum detections per image during validation.")
    parser.add_argument("--print-config", action="store_true", help="Print effective parameters without evaluation.")
    args = parser.parse_args()
    saved = load_settings()
    for field in ("baseline", "exp", "data", "images", "labels", "output", "imgsz", "batch", "device", "workers", "max_det"):
        if getattr(args, field) is None and field in saved:
            setattr(args, field, Path(saved[field]) if field in {"baseline", "exp", "data", "images", "labels", "output"} else saved[field])
    if args.output is None:
        args.output = Path("A2OR/aps_comparison.json")
    if args.imgsz is None:
        args.imgsz = 800
    if args.batch is None:
        args.batch = 16
    if args.device is None:
        args.device = "0"
    if args.workers is None:
        args.workers = 8
    if args.max_det is None:
        args.max_det = 300
    return args


def load_settings() -> dict:
    """Load persisted comparison parameters, if available."""
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    params = value.get("params", {}) if isinstance(value, dict) else {}
    return params if isinstance(params, dict) else {}


def save_settings(args: argparse.Namespace) -> None:
    """Persist the complete comparison configuration."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "baseline": str(args.baseline), "exp": str(args.exp), "data": str(args.data),
        "images": str(args.images), "labels": str(args.labels), "output": str(args.output),
        "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
        "workers": args.workers, "max_det": args.max_det,
    }
    SETTINGS_PATH.write_text(json.dumps({"params": params}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    """Validate required paths and numeric evaluation settings."""
    required = ("baseline", "exp", "data", "images", "labels")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing comparison parameters: {', '.join('--' + name for name in missing)}")
    if args.imgsz < 1 or args.batch < 1 or args.workers < 0 or args.max_det < 1:
        raise SystemExit("imgsz, batch, and max-det must be positive; workers must be non-negative")
    for name in required:
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(path)


def yolo_ground_truth(images_dir: Path, labels_dir: Path) -> tuple[dict, dict[str, int]]:
    from PIL import Image

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No validation images found in {images_dir}")

    images, annotations = [], []
    filename_to_id: dict[str, int] = {}
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        filename_to_id[image_path.name] = image_id
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Expected 5 fields in {label_path}:{line_number}, got {len(values)}")
            class_id, x_center, y_center, box_width, box_height = map(float, values)
            class_index = int(class_id)
            if class_id != class_index or not 0 <= class_index < len(VISDRONE_NAMES):
                raise ValueError(f"Invalid class ID {class_id} in {label_path}:{line_number}")

            x1 = max(0.0, (x_center - box_width / 2) * width)
            y1 = max(0.0, (y_center - box_height / 2) * height)
            x2 = min(float(width), (x_center + box_width / 2) * width)
            y2 = min(float(height), (y_center + box_height / 2) * height)
            pixel_width, pixel_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if pixel_width == 0 or pixel_height == 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_index + 1,
                    "bbox": [x1, y1, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    dataset = {
        "info": {"description": "VisDrone validation labels converted from YOLO format"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(VISDRONE_NAMES)],
    }
    return dataset, filename_to_id


def remap_predictions(predictions_path: Path, filename_to_id: dict[str, int]) -> list[dict]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    remapped = []
    for prediction in predictions:
        filename = prediction.get("file_name") or f"{prediction['image_id']}.jpg"
        image_id = filename_to_id.get(Path(filename).name)
        if image_id is None:
            continue
        remapped.append(
            {
                "image_id": image_id,
                "category_id": int(prediction["category_id"]),
                "bbox": [float(value) for value in prediction["bbox"]],
                "score": float(prediction["score"]),
            }
        )
    if not remapped:
        raise ValueError(f"No predictions from {predictions_path} matched validation filenames")
    return remapped


def coco_metrics(coco_gt: COCO, predictions: list[dict], max_det: int) -> dict[str, float]:
    from faster_coco_eval import COCOeval_faster

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox", print_function=lambda *_: None)
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {key: float(value) for key, value in evaluator.stats_as_dict.items()}


def validate_checkpoint(
    name: str,
    checkpoint: Path,
    args: argparse.Namespace,
    temporary_root: Path,
    coco_gt: COCO,
    filename_to_id: dict[str, int],
) -> dict:
    from ultralytics import YOLO

    print(f"Evaluating {name}: {checkpoint}", flush=True)
    validation = YOLO(str(checkpoint)).val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        save_json=True,
        project=str(temporary_root),
        name=name,
        exist_ok=True,
        verbose=False,
        max_det=args.max_det,
    )
    predictions_path = Path(validation.save_dir) / "predictions.json"
    predictions = remap_predictions(predictions_path, filename_to_id)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "ultralytics": {key: float(value) for key, value in validation.results_dict.items()},
        "coco_max_dets_100": coco_metrics(coco_gt, predictions, 100),
        "dense_max_dets_300": coco_metrics(coco_gt, predictions, 300),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.mode == "set":
        save_settings(args)
        print(f"Saved comparison settings to {SETTINGS_PATH.resolve()}")
        saved = load_settings()
        print(f"saved_parameter_count={len(saved)}")
        print(json.dumps(saved, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.print_config:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, ensure_ascii=False, indent=2, sort_keys=True))
        return

    from faster_coco_eval import COCO

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset, filename_to_id = yolo_ground_truth(args.images.resolve(), args.labels.resolve())
    coco_gt = COCO(dataset, print_function=lambda *_: None)
    with tempfile.TemporaryDirectory(prefix="a2or_aps_", dir=args.output.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        baseline = validate_checkpoint("baseline", args.baseline.resolve(), args, temporary_root, coco_gt, filename_to_id)
        exp = validate_checkpoint("exp", args.exp.resolve(), args, temporary_root, coco_gt, filename_to_id)

    primary_baseline = baseline["coco_max_dets_100"]
    primary_exp = exp["coco_max_dets_100"]
    delta = {
        key: (primary_exp[key] - primary_baseline[key]) * 100
        for key in ("AP_all", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")
    }
    report = {
        "protocol": {
            "area_ranges_original_pixels": {"small": [0, 1024], "medium": [1024, 9216], "large": [9216, 1e10]},
            "primary": "COCO maxDets=100",
            "supplemental": "VisDrone-dense maxDets=300",
            "validation_images": len(dataset["images"]),
            "ground_truth_boxes": len(dataset["annotations"]),
            "imgsz": args.imgsz,
        },
        "baseline": baseline,
        "exp": exp,
        "exp_minus_baseline_points": delta,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCOCO maxDets=100 (percentage points)")
    print("metric       baseline       exp       delta")
    for label, key in (("AP", "AP_all"), ("AP50", "AP_50"), ("APs", "AP_small"), ("APm", "AP_medium"), ("APl", "AP_large")):
        baseline_value = primary_baseline[key] * 100
        exp_value = primary_exp[key] * 100
        print(f"{label:<8} {baseline_value:10.3f} {exp_value:10.3f} {exp_value - baseline_value:+10.3f}")
    print(f"\nSaved auditable results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
