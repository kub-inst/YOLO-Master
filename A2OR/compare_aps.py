"""Compare two VisDrone checkpoints with COCO small-object AP using no A2 imports."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
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
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline checkpoint, e.g. weights/epoch29.pt.")
    parser.add_argument("--exp", type=Path, required=True, help="exp checkpoint from the same training epoch.")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--images", type=Path, required=True, help="Validation images directory.")
    parser.add_argument("--labels", type=Path, required=True, help="Validation YOLO labels directory.")
    parser.add_argument("--output", type=Path, default=Path("A2OR/aps_comparison.json"))
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size; does not change the metric.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def yolo_ground_truth(images_dir: Path, labels_dir: Path) -> tuple[dict, dict[str, int]]:
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
        max_det=300,
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
    for path in (args.baseline, args.exp, args.data, args.images, args.labels):
        if not path.exists():
            raise FileNotFoundError(path)
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
