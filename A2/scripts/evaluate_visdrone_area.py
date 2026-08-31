"""Evaluate VisDrone YOLO labels and COCO-style predictions by object area.

The evaluator converts the validation labels to COCO coordinates in the original image space, remaps Ultralytics
prediction image IDs, and reports both the standard COCO maxDets=100 protocol and a dense-scene maxDets=300 variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image


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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Validation image directory.")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO validation label directory.")
    parser.add_argument("--predictions", type=Path, required=True, help="Ultralytics COCO-style predictions JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output metrics JSON path.")
    parser.add_argument("--gt-json", type=Path, help="Optional path for the generated COCO ground-truth JSON.")
    parser.add_argument("--max-dets", type=int, nargs="+", default=(100, 300), help="Maximum detections per image.")
    return parser.parse_args()


def yolo_to_coco(
    images_dir: Path, labels_dir: Path, image_list: Path | None = None
) -> tuple[dict, dict[str, int]]:
    """Convert YOLO labels to an in-memory COCO dataset using original image coordinates."""
    if image_list is None:
        image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        image_paths = [Path(line).resolve() for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(path.parent != images_dir.resolve() or path.suffix.lower() not in IMAGE_SUFFIXES for path in image_paths):
            raise ValueError(f"Every image in {image_list} must be a supported direct child of {images_dir}")
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
                raise ValueError(f"Expected 5 YOLO fields in {label_path}:{line_number}, got {len(values)}")
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
        "info": {"description": "VisDrone2019-DET validation labels converted from YOLO format"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(VISDRONE_NAMES)],
    }
    return dataset, filename_to_id


def remap_predictions(predictions_path: Path, filename_to_id: dict[str, int]) -> list[dict]:
    """Remap Ultralytics string image IDs to the integer IDs used by the generated COCO dataset."""
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    remapped = []
    for prediction in predictions:
        filename = prediction.get("file_name")
        if filename is None:
            filename = f"{prediction['image_id']}.jpg"
        image_id = filename_to_id.get(Path(filename).name)
        if image_id is None:
            continue
        item = {
            "image_id": image_id,
            "category_id": int(prediction["category_id"]),
            "bbox": [float(value) for value in prediction["bbox"]],
            "score": float(prediction["score"]),
        }
        remapped.append(item)
    return remapped


def evaluate(coco_gt: COCO, predictions: list[dict], max_det: int, verbose: bool = True) -> dict[str, float]:
    """Run COCO bounding-box evaluation for one maximum-detection setting."""
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval_faster(
        coco_gt, coco_dt, iouType="bbox", print_function=print if verbose else lambda *_: None
    )
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {key: float(value) for key, value in evaluator.stats_as_dict.items()}


def write_markdown(output_path: Path, report: dict) -> None:
    """Write a compact, auditable Markdown companion report."""
    lines = [
        "# VisDrone P0 面积分档评测",
        "",
        "面积按原图像素计算：small `<32²`，medium `32²–96²`，large `≥96²`。",
        "",
        f"- 验证图像：{report['dataset']['images']}",
        f"- GT 数量：{report['dataset']['annotations']}",
        f"- 预测数量：{report['dataset']['predictions']}",
        "",
        "| maxDets | AP | AP50 | AP75 | APs | APm | APl | ARs | ARm | ARl |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for max_det, metrics in report["metrics"].items():
        lines.append(
            "| {max_det} | {AP_all:.4f} | {AP_50:.4f} | {AP_75:.4f} | {AP_small:.4f} | "
            "{AP_medium:.4f} | {AP_large:.4f} | {AR_small:.4f} | {AR_medium:.4f} | {AR_large:.4f} |".format(
                max_det=max_det, **metrics
            )
        )
    lines.extend(
        (
            "",
            "`maxDets=100` 是标准 COCO 口径；`maxDets=300` 是面向 VisDrone 密集场景的补充口径。",
            "所有后续 P1/P2 实验必须复用相同口径。",
            "",
        )
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build ground truth, evaluate predictions, and persist the protocol and metrics."""
    args = parse_args()
    for path in (args.images, args.labels, args.predictions):
        if not path.exists():
            raise FileNotFoundError(path)
    if any(value < 10 for value in args.max_dets):
        raise ValueError("Every --max-dets value must be at least 10")

    dataset, filename_to_id = yolo_to_coco(args.images, args.labels)
    predictions = remap_predictions(args.predictions, filename_to_id)
    if not predictions:
        raise ValueError("No predictions matched validation image filenames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gt_json = args.gt_json or args.output.with_name("visdrone_val_coco_gt.json")
    gt_json.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    coco_gt = COCO(dataset, print_function=print)
    metrics = {str(max_det): evaluate(coco_gt, predictions, max_det) for max_det in args.max_dets}
    report = {
        "protocol": {
            "coordinates": "original_image_pixels",
            "area_ranges": {"small": [0, 32**2], "medium": [32**2, 96**2], "large": [96**2, 1e10]},
            "iou_thresholds": [round(0.5 + 0.05 * index, 2) for index in range(10)],
            "max_dets": args.max_dets,
            "primary": "COCO maxDets=100",
            "supplemental": "VisDrone-dense maxDets=300",
        },
        "dataset": {
            "images": len(dataset["images"]),
            "annotations": len(dataset["annotations"]),
            "predictions": len(predictions),
        },
        "artifacts": {
            "predictions": str(args.predictions.resolve()),
            "ground_truth": str(gt_json.resolve()),
        },
        "metrics": metrics,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
