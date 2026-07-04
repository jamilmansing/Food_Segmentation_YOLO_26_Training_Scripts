from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ultralytics.data.converter import merge_multi_segment
except Exception:  # pragma: no cover - fallback only used without ultralytics
    merge_multi_segment = None


DEFAULT_SOURCE = r"C:\Users\Jamil Gwapo\Downloads\to_combine_4_actual"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert this project's COCO polygon export to Ultralytics YOLO "
            "instance-segmentation labels."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_SOURCE,
        help="Source dataset root containing images/ and annotations_*_coco.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output YOLO dataset root. Defaults to --dataset-dir and writes labels in-place. "
            "If different from --dataset-dir, images are copied into the output dataset."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to convert.",
    )
    parser.add_argument(
        "--yaml-name",
        default="food_seg.yaml",
        help="Dataset YAML filename to write in the output root.",
    )
    parser.add_argument(
        "--split-ratio",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        help=(
            "Create a fresh split from the combined COCO JSON, e.g. "
            "--split-ratio 0.7 0.15 0.15. Defaults output to dataset_dir/yolo_resplit."
        ),
    )
    parser.add_argument(
        "--combined-json",
        default="annotations_coco.json",
        help="Combined COCO JSON to use with --split-ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --split-ratio.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print a summary without writing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annotation_path(dataset_dir: Path, split: str) -> Path:
    candidates = [
        dataset_dir / f"annotations_{split}_coco.json",
        dataset_dir / "annotations" / f"instances_{split}.json",
        dataset_dir / "annotations" / f"instances_{split}_coco.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find an annotation JSON for split "
        f"{split!r}. Checked: {', '.join(str(p) for p in candidates)}"
    )


def is_background(category: dict[str, Any]) -> bool:
    return str(category.get("name", "")).strip().lower() == "background"


def category_mapping(coco: dict[str, Any]) -> tuple[dict[int, int], list[str]]:
    categories = sorted(
        (c for c in coco.get("categories", []) if not is_background(c)),
        key=lambda c: int(c["id"]),
    )
    cat_id_to_yolo = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    names = [str(cat["name"]) for cat in categories]
    return cat_id_to_yolo, names


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_flat_polygon(points: list[float], width: int, height: int) -> list[float] | None:
    if len(points) < 6 or len(points) % 2:
        return None

    normalized: list[float] = []
    for i in range(0, len(points), 2):
        normalized.append(clamp01(float(points[i]) / width))
        normalized.append(clamp01(float(points[i + 1]) / height))
    return normalized


def segmentation_to_polygon(segmentation: Any, width: int, height: int) -> list[float] | None:
    """Return one normalized YOLO polygon from a COCO segmentation field."""
    if not segmentation:
        return None

    # RLE dictionaries are valid COCO but not handled by this lightweight polygon converter.
    if isinstance(segmentation, dict):
        return None

    if not isinstance(segmentation, list):
        return None

    if len(segmentation) == 1:
        return normalize_flat_polygon(segmentation[0], width, height)

    if merge_multi_segment is not None:
        merged = merge_multi_segment(segmentation)
        flat = (np.concatenate(merged, axis=0) / np.array([width, height])).reshape(-1).tolist()
        return [clamp01(float(v)) for v in flat]

    # Fallback: keep the largest polygon by coordinate count if Ultralytics is unavailable.
    largest = max(segmentation, key=len)
    return normalize_flat_polygon(largest, width, height)


def source_image_path(dataset_dir: Path, file_name: str, split: str) -> Path:
    candidates = [
        dataset_dir / file_name,
        dataset_dir / "images" / split / Path(file_name).name,
        dataset_dir / "images" / Path(file_name).name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def write_yaml(output_dir: Path, yaml_name: str, names: list[str], include_test: bool, dry_run: bool) -> None:
    yaml_path = output_dir / yaml_name
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend(["", "names:"])
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    text = "\n".join(lines) + "\n"

    if dry_run:
        print(f"[DRY-RUN] Would write dataset YAML: {yaml_path}")
        print(text)
        return

    yaml_path.write_text(text, encoding="utf-8")
    print(f"[OK] Wrote dataset YAML: {yaml_path}")


def convert_split(
    dataset_dir: Path,
    output_dir: Path,
    split: str,
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[str], dict[str, int]]:
    ann_path = annotation_path(dataset_dir, split)
    coco = load_json(ann_path)
    cat_id_to_yolo, names = category_mapping(coco)

    images = {int(img["id"]): img for img in coco.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    label_dir = output_dir / "labels" / split
    image_out_dir = output_dir / "images" / split
    copy_images = dataset_dir.resolve() != output_dir.resolve()

    if not dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)
        image_out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": len(images),
        "annotations": len(coco.get("annotations", [])),
        "written_objects": 0,
        "empty_images": 0,
        "skipped_crowd": 0,
        "skipped_category": 0,
        "skipped_segmentation": 0,
        "missing_images": 0,
    }

    for image_id, image_info in sorted(images.items()):
        width = int(image_info["width"])
        height = int(image_info["height"])
        file_name = str(image_info["file_name"])
        image_name = Path(file_name).name
        label_path = label_dir / f"{Path(image_name).stem}.txt"

        if label_path.exists() and not overwrite and not dry_run:
            raise FileExistsError(f"Label file already exists: {label_path}. Use --overwrite.")

        source_image = source_image_path(dataset_dir, file_name, split)
        if not source_image.exists():
            stats["missing_images"] += 1

        if copy_images and source_image.exists() and not dry_run:
            target_image = image_out_dir / image_name
            if not target_image.exists() or overwrite:
                shutil.copy2(source_image, target_image)

        lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            if int(ann.get("iscrowd", 0)):
                stats["skipped_crowd"] += 1
                continue

            category_id = int(ann["category_id"])
            if category_id not in cat_id_to_yolo:
                stats["skipped_category"] += 1
                continue

            polygon = segmentation_to_polygon(ann.get("segmentation"), width, height)
            if not polygon:
                stats["skipped_segmentation"] += 1
                continue

            class_id = cat_id_to_yolo[category_id]
            coords = " ".join(f"{value:.6f}" for value in polygon)
            lines.append(f"{class_id} {coords}")
            stats["written_objects"] += 1

        if not lines:
            stats["empty_images"] += 1

        if not dry_run:
            label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    print(
        f"[{split}] images={stats['images']} annotations={stats['annotations']} "
        f"objects_written={stats['written_objects']} empty_images={stats['empty_images']} "
        f"skipped_segmentation={stats['skipped_segmentation']} missing_images={stats['missing_images']}"
    )
    return names, stats


def ratio_split_images(
    images: list[dict[str, Any]],
    ratios: list[float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("--split-ratio values must be non-negative and sum to more than 0.")

    total_ratio = sum(ratios)
    train_ratio, val_ratio, _ = [r / total_ratio for r in ratios]
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_count = int(n * train_ratio)
    val_count = int(n * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def convert_resplit(
    dataset_dir: Path,
    output_dir: Path,
    combined_json: str,
    ratios: list[float],
    seed: int,
    overwrite: bool,
    dry_run: bool,
) -> list[str]:
    ann_path = dataset_dir / combined_json
    if not ann_path.exists():
        raise FileNotFoundError(f"Combined COCO JSON does not exist: {ann_path}")

    coco = load_json(ann_path)
    cat_id_to_yolo, names = category_mapping(coco)
    split_images = ratio_split_images(coco.get("images", []), ratios, seed)

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    for split, images in split_images.items():
        label_dir = output_dir / "labels" / split
        image_out_dir = output_dir / "images" / split
        if not dry_run:
            label_dir.mkdir(parents=True, exist_ok=True)
            image_out_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "images": len(images),
            "written_objects": 0,
            "empty_images": 0,
            "skipped_crowd": 0,
            "skipped_category": 0,
            "skipped_segmentation": 0,
            "missing_images": 0,
        }

        for image_info in images:
            image_id = int(image_info["id"])
            width = int(image_info["width"])
            height = int(image_info["height"])
            file_name = str(image_info["file_name"])
            image_name = Path(file_name).name
            label_path = label_dir / f"{Path(image_name).stem}.txt"

            if label_path.exists() and not overwrite and not dry_run:
                raise FileExistsError(f"Label file already exists: {label_path}. Use --overwrite.")

            source_image = source_image_path(dataset_dir, file_name, split)
            if not source_image.exists():
                stats["missing_images"] += 1
            elif not dry_run:
                target_image = image_out_dir / image_name
                if not target_image.exists() or overwrite:
                    shutil.copy2(source_image, target_image)

            lines: list[str] = []
            for ann in anns_by_image.get(image_id, []):
                if int(ann.get("iscrowd", 0)):
                    stats["skipped_crowd"] += 1
                    continue

                category_id = int(ann["category_id"])
                if category_id not in cat_id_to_yolo:
                    stats["skipped_category"] += 1
                    continue

                polygon = segmentation_to_polygon(ann.get("segmentation"), width, height)
                if not polygon:
                    stats["skipped_segmentation"] += 1
                    continue

                class_id = cat_id_to_yolo[category_id]
                coords = " ".join(f"{value:.6f}" for value in polygon)
                lines.append(f"{class_id} {coords}")
                stats["written_objects"] += 1

            if not lines:
                stats["empty_images"] += 1

            if not dry_run:
                label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

        print(
            f"[{split}] images={stats['images']} objects_written={stats['written_objects']} "
            f"empty_images={stats['empty_images']} skipped_segmentation={stats['skipped_segmentation']} "
            f"missing_images={stats['missing_images']}"
        )

    return names


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if args.split_ratio and args.output_dir is None:
        output_dir = dataset_dir / "yolo_resplit"
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_dir

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    if args.split_ratio:
        names = convert_resplit(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            combined_json=args.combined_json,
            ratios=args.split_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        write_yaml(output_dir, args.yaml_name, names, include_test=True, dry_run=args.dry_run)
        print(f"[OK] YOLO segmentation dataset ready at: {output_dir}")
        return

    all_names: list[str] | None = None
    converted_splits: list[str] = []
    total_missing = 0

    for split in args.splits:
        names, stats = convert_split(dataset_dir, output_dir, split, args.overwrite, args.dry_run)
        converted_splits.append(split)
        total_missing += stats["missing_images"]
        if all_names is None:
            all_names = names
        elif names != all_names:
            raise SystemExit(f"Class names differ in split {split}: {names} != {all_names}")

    if all_names is None:
        raise SystemExit("No splits converted.")

    write_yaml(output_dir, args.yaml_name, all_names, "test" in converted_splits, args.dry_run)

    if total_missing:
        raise SystemExit(f"Done with warnings: {total_missing} referenced images were missing.")

    print(f"[OK] YOLO segmentation dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
