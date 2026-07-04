from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def cache_arg(value: str):
    value = value.lower()
    if value in {"none", "false", "off", "0"}:
        return False
    if value in {"true", "ram"}:
        return "ram"
    if value == "disk":
        return "disk"
    raise argparse.ArgumentTypeError("cache must be one of: none, ram, disk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an Ultralytics YOLO instance-segmentation model."
    )
    parser.add_argument("--data", required=True, help="Path to YOLO segmentation dataset YAML.")
    parser.add_argument("--model", default="yolo26n-seg.pt", help="Model checkpoint or YAML.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="-1", help="Batch size, or -1 for Ultralytics auto batch.")
    parser.add_argument(
        "--val-batch",
        default=None,
        help=(
            "Validation batch size. Defaults to --batch when positive, otherwise 16. "
            "Use this because Ultralytics AutoBatch (-1) is train-only."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="val",
        help="Dataset split to use for eval-only and final validation.",
    )
    parser.add_argument("--device", default="0", help="GPU id, e.g. 0, or comma list such as 0,1.")
    parser.add_argument("--workers", type=int, default=4, help="CPU dataloader workers per rank/GPU.")
    parser.add_argument("--cache", type=cache_arg, default="disk", help="none, ram, or disk.")
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile",
        default="False",
        help='PyTorch compile setting: False, True, default, reduce-overhead, or max-autotune-no-cudagraphs.',
    )
    parser.add_argument("--mask-ratio", type=int, default=4, help="Segmentation mask downsample ratio.")
    parser.add_argument("--overlap-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--copy-paste", type=float, default=None)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="food_yolo26_seg")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--light-augment",
        action="store_true",
        help="Disable costly online augmentations to reduce CPU-side load.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run validation on --model.",
    )
    return parser.parse_args()


def parse_batch(value: str):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--batch must be an int, float, or -1") from None


def parse_compile(value: str):
    lowered = value.lower()
    if lowered in {"false", "0", "off", "none"}:
        return False
    if lowered in {"true", "1", "on"}:
        return True
    return value


def json_ready(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def split_path_from_yaml(data_yaml: str, split: str) -> Path | None:
    data_path = Path(data_yaml)
    if not data_path.exists():
        return None

    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    split_value = cfg.get(split)
    if split_value is None:
        return None

    root = Path(cfg.get("path", data_path.parent))
    if not root.is_absolute():
        root = data_path.parent / root

    values = split_value if isinstance(split_value, list) else [split_value]
    first = Path(str(values[0]))
    return first if first.is_absolute() else root / first


def count_image_files(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes)


def label_dir_from_image_dir(image_dir: Path | None) -> Path | None:
    if image_dir is None:
        return None
    parts = list(image_dir.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return None


def count_label_files(label_dir: Path | None) -> int | None:
    if label_dir is None or not label_dir.exists() or not label_dir.is_dir():
        return None
    return sum(1 for p in label_dir.glob("*.txt") if p.is_file())


def dataset_counts_from_metrics(metrics, data_yaml: str, split: str) -> dict:
    per_class = metrics.summary(decimals=6) if hasattr(metrics, "summary") else []
    image_dir = split_path_from_yaml(data_yaml, split)
    label_dir = label_dir_from_image_dir(image_dir)
    total_instances = sum(int(row.get("Instances", 0)) for row in per_class)

    return {
        "images": count_image_files(image_dir),
        "label_files": count_label_files(label_dir),
        "classes_present": len(per_class),
        "instances": total_instances,
        "class_image_occurrences": sum(int(row.get("Images", 0)) for row in per_class),
        "image_dir": str(image_dir) if image_dir else None,
        "label_dir": str(label_dir) if label_dir else None,
    }


def validator_save_dir(model: YOLO, fallback: Path) -> Path:
    validator = getattr(model, "validator", None)
    save_dir = getattr(validator, "save_dir", None)
    return Path(save_dir) if save_dir else fallback


def save_metrics_report(metrics, save_dir: Path, split: str, data_yaml: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": split,
        "dataset": dataset_counts_from_metrics(metrics, data_yaml, split),
        "aggregate": json_ready(getattr(metrics, "results_dict", {})),
        "speed": json_ready(getattr(metrics, "speed", {})),
    }
    (save_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if hasattr(metrics, "to_csv"):
        (save_dir / "metrics_per_class.csv").write_text(metrics.to_csv(decimals=6), encoding="utf-8")
    if hasattr(metrics, "to_json"):
        (save_dir / "metrics_per_class.json").write_text(metrics.to_json(decimals=6), encoding="utf-8")

    print(f"Saved metric reports to {save_dir}")


def main() -> None:
    args = parse_args()
    data = str(Path(args.data))
    batch = parse_batch(str(args.batch))
    val_batch = parse_batch(str(args.val_batch)) if args.val_batch is not None else (batch if batch > 0 else 16)
    compile_value = parse_compile(str(args.compile))

    common = {
        "data": data,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "plots": args.plots,
        "rect": args.rect,
        "compile": compile_value,
    }

    train_common = {**common, "batch": batch}
    val_common = {**common, "batch": val_batch, "split": args.split}

    if args.eval_only:
        model = YOLO(args.model)
        metrics = model.val(**val_common)
        save_metrics_report(metrics, validator_save_dir(model, Path(args.project) / args.name), args.split, data)
        print(metrics)
        return

    train_args = {
        **train_common,
        "epochs": args.epochs,
        "cache": args.cache,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "amp": args.amp,
        "val": args.val,
        "mask_ratio": args.mask_ratio,
        "overlap_mask": args.overlap_mask,
        "multi_scale": args.multi_scale,
        "close_mosaic": args.close_mosaic,
    }

    if args.light_augment:
        train_args.update(
            {
                "mosaic": 0.0,
                "copy_paste": 0.0,
                "mixup": 0.0,
                "multi_scale": 0.0,
            }
        )

    if args.mosaic is not None:
        train_args["mosaic"] = args.mosaic
    if args.mixup is not None:
        train_args["mixup"] = args.mixup
    if args.copy_paste is not None:
        train_args["copy_paste"] = args.copy_paste

    model = YOLO(args.model)
    train_results = model.train(**train_args)
    print(train_results)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    eval_model = YOLO(str(best) if best.exists() else args.model)
    metrics = eval_model.val(**val_common)
    save_metrics_report(metrics, validator_save_dir(eval_model, Path(args.project) / args.name), args.split, data)
    print(metrics)


if __name__ == "__main__":
    main()
