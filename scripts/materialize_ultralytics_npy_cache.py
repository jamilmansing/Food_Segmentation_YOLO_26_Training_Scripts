from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".pfm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-create .npy image caches compatible with Ultralytics cache=disk. "
            "This caches decoded images only; augmentations still run during training."
        )
    )
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset YAML keys to materialize, usually train val.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate .npy files even when they already exist.",
    )
    return parser.parse_args()


def as_paths(entry, yaml_dir: Path, root: Path) -> list[Path]:
    if entry is None:
        return []
    values = entry if isinstance(entry, list) else [entry]
    paths: list[Path] = []
    for value in values:
        p = Path(str(value))
        if not p.is_absolute():
            p = root / p
            if not p.exists():
                p = yaml_dir / value
        paths.append(p)
    return paths


def image_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".txt":
        files = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                files.append((path.parent / line).resolve() if not Path(line).is_absolute() else Path(line))
        return files
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    return []


def main() -> None:
    args = parse_args()
    yaml_path = Path(args.data).resolve()
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    yaml_dir = yaml_path.parent
    root = Path(cfg.get("path", yaml_dir))
    if not root.is_absolute():
        root = yaml_dir / root

    candidates: list[Path] = []
    for split in args.splits:
        for p in as_paths(cfg.get(split), yaml_dir, root):
            candidates.extend(image_files(p))

    candidates = sorted({p.resolve() for p in candidates})
    if not candidates:
        raise SystemExit(f"No images found from {yaml_path} splits: {', '.join(args.splits)}")

    made = skipped = failed = 0
    for image_path in candidates:
        npy_path = image_path.with_suffix(".npy")
        if npy_path.exists() and not args.overwrite:
            skipped += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"FAILED unreadable image: {image_path}")
            failed += 1
            continue
        try:
            np.save(npy_path.as_posix(), image, allow_pickle=False)
            made += 1
        except OSError as exc:
            print(f"FAILED writing {npy_path}: {exc}")
            failed += 1

    print(f"Materialized {made} .npy files, skipped {skipped}, failed {failed}.")


if __name__ == "__main__":
    main()
