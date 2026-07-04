from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO segmentation weights to ONNX and web metadata.")
    parser.add_argument("--model", default="runs/weights/best.pt", help="Path to trained .pt weights.")
    parser.add_argument("--data", default="datasets/second_dataset/food_seg.yaml", help="Dataset YAML with names.")
    parser.add_argument("--out", default="onnxruntime_web_demo/public/models", help="Output web model asset folder.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX when supported.")
    parser.add_argument("--nms", action="store_true", help="Try to include NMS in exported ONNX.")
    return parser.parse_args()


def load_names(data_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    data_yaml = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        simplify=True,
        half=args.half,
        nms=args.nms,
    )
    exported_path = Path(exported).resolve()
    target_model = out_dir / "best.onnx"
    if exported_path != target_model:
        shutil.copy2(exported_path, target_model)

    metadata = {
        "modelType": "yolo-seg",
        "modelFile": "best.onnx",
        "task": "segment",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "names": load_names(data_yaml),
        "export": {
            "source_model": str(model_path),
            "nms": args.nms,
            "half": args.half,
            "note": "Browser demo assumes raw YOLO segmentation outputs when nms=false.",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"ONNX model: {target_model}")
    print(f"Metadata:   {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
