# YOLO Segmentation Training Scripts

This folder contains a small runner for Ultralytics YOLO instance segmentation.

- `train_eval_yolo_seg.py` trains a YOLO segmentation model and then evaluates the best checkpoint.
- `materialize_ultralytics_npy_cache.py` is optional and usually unnecessary because Ultralytics already creates `.npy` image caches when `cache=disk` is used.

For normal use, start with `train_eval_yolo_seg.py`.

## Install

Install a CUDA-enabled PyTorch build first, then Ultralytics.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U ultralytics
```

`torchaudio` is not required for YOLO training. It is okay to install it if you use the standard PyTorch command.

Check CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Basic Training

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model yolo26n-seg.pt `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --val-batch 16 `
  --workers 4 `
  --cache disk
```

After training, the script evaluates:

```text
runs/segment/food_yolo26_seg/weights/best.pt
```

## Dataset Format

Ultralytics instance segmentation uses a dataset YAML plus one label `.txt` file per image.

Recommended folder layout:

```text
dataset_root/
  images/
    train/
      img_0001.jpg
      img_0002.jpg
    val/
      img_0101.jpg
  labels/
    train/
      img_0001.txt
      img_0002.txt
    val/
      img_0101.txt
  food_seg.yaml
```

Example `food_seg.yaml`:

```yaml
path: C:/NFU_PROJECT_FOOD_SEGMENTATION_3/datasets/food_seg
train: images/train
val: images/val
test: images/test

names:
  0: rice
  1: meat
  2: vegetable
  3: soup
```

`test` is optional. `train` and `val` are the important ones.

Each image should have a matching label file with the same stem:

```text
images/train/img_0001.jpg
labels/train/img_0001.txt
```

For instance segmentation, each row in the `.txt` file is one object instance:

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Example:

```text
0 0.412 0.203 0.438 0.198 0.481 0.236 0.463 0.288 0.421 0.276
2 0.122 0.650 0.201 0.621 0.260 0.704 0.172 0.756
```

Rules:

- `class_id` starts at `0`.
- Polygon coordinates are normalized to `0.0-1.0`.
- Coordinates are `x y` pairs.
- Each polygon needs at least 3 points.
- The label file can contain multiple rows for multiple food instances.
- If an image has no objects, its label file can be empty.
- Image and label names must match exactly except for extension.

Do not use mask PNG files directly for YOLO instance segmentation unless you convert them first. The model expects polygon labels, not per-pixel class mask files, for the `segment` task.

If your labels are currently COCO JSON, binary masks, or semantic PNG masks, convert them into YOLO polygon `.txt` labels before training.

## Convert COCO JSON To YOLO Segmentation

Ultralytics includes a COCO converter in the package:

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="path/to/coco/annotations",
    save_dir="path/to/yolo_converted",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)
```

Use `use_segments=True` for instance segmentation. If this is left as `False`, the converter writes bounding-box labels instead of segmentation polygons.

`labels_dir` should point to the folder containing COCO JSON files such as:

```text
instances_train.json
instances_val.json
```

The converter writes YOLO label files under:

```text
path/to/yolo_converted/labels/train/
path/to/yolo_converted/labels/val/
```

depending on the JSON names. For example, `instances_train.json` becomes `labels/train/`.

Important: this converts annotations. It does not magically inspect your whole project and build a final dataset YAML for you. Make sure the images are arranged under matching image folders, then create a dataset YAML that points to those image folders.

Example after conversion:

```text
yolo_converted/
  images/
    train/
    val/
  labels/
    train/
    val/
  food_seg.yaml
```

Example YAML:

```yaml
path: C:/NFU_PROJECT_FOOD_SEGMENTATION_3/datasets/yolo_converted
train: images/train
val: images/val

names:
  0: rice
  1: meat
  2: vegetable
```

If your COCO category IDs are already your own contiguous class IDs starting from `1`, use `cls91to80=False`. The converter subtracts 1, producing YOLO class IDs starting from `0`.

Only use `cls91to80=True` for standard COCO category IDs that need mapping from COCO's 91-category ID space to the usual 80-class YOLO indexing.

## Convert This COCO Export

For the current dataset export:

```text
C:\Users\Jamil Gwapo\Downloads\to_combine_4_actual
```

use the project converter:

```powershell
.\.venv\Scripts\python.exe scripts\convert_coco_loco_to_yolo_seg.py `
  --dataset-dir "C:\Users\Jamil Gwapo\Downloads\to_combine_4_actual" `
  --overwrite
```

This reads:

```text
annotations_train_coco.json
annotations_val_coco.json
annotations_test_coco.json
```

and writes:

```text
labels/train/*.txt
labels/val/*.txt
labels/test/*.txt
food_seg.yaml
```

Dry-run first if you want to validate without writing:

```powershell
.\.venv\Scripts\python.exe scripts\convert_coco_to_yolo_seg.py `
  --dataset-dir "C:\Users\Jamil Gwapo\Downloads\to_combine_4_actual" `
  --dry-run
```

The dry run for this export reported:

```text
train: 128 images, 587 objects
val:    28 images, 146 objects
test:   28 images, 147 objects
```

To make a fresh train/val/test split from the combined `annotations_coco.json`, use `--split-ratio`.

Dry-run:

```powershell
.\.venv\Scripts\python.exe scripts\convert_coco_to_yolo_seg.py `
  --dataset-dir "C:\Users\Jamil Gwapo\Downloads\to_combine_4_actual" `
  --seed 42 `
  --dry-run
```

Write the resplit dataset:

```powershell
.\.venv\Scripts\python.exe scripts\convert_coco_to_yolo_seg.py `
  --dataset-dir "C:\Users\Jamil Gwapo\Downloads\to_combine_4_actual" `
  --seed 42 `
  --overwrite
```

By default, resplitting writes a new dataset under:

```text
C:\Users\Jamil Gwapo\Downloads\to_combine_4_actual\yolo_resplit
```

This keeps the original export untouched. For this current dataset, the existing split is already about 70/15/15 by image count, so resplitting is optional.

## Evaluation Only

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model runs\segment\food_yolo26_seg\weights\best.pt `
  --device 0 `
  --workers 4 `
  --val-batch 16 `
  --split val `
  --eval-only
```

For test-set evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model runs\segment\food_yolo26_seg\weights\best.pt `
  --device 0 `
  --workers 4 `
  --val-batch 16 `
  --split test `
  --project runs/segment `
  --name food_test `
  --eval-only
```

## Arguments

### Required

`--data`

Path to the Ultralytics dataset YAML. For instance segmentation, labels should use YOLO polygon format.

Example:

```powershell
--data data\food_seg.yaml
```

### Model And Training Length

`--model`

Model checkpoint or model YAML. For pretrained YOLO26 segmentation:

```powershell
--model yolo26n-seg.pt
```

Ultralytics downloads pretrained weights automatically the first time.

`--epochs`

Number of training epochs.

```powershell
--epochs 100
```

`--patience`

Early stopping patience. Training stops if validation does not improve for this many epochs.

```powershell
--patience 50
```

### GPU And Batch Settings

`--device`

Device to train on.

```powershell
--device 0
--device 0,1
--device cpu
```

`--batch`

Training batch size. Use `-1` for Ultralytics train-time AutoBatch.

```powershell
--batch -1
--batch 8
--batch 16
```

`batch=-1` is for training AutoBatch only. Standalone validation expects a positive batch size, so this wrapper uses `--val-batch` for final validation.

`--val-batch`

Validation batch size. If omitted, it uses `--batch` when `--batch` is positive; otherwise it defaults to `16`.

```powershell
--val-batch 16
--val-batch 8
```

`--split`

Dataset split to use for `--eval-only` and final validation after training.

```powershell
--split val
--split test
--split train
```

`--imgsz`

Training image size.

```powershell
--imgsz 640
```

Larger values can improve small-object/detail performance but increase GPU memory and CPU preprocessing cost.

`--amp` / `--no-amp`

Enable or disable automatic mixed precision. Default is enabled.

```powershell
--amp
```

`--compile`

PyTorch 2 compile setting. Default is `False`.

```powershell
--compile False
--compile True
--compile reduce-overhead
--compile max-autotune-no-cudagraphs
```

Recommendation: leave it off until you have a stable baseline, then test `reduce-overhead`.

### CPU/Data Pipeline Settings

`--workers`

Number of dataloader workers per GPU/rank.

```powershell
--workers 4
--workers 8
```

Higher values can keep the GPU fed, but they increase CPU and RAM pressure. On Windows, if training hangs, try `--workers 0`, then increase to `2`, `4`, and `8`.

Ultralytics uses `InfiniteDataLoader`, so worker reuse is already handled internally. It also sets `prefetch_factor=4` when `workers > 0`. These are not exposed as normal training args.

`--cache`

Image caching mode.

```powershell
--cache none
--cache disk
--cache ram
```

- `none`: read image files normally.
- `disk`: create/load `.npy` decoded-image caches beside images.
- `ram`: cache resized images in memory when enough RAM is available.

Recommendation: use `disk` first. Use `ram` only if you have enough memory.

`--rect` / `--no-rect`

Enable rectangular training batches.

```powershell
--rect
```

This can reduce padding and wasted compute for datasets with varied aspect ratios. It can also reduce some augmentation behavior, so compare results.

### Segmentation Mask Settings

`--mask-ratio`

Downsample ratio for segmentation masks during training.

```powershell
--mask-ratio 4
--mask-ratio 8
```

Default is `4`. Larger values reduce mask resolution and can reduce memory/compute, but may hurt mask detail.

`--overlap-mask` / `--no-overlap-mask`

Controls how overlapping instance masks are represented during training.

```powershell
--overlap-mask
--no-overlap-mask
```

Default is enabled. Keep the default unless you have a reason to compare behavior.

### Augmentation Settings

`--light-augment`

Convenience option that disables heavier CPU-side augmentations:

```text
mosaic=0
copy_paste=0
mixup=0
multi_scale=0
```

Use this if CPU load is too high or GPU is waiting on the dataloader.

`--mosaic`

Mosaic augmentation probability.

```powershell
--mosaic 1.0
--mosaic 0.0
```

Mosaic can help generalization but increases CPU-side loading/augmentation work.

`--mixup`

MixUp augmentation probability.

```powershell
--mixup 0.0
```

`--copy-paste`

Segmentation copy-paste augmentation probability.

```powershell
--copy-paste 0.0
```

Can be useful for segmentation, but adds CPU-side augmentation cost.

`--multi-scale`

Randomly varies training image size by a fraction of `imgsz`.

```powershell
--multi-scale 0.0
--multi-scale 0.25
```

Can improve robustness but adds variability in compute and memory load.

`--close-mosaic`

Disable mosaic during the final N epochs.

```powershell
--close-mosaic 10
--close-mosaic 0
```

Default is `10`.

### Output Settings

`--project`

Base output folder.

```powershell
--project runs/segment
```

`--name`

Run name inside the project folder.

```powershell
--name food_yolo26_seg
```

`--plots` / `--no-plots`

Enable or disable training plots.

```powershell
--plots
```

`--val` / `--no-val`

Enable or disable validation during training.

```powershell
--val
```

`--eval-only`

Skip training and run validation only.

```powershell
--eval-only
```

## Recommended Profiles

### Balanced Starting Point

Use this first.

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model yolo26n-seg.pt `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --workers 4 `
  --cache disk `
  --amp `
  --mask-ratio 4
```

### Higher GPU Utilization

Use when GPU utilization is low and CPU/RAM are not saturated.

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model yolo26n-seg.pt `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --workers 8 `
  --cache ram `
  --amp
```

### Lower CPU Load

Use when CPU is pegged, fans are screaming, or training stalls waiting on preprocessing.

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model yolo26n-seg.pt `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --workers 2 `
  --cache disk `
  --light-augment
```

### Lower Memory Pressure

Use if GPU memory is tight.

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model yolo26n-seg.pt `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch 4 `
  --workers 4 `
  --cache disk `
  --mask-ratio 8 `
  --light-augment
```

### Test PyTorch Compile

Try this only after a normal run works.

```powershell
.\.venv\Scripts\python.exe scripts\train_eval_yolo_seg.py `
  --data path\to\dataset.yaml `
  --model yolo26n-seg.pt `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --workers 4 `
  --cache disk `
  --compile reduce-overhead
```

## Tuning Guide

If GPU utilization is low:

- Increase `--workers` from `4` to `8`.
- Use `--cache disk` or `--cache ram`.
- Check whether the dataset is on an SSD.
- Reduce costly augmentations with `--light-augment`.

If CPU utilization is too high:

- Reduce `--workers`.
- Use `--light-augment`.
- Keep `--multi-scale 0.0`.
- Keep `--mosaic 0.0` if needed.

If RAM is too high:

- Use `--cache disk` instead of `--cache ram`.
- Reduce `--workers`.
- Reduce `--batch`.

If GPU memory is too high:

- Reduce `--batch`.
- Reduce `--imgsz`.
- Increase `--mask-ratio` from `4` to `8`.
- Use a smaller model such as `yolo26n-seg.pt`.

If training hangs on Windows:

- Try `--workers 0`.
- If stable, try `--workers 2`, then `4`.
- Make sure Python training entrypoints use `if __name__ == "__main__":`, which this script already does.

## About Dataset Preconversion

Ultralytics already supports image materialization through:

```powershell
--cache disk
```

This creates `.npy` decoded-image caches beside your image files. It speeds up repeated image loading by avoiding repeated JPEG/PNG decoding.

It does not precompute random augmentations. Augmentations remain online during training so they can vary between batches and epochs.

The optional `materialize_ultralytics_npy_cache.py` script only exists if you want to pre-create those `.npy` files before training. For most runs, skip it and let `--cache disk` handle caching automatically.
