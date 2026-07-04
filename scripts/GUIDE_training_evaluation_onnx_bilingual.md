# Food Segmentation Training, Evaluation, and ONNX Guide

English / 繁體中文雙語說明

This guide explains how to use the tracked scripts in this repository for dataset conversion, model training, evaluation, ONNX export, and browser/PWA deployment preparation.  
本文件說明如何使用本 repository 中有版本控管的 scripts，進行資料轉換、模型訓練、模型評估、ONNX 匯出，以及瀏覽器/PWA 部署準備。

Important rule: this guide intentionally uses placeholders for generated data folders, training outputs, model weights, browser build folders, virtual environments, and result summaries. Those artifacts are ignored by git and should not be treated as submitted source files.  
交接重點：本文件刻意使用 placeholder 表示資料集、訓練輸出、模型權重、瀏覽器建置資料夾、虛擬環境與報告輸出。這些檔案通常不會被 git 追蹤，不應視為提交的原始碼。

## 1. Tracked Files In This Repository / 本 repository 會用到的追蹤檔案

The important tracked scripts are:

```text
scripts/convert_coco_to_yolo_seg.py
scripts/train_eval_yolo_seg.py
scripts/prepare_onnx_web_assets.py
scripts/materialize_ultralytics_npy_cache.py
scripts/README_yolo_seg_training.md
scripts/GUIDE_training_evaluation_onnx_bilingual.md
requirements.txt
```

中文說明：

- `convert_coco_to_yolo_seg.py`：將 COCO polygon annotations 轉成 YOLO segmentation labels。
- `train_eval_yolo_seg.py`：訓練與評估 Ultralytics YOLO segmentation model。
- `prepare_onnx_web_assets.py`：將 YOLO `.pt` 權重匯出成 ONNX，並產生 metadata。
- `materialize_ultralytics_npy_cache.py`：可選，用來預先產生 Ultralytics disk cache 相容的 `.npy` 圖片快取。
- `requirements.txt`：記錄此環境使用過的 Python packages。

Use placeholders below:

```text
<PROJECT_ROOT>       Root of this repository
<VENV_DIR>           Python virtual environment directory
<ORIGINAL_COCO_DIR>  Original COCO-format dataset folder
<YOLO_DATASET_DIR>   Output YOLO-format dataset folder
<YOLO_RUN_DIR>       YOLO training output folder
<YOLO_WEIGHTS>       Trained YOLO .pt checkpoint
<WEB_MODEL_DIR>      Folder where browser/PWA ONNX files should be copied
<RFDETR_ROOT>        Separate RF-DETR helper repository folder
<RFDETR_DATASET_DIR> RF-DETR-ready dataset folder
<RFDETR_RUN_DIR>     RF-DETR training output folder
```

中文說明：  
以上 placeholder 代表使用者本機上的資料或輸出位置。請依實際電腦路徑替換，不要把大型資料集、模型權重、訓練輸出直接放進 git。

## 2. Python Environment Setup / Python 環境安裝

Recommended Python version:

```text
Python 3.10 or newer
```

Create and activate a virtual environment:

```powershell
cd <PROJECT_ROOT>
py -3.11 -m venv <VENV_DIR>
<VENV_DIR>\Scripts\activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch first. Example for CUDA 12.8:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install Ultralytics YOLO and helper packages:

```powershell
python -m pip install -U ultralytics
python -m pip install pyyaml opencv-python numpy matplotlib onnx onnxruntime onnxslim
```

Optional: install from the recorded package list:

```powershell
python -m pip install -r requirements.txt
```

Verify GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

中文說明：

- 建議使用 Python virtual environment，避免 package 衝突。
- 若要使用 NVIDIA GPU，PyTorch 必須安裝 CUDA 版本。
- 請以 PyTorch 官方 installation selector 為準，選擇符合機器的 CUDA 版本。

## 3. YOLO Dataset Format / YOLO 資料格式

YOLO instance segmentation expects this layout:

```text
<YOLO_DATASET_DIR>/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  food_seg.yaml
```

Each image has a matching `.txt` label file. Each row is one object instance:

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Rules:

- `class_id` starts at `0`.
- Coordinates are normalized from `0.0` to `1.0`.
- Coordinates are `x y` polygon point pairs.
- Each polygon must have at least 3 points.

中文說明：

- YOLO segmentation 不直接使用 COCO JSON 訓練。
- 每張圖片需要對應一個 `.txt` label。
- 每一行代表一個 instance polygon。

## 4. Convert COCO To YOLO / COCO 轉 YOLO

Dry run:

```powershell
cd <PROJECT_ROOT>

python scripts/convert_coco_to_yolo_seg.py `
  --dataset-dir "<ORIGINAL_COCO_DIR>" `
  --output-dir "<YOLO_DATASET_DIR>" `
  --dry-run
```

Write converted dataset:

```powershell
python scripts/convert_coco_to_yolo_seg.py `
  --dataset-dir "<ORIGINAL_COCO_DIR>" `
  --output-dir "<YOLO_DATASET_DIR>" `
  --overwrite
```

Optional: create a fresh split:

```powershell
python scripts/convert_coco_to_yolo_seg.py `
  --dataset-dir "<ORIGINAL_COCO_DIR>" `
  --output-dir "<YOLO_DATASET_DIR>" `
  --split-ratio 0.70 0.15 0.15 `
  --seed 42 `
  --overwrite
```

Main output:

```text
<YOLO_DATASET_DIR>/food_seg.yaml
```

中文說明：

- `--dry-run` 只檢查與顯示摘要，不寫入檔案。
- `--overwrite` 允許覆蓋既有 label files。
- `--split-ratio` 可重新產生 train/val/test split。

## 5. Train YOLO Segmentation / 訓練 YOLO 分割模型

Example:

```powershell
cd <PROJECT_ROOT>

python scripts/train_eval_yolo_seg.py `
  --data "<YOLO_DATASET_DIR>\food_seg.yaml" `
  --model "yolo26l-seg.pt" `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --val-batch 2 `
  --workers 4 `
  --cache disk `
  --project "<YOLO_RUN_DIR>" `
  --name "food_yolo26_seg" `
  --amp
```

Key options:

| Option | English | 中文 |
|---|---|---|
| `--data` | Dataset YAML path | 資料集 YAML 路徑 |
| `--model` | Starting model or checkpoint | 預訓練模型或 checkpoint |
| `--device 0` | Use GPU 0 | 使用第 0 張 GPU |
| `--batch -1` | Ultralytics training AutoBatch | 訓練時自動估計 batch size |
| `--val-batch` | Validation batch size | 驗證 batch size |
| `--workers` | CPU dataloader workers | CPU 資料載入 worker 數量 |
| `--cache disk` | Store decoded image cache on disk | 將解碼後圖片快取在硬碟 |
| `--amp` | Mixed precision training | 混合精度訓練 |

Important:

- `--batch -1` is for training AutoBatch only.
- Validation/evaluation should use a positive `--val-batch`.
- `--cache disk` caches decoded images; it does not precompute random augmentations.

中文提醒：

- `--batch -1` 只適合訓練 AutoBatch。
- 驗證或測試請使用正數 `--val-batch`。
- `--cache disk` 只快取圖片解碼結果，不會預先產生 augmentation。

## 6. Evaluate YOLO / 評估 YOLO

Validation split:

```powershell
python scripts/train_eval_yolo_seg.py `
  --data "<YOLO_DATASET_DIR>\food_seg.yaml" `
  --model "<YOLO_WEIGHTS>" `
  --device 0 `
  --workers 4 `
  --val-batch 2 `
  --split val `
  --project "<YOLO_RUN_DIR>" `
  --name "food_val" `
  --eval-only
```

Test split:

```powershell
python scripts/train_eval_yolo_seg.py `
  --data "<YOLO_DATASET_DIR>\food_seg.yaml" `
  --model "<YOLO_WEIGHTS>" `
  --device 0 `
  --workers 4 `
  --val-batch 2 `
  --split test `
  --project "<YOLO_RUN_DIR>" `
  --name "food_test" `
  --eval-only
```

Evaluation output files are written inside the selected project/name output directory:

```text
metrics_summary.json
metrics_per_class.csv
metrics_per_class.json
```

中文說明：

- `metrics_summary.json`：整體 precision、recall、mAP。
- `metrics_per_class.csv`：各類別表現。
- 若要公平比較模型，請使用相同 test split。

## 7. Export YOLO To ONNX / YOLO 匯出 ONNX

```powershell
cd <PROJECT_ROOT>

python scripts/prepare_onnx_web_assets.py `
  --model "<YOLO_WEIGHTS>" `
  --data "<YOLO_DATASET_DIR>\food_seg.yaml" `
  --out "<WEB_MODEL_DIR>" `
  --imgsz 640 `
  --batch 1
```

Expected output inside `<WEB_MODEL_DIR>`:

```text
best.onnx
metadata.json
```

中文說明：

- `best.onnx` 是瀏覽器或 PWA 可載入的 YOLO segmentation model。
- `metadata.json` 包含 model type、class names、input size 等資訊。
- 預設不把 NMS 包進 ONNX，後處理由 JavaScript/PWA inference code 完成。

## 8. Optional Image Cache Materialization / 可選圖片快取預先產生

Usually this is not required because Ultralytics can do it during training with `--cache disk`.

```powershell
python scripts/materialize_ultralytics_npy_cache.py `
  --data "<YOLO_DATASET_DIR>\food_seg.yaml" `
  --splits train val
```

中文說明：

- 這只會產生 decoded image `.npy` cache。
- 不會預先產生 random augmentation。
- 一般使用 `--cache disk` 即可，不一定需要執行此 script。

## 9. RF-DETR Helper Workflow / RF-DETR 輔助流程

RF-DETR was handled in a separate helper repository. Use placeholders here and replace them with the actual local paths:

```text
<RFDETR_ROOT>
<RFDETR_DATASET_DIR>
<RFDETR_RUN_DIR>
```

Important RF-DETR helper scripts:

```text
<RFDETR_ROOT>/scripts/prepare_rfdetr_coco.py
<RFDETR_ROOT>/scripts/rfdetr_train_eval.py
<RFDETR_ROOT>/prepare_rfdetr_web_assets.py
```

Install RF-DETR environment:

```powershell
cd <RFDETR_ROOT>
py -3.11 -m venv <VENV_DIR>
<VENV_DIR>\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install rfdetr
python -m pip install "rfdetr[train,loggers,onnx]"
python -m pip install onnx onnxruntime onnxslim supervision opencv-python matplotlib
```

Prepare RF-DETR COCO layout:

```powershell
python scripts/prepare_rfdetr_coco.py `
  --source-dir "<ORIGINAL_COCO_DIR>" `
  --output-dir "<RFDETR_DATASET_DIR>" `
  --mode hardlink
```

Check data:

```powershell
python scripts/rfdetr_train_eval.py check-data `
  --dataset-dir "<RFDETR_DATASET_DIR>"
```

Train RF-DETR segmentation:

```powershell
python scripts/rfdetr_train_eval.py train `
  --dataset-dir "<RFDETR_DATASET_DIR>" `
  --dataset-format roboflow `
  --model seg-large `
  --output-dir "<RFDETR_RUN_DIR>" `
  --epochs 100 `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --num-workers 8 `
  --pin-memory `
  --persistent-workers `
  --prefetch-factor 2 `
  --resolution 504 `
  --accelerator gpu `
  --devices 1 `
  --tensorboard
```

Evaluate RF-DETR:

```powershell
python scripts/rfdetr_train_eval.py eval `
  --dataset-dir "<RFDETR_DATASET_DIR>" `
  --dataset-format roboflow `
  --model seg-large `
  --checkpoint "<RFDETR_RUN_DIR>\checkpoint_best_ema.pth" `
  --split test `
  --output-dir "<RFDETR_RUN_DIR>\eval_test" `
  --batch-size 4 `
  --num-workers 8 `
  --pin-memory `
  --persistent-workers `
  --prefetch-factor 2 `
  --accelerator gpu `
  --devices 1
```

Export RF-DETR ONNX web assets:

```powershell
python prepare_rfdetr_web_assets.py `
  --run-dir "<RFDETR_RUN_DIR>" `
  --checkpoint "<RFDETR_RUN_DIR>\checkpoint_best_ema.pth" `
  --out "<WEB_MODEL_DIR>" `
  --model-file "rfdetr.onnx"
```

中文說明：

- RF-DETR 使用 Roboflow-style COCO layout。
- `persistent-workers` 與 `prefetch-factor` 可改善 CPU dataloader 到 GPU 的資料供應。
- 若要與 YOLO 公平比較，請使用同一個 test split。

## 10. Browser / PWA ONNX Usage / 瀏覽器與 PWA 使用 ONNX

The browser or PWA project should install ONNX Runtime Web:

```powershell
npm install onnxruntime-web
```

The web app should cache and load:

```text
best.onnx
rfdetr.onnx
metadata.json
ONNX Runtime Web WASM assets
```

The app must implement model-specific preprocessing and postprocessing:

| Model | Preprocessing | Postprocessing |
|---|---|---|
| YOLO Seg | Letterbox resize to model size | Decode YOLO boxes, classes, masks, NMS |
| RF-DETR Seg | Direct square resize + ImageNet normalization | Decode RF-DETR boxes/classes/masks |

中文說明：

- YOLO 與 RF-DETR 的 preprocessing 不一樣，不能共用同一段影像處理流程。
- ONNX 只包含模型計算圖，不一定包含完整後處理。
- PWA 需要 cache `.onnx`、metadata、以及 ONNX Runtime Web 的 WASM assets。

## 11. Result Files To Show / 建議展示的結果檔

Do not commit generated result files unless explicitly required by the submission rules. For presentation, prepare a separate external handoff folder containing:

```text
compiled_summary.json
dataset_split_counts.csv
model_metric_comparison.csv
model_files.csv
dataset_split_counts.png
model_metric_comparison.png
segmentation_map50_95_over_training.png
yolo_loss_curves_clean.png
rfdetr_validation_curves.png
metrics_summary.json
metrics_per_class.csv
```

中文說明：

- 給教授看的圖表與 metrics 可以放在獨立交付資料夾。
- 不建議把大型模型、資料集、訓練輸出直接提交到 git。
- 若需要提交模型，建議使用 Git LFS 或雲端下載連結。

## 12. Troubleshooting / 常見問題

### YOLO validation fails with batch `-1`

English: `batch=-1` is only for training AutoBatch. Use a positive `--val-batch` for evaluation.  
中文：`batch=-1` 只適用訓練 AutoBatch。評估時請使用正數 `--val-batch`。

### GPU utilization is low

English:

- Increase `--workers`.
- Use `--cache disk` or sufficient RAM cache.
- Reduce expensive augmentations using `--light-augment`.

中文：

- 增加 `--workers`。
- 使用 disk cache 或足夠 RAM 的 cache。
- 使用 `--light-augment` 降低 CPU augmentation 負擔。

### CPU or RAM is overloaded

English:

- Lower `--workers`.
- Prefer disk cache over RAM cache.
- Reduce batch size.

中文：

- 降低 `--workers`。
- RAM 不足時優先使用 disk cache。
- 降低 batch size。

### ONNX protobuf parsing failed

English: the `.onnx` file is missing, corrupted, incomplete, or replaced by a non-model file. Re-export and verify the file size.  
中文：`.onnx` 可能不存在、損壞、匯出不完整，或被非模型檔案覆蓋。請重新匯出並確認檔案大小。

### Browser cannot load ONNX Runtime WASM assets

English: reinstall web dependencies and ensure ONNX Runtime Web assets are copied to the web app public asset folder.  
中文：請重新安裝 web dependencies，並確認 ONNX Runtime Web 的 WASM assets 已複製到 web app 的 public asset folder。

## 13. References / 參考來源

- Ultralytics Quickstart: https://docs.ultralytics.com/quickstart/
- Ultralytics Train mode: https://docs.ultralytics.com/modes/train/
- Ultralytics Validation mode: https://docs.ultralytics.com/modes/val/
- Ultralytics Export mode: https://docs.ultralytics.com/modes/export/
- PyTorch local installation selector: https://pytorch.org/get-started/locally/
- RF-DETR GitHub: https://github.com/roboflow/rf-detr
- ONNX Runtime Web docs: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
