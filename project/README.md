# YOLO Cassava Leaf Disease Classification Benchmark

Reproducible benchmark (YOLOv3–v12) for 5-class cassava leaf disease classification (`cbb`, `cbsd`, `cgm`, `cmd`, `healthy`) on the iCassava 2019 / FGVCx dataset.

## Citation
Mwebaze et al., "iCassava 2019 Fine-Grained Visual Categorization Challenge", https://arxiv.org/pdf/1908.02900.pdf
Ultralytics YOLO: https://github.com/ultralytics/ultralytics

## Protocol
- Internal stratified hold-out (`data/dataset`): 4512 train / 564 val / 565 test images, split seed 42. Kaggle `data/test` is separate and untouched.
- Training: seed 0, 30 epochs, patience 10, batch 16, imgsz 224, AdamW lr0=1e-3 with cosine decay, class-weighted cross-entropy, label smoothing 0.1, AMP, workers 2, cache off. All models share these settings.
- Routes: **R1** = ImageNet-pretrained native `*-cls` classifiers (v8, v11). **R3** = COCO detection backbone + new `Classify` head (v3, v5, v9, v10, v12); v3 is YOLOv3-tiny. R4 (v4/v6/v7) was skipped because no native classification weights are available.
- The final model was selected by validation macro-F1 **before** the hold-out was opened; the best R3 model was also evaluated once for a route-level comparison. No tuning on the hold-out.

## Final results — 565-image internal hold-out
| Final model | Route | Val top-1 | Val macro-F1 | Test accuracy | Test macro-F1 | Test macro-P | Test macro-R | Test balanced acc |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo11s-cls` (best overall) | R1 | 0.870567 | 0.829294 | 0.8619 | 0.8124 | 0.7988 | 0.8306 | 0.8306 |
| `yolov9s-cls` (best R3) | R3 | 0.842199 | 0.787018 | 0.8637 | 0.8091 | 0.8006 | 0.8315 | 0.8315 |

Full 13-model validation table: [`results/comparison_val.csv`](results/comparison_val.csv). Route-tagged grouped-bar chart: [`results/comparison_val.png`](results/comparison_val.png).

Per-final-model artifacts (per-class metrics, classification report, confusion matrices, PR/accuracy/overall charts, predictions, summary JSON):
- [`results/final/v11_s_pretrained_seed0_test/`](results/final/v11_s_pretrained_seed0_test/)
- [`results/final/v9_s_pretrained_seed0_test/`](results/final/v9_s_pretrained_seed0_test/)

## Setup (Windows)
- Use Python 3.10+ in a venv. Install CUDA PyTorch separately first (see `requirements.txt`), then `pip install -r requirements.txt`; verify `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `2.5.1+cu121 True`.
- Data layout expected under `data set/` (download the iCassava 2019 / FGVCx images separately). Run `python src/prepare_data.py` from `project/`.

## Repository contents
Tracked here: source code, configs, splits, results and the lightweight pretrained weights. Not tracked: raw images (`data set/`, `project/data/{train,val,test,dataset,dataset_smoke}/`), per-epoch checkpoints (`project/runs/`, `project/runs_smoke/`), the local `.venv/`, and `project/yolov3u.pt` (198 MB, above GitHub's 100 MB per-file limit). Download or recreate those locally before reproducing the benchmark.

## Reproduce
```powershell
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\run_benchmark.py --full
.\.venv\Scripts\python.exe src\final_report.py
.\.venv\Scripts\python.exe src\final_report.py --model v9_s_pretrained_seed0
```
Smoke/loader check without touching the hold-out or GPU: `.\.venv\Scripts\python.exe src\final_report.py --dry-run`.

Key code: `src/run_benchmark.py` (sequential training + lock + resume guard + route patch), `src/custom_heads.py` (R3 loader/head), `src/evaluate.py` (validation metrics), `src/final_report.py` (hold-out gates, plots, comparison table), `configs/benchmark.yaml` (single source of truth). Run status: [`PROGRESS.md`](PROGRESS.md).
