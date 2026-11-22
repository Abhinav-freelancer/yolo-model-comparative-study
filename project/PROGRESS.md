# PROGRESS — Cassava YOLO Benchmark

**Last updated**: 2026-11-20 13:38:32

## Training Status

| # | Model | Version | Size | Route | Status | Val Top-1 | Val Macro-F1 | Epochs | Best Ep | Min |
| --: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `yolov8n-cls` | v8 | n | `R1` | ✅ passed | 0.845745 | 0.794849 | 27 | 17 | 12.7 |
| 2 | `yolov8s-cls` | v8 | s | `R1` | ✅ passed | 0.868794 | 0.828997 | 21 | 11 | 8.6 |
| 3 | `yolo11n-cls` | v11 | n | `R1` | ✅ passed | 0.856383 | 0.803592 | 30 | 25 | 13.3 |
| 4 | `yolo11s-cls` | v11 | s | `R1` | ✅ passed | 0.870567 | 0.829294 | 30 | 20 | 12.5 |
| 5 | `yolov3n-cls` | v3 | n | `R3` | ✅ passed | 0.85461 | 0.803331 | 30 | 25 | 12.1 |
| 6 | `yolov5n-cls` | v5 | n | `R3` | ✅ passed | 0.820922 | 0.763419 | 30 | 28 | 13.0 |
| 7 | `yolov5s-cls` | v5 | s | `R3` | ✅ passed | 0.81383 | 0.755818 | 30 | 24 | 17.0 |
| 8 | `yolov9t-cls` | v9 | n | `R3` | ✅ passed | 0.824468 | 0.767563 | 30 | 29 | 21.5 |
| 9 | `yolov9s-cls` | v9 | s | `R3` | ✅ passed | 0.842199 | 0.787018 | 30 | 29 | 20.1 |
| 10 | `yolov10n-cls` | v10 | n | `R3` | ✅ passed | 0.810284 | 0.73668 | 30 | 27 | 16.4 |
| 11 | `yolov10s-cls` | v10 | s | `R3` | ✅ passed | 0.771277 | 0.708735 | 30 | 20 | 17.1 |
| 12 | `yolo12n-cls` | v12 | n | `R3` | ✅ passed | 0.810284 | 0.750708 | 30 | 29 | 17.2 |
| 13 | `yolo12s-cls` | v12 | s | `R3` | ✅ passed | 0.810284 | 0.743706 | 30 | 29 | 16.9 |

## Notes
- R1 models: ImageNet-pretrained classifiers (native Ultralytics *-cls.pt)
- R3 models: COCO detection backbones + new Classify head (custom R3 route)
- R1 vs R3 performance gaps are NOT purely architectural (pretraining domain differs)
- Protocol: 30 epochs, patience 10, batch 16, imgsz 224, class-weighted CE, label smoothing 0.1, seed 0
- Hold-out (565 images) evaluated ONCE at the end for final models only

<!-- STATE_JSON:{"models_done": [{"version": "v8", "size": "n", "route": "R1", "model_name": "yolov8n-cls", "train_status": "passed", "val_top1": "0.845745", "val_macro_f1": "0.794849", "epochs_run": "27", "best_epoch": "17", "minutes": 12.7}, {"version": "v8", "size": "s", "route": "R1", "model_name": "yolov8s-cls", "train_status": "passed", "val_top1": "0.868794", "val_macro_f1": "0.828997", "epochs_run": "21", "best_epoch": "11", "minutes": 8.6}, {"version": "v11", "size": "n", "route": "R1", "model_name": "yolo11n-cls", "train_status": "passed", "val_top1": "0.856383", "val_macro_f1": "0.803592", "epochs_run": "30", "best_epoch": "25", "minutes": 13.3}, {"version": "v11", "size": "s", "route": "R1", "model_name": "yolo11s-cls", "train_status": "passed", "val_top1": "0.870567", "val_macro_f1": "0.829294", "epochs_run": "30", "best_epoch": "20", "minutes": 12.5}, {"version": "v3", "size": "n", "route": "R3", "model_name": "yolov3n-cls", "train_status": "passed", "val_top1": "0.85461", "val_macro_f1": "0.803331", "epochs_run": "30", "best_epoch": "25", "minutes": 12.1}, {"version": "v5", "size": "n", "route": "R3", "model_name": "yolov5n-cls", "train_status": "passed", "val_top1": "0.820922", "val_macro_f1": "0.763419", "epochs_run": "30", "best_epoch": "28", "minutes": 13.0}, {"version": "v5", "size": "s", "route": "R3", "model_name": "yolov5s-cls", "train_status": "passed", "val_top1": "0.81383", "val_macro_f1": "0.755818", "epochs_run": "30", "best_epoch": "24", "minutes": 17.0}, {"version": "v9", "size": "n", "route": "R3", "model_name": "yolov9t-cls", "train_status": "passed", "val_top1": "0.824468", "val_macro_f1": "0.767563", "epochs_run": "30", "best_epoch": "29", "minutes": 21.5}, {"version": "v9", "size": "s", "route": "R3", "model_name": "yolov9s-cls", "train_status": "passed", "val_top1": "0.842199", "val_macro_f1": "0.787018", "epochs_run": "30", "best_epoch": "29", "minutes": 20.1}, {"version": "v10", "size": "n", "route": "R3", "model_name": "yolov10n-cls", "train_status": "passed", "val_top1": "0.810284", "val_macro_f1": "0.73668", "epochs_run": "30", "best_epoch": "27", "minutes": 16.4}, {"version": "v10", "size": "s", "route": "R3", "model_name": "yolov10s-cls", "train_status": "passed", "val_top1": "0.771277", "val_macro_f1": "0.708735", "epochs_run": "30", "best_epoch": "20", "minutes": 17.1}, {"version": "v12", "size": "n", "route": "R3", "model_name": "yolo12n-cls", "train_status": "passed", "val_top1": "0.810284", "val_macro_f1": "0.750708", "epochs_run": "30", "best_epoch": "29", "minutes": 17.2}, {"version": "v12", "size": "s", "route": "R3", "model_name": "yolo12s-cls", "train_status": "passed", "val_top1": "0.810284", "val_macro_f1": "0.743706", "epochs_run": "30", "best_epoch": "29", "minutes": 16.9}], "last_update": "2026-11-20T13:38:32"}:STATE_JSON -->
<!-- final_report 2026-11-20 13:39:26 -->
All 13 models finished under PID 34712; FULL BENCHMARK COMPLETE, lock released, no python processes. metrics_val.csv has 13 rows (4 R1 + 9 R3), no duplicates. Best val macro-F1: v11_s_pretrained_seed0 (R1, 0.829294); best R3: v9_s_pretrained_seed0 (0.787018). Proceeding to route-tagged comparison + single hold-out evaluation per final model.


<!-- final_report 2026-11-20 13:39:48 -->
- final_report HOLD-OUT: `v11_s_pretrained_seed0` [R1] | split=test | n=565 | acc=0.8619 | macroF1=0.8124 | artifacts=results\final\v11_s_pretrained_seed0_test


<!-- final_report 2026-11-20 13:40:09 -->
- final_report HOLD-OUT: `v9_s_pretrained_seed0` [R3] | split=test | n=565 | acc=0.8637 | macroF1=0.8091 | artifacts=results\final\v9_s_pretrained_seed0_test


<!-- final_report 2026-11-20 15:06:09 -->
STEP 0 STOPPED: Get-Process python = none, but nvidia-smi failed with exact error: NVIDIA-SMI has failed because you do not have sufficient permissions. Please try running as an administrator. Per instruction (stop on error), no further steps run.


<!-- final_report 2026-11-20 15:23:58 -->
FINAL PIPELINE ERROR: see results\final_pipeline_error.log


<!-- final_report 2026-11-20 15:25:23 -->
Pipeline helper argument corrected: collect_images needs name->index map (KeyError was in helper setup, not benchmark data). Relaunching final pipeline.


<!-- final_report 2026-11-20 15:26:05 -->
[STEP0] failures.csv: results\failures.csv absent; no failed runs were recorded (all 13 passed); v10n/v12n/v12s each 457/564 correct on shared val with 3 distinct best.pt files and 3 distinct saved prediction CSVs; v3 display label set to YOLOv3-tiny.


<!-- final_report 2026-11-20 15:26:20 -->
- final_report HOLD-OUT: `v8_s_pretrained_seed0` [R1] | split=test | n=565 | acc=0.8743 | macroF1=0.8292 | artifacts=results\final\v8_s_pretrained_seed0_test


<!-- final_report 2026-11-20 15:26:22 -->
[STEP1] hold-out v11_s_pretrained_seed0: n=565 acc=0.861947 macroF1=0.812358 balanced_acc=0.830601; artifacts=D:\personal projects\yolo models project\project\results\final\v11_s_pretrained_seed0_test; reused prior single eval


<!-- final_report 2026-11-20 15:26:22 -->
[STEP1] hold-out v8_s_pretrained_seed0: n=565 acc=0.874336 macroF1=0.829212 balanced_acc=0.845607; artifacts=D:\personal projects\yolo models project\project\results\final\v8_s_pretrained_seed0_test; evaluated once now


<!-- final_report 2026-11-20 15:26:22 -->
[STEP2] wrote route-tagged 13-model comparison table comparison_val.csv and one grouped bar chart comparison_val.png; v3 labelled YOLOv3-tiny


<!-- final_report 2026-11-20 15:28:02 -->
[STEP3] wrote efficiency.csv/json for 13 models: thop GFLOPs at 224, params, best.pt MB, GPU latency batch1/batch32 (warmup 20, mean over 200) on RTX 3050 Laptop 4GB


<!-- final_report 2026-11-20 15:28:16 -->
[STEP4] bootstrap 95% CI and McNemar complete; p=0.391603; significant_at_0_05=False


<!-- final_report 2026-11-20 15:32:14 -->
[STEP5] created 7 Kaggle submissions for top3 ['v11_s_pretrained_seed0', 'v8_s_pretrained_seed0', 'v11_n_pretrained_seed0'] (plain, flip-TTA, top3 mean-softmax ensemble); all CSVs passed structural validation against sample; hidden labels not checkable


<!-- final_report 2026-11-20 15:32:14 -->
FINAL PIPELINE ERROR: see results\final_pipeline_error.log


<!-- final_report 2026-11-20 15:33:55 -->
Report step helper corrected: confusion matrix recomputed from saved hold-out predictions.csv because summary.json intentionally omits it. Running report-only.


<!-- final_report 2026-11-20 15:34:09 -->
[STEP6] wrote D:\personal projects\yolo models project\project\results\REPORT.md with dataset, routes, protocol, ranked table, per-class results, efficiency, statistics, error analysis, submissions, recommendations, limitations and citations


<!-- final_report 2026-11-20 15:35:48 -->
Report regenerated with per-class tables for both v11s and v8s and overall/balanced values included.


<!-- final_report 2026-11-20 15:36:11 -->
[STEP6] wrote D:\personal projects\yolo models project\project\results\REPORT.md with dataset, routes, protocol, ranked table, per-class results, efficiency, statistics, error analysis, submissions, recommendations, limitations and citations

