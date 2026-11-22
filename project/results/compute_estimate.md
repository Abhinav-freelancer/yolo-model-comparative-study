# Cassava YOLO Benchmark — Compute & Smoke Test Estimate (30 epochs)

- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB), CUDA 12.1
- **Total Models (Smoke)**: 13
- **Smoke Protocol**: 2 epochs on 10% subset, batch 16, imgsz 224
- **Full Protocol**: 30 epochs, patience 10, batch 16, label smoothing 0.1
- **Budget Cap**: 15 GPU-hours
- **Estimated Full Training Time (30 epochs)**: **14.08 GPU-hours**
- **Budget Status**: ✅ Fits within budget

## Per-Model Smoke & Estimate

| Model | Route | Size | Status | Batch | Smoke s/ep | Full s/ep est | 30ep GPU-h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolov5n-cls` | `R3` | `n` | ✅ passed | 16 | 18.78s | 187.8s | **1.565h** |
| `yolov5s-cls` | `R3` | `s` | ✅ passed | 16 | 12.28s | 122.8s | **1.023h** |
| `yolov8n-cls` | `R1` | `n` | ✅ passed | 16 | 10.76s | 107.6s | **0.897h** |
| `yolov8s-cls` | `R1` | `s` | ✅ passed | 16 | 12.37s | 123.7s | **1.031h** |
| `yolov9t-cls` | `R3` | `n` | ✅ passed | 16 | 13.6s | 136.0s | **1.133h** |
| `yolov9s-cls` | `R3` | `s` | ✅ passed | 16 | 13.87s | 138.7s | **1.156h** |
| `yolov10n-cls` | `R3` | `n` | ✅ passed | 16 | 12.2s | 122.0s | **1.017h** |
| `yolov10s-cls` | `R3` | `s` | ✅ passed | 16 | 12.42s | 124.2s | **1.035h** |
| `yolo11n-cls` | `R1` | `n` | ✅ passed | 16 | 12.17s | 121.7s | **1.014h** |
| `yolo11s-cls` | `R1` | `s` | ✅ passed | 16 | 12.5s | 125.0s | **1.042h** |
| `yolov3n-cls` | `R3` | `n` | ✅ passed | 16 | 13.6s | 136.0s | **1.133h** |
| `yolo12n-cls` | `R3` | `n` | ✅ passed | 16 | 12.31s | 123.1s | **1.026h** |
| `yolo12s-cls` | `R3` | `s` | ✅ passed | 16 | 12.05s | 120.5s | **1.004h** |

## GPU Thermal & Clock Summary

✅ No thermal throttling (all runs ≤ 51°C, well below 85°C warning). Low SM clocks after run are normal P8 idle downclock state.