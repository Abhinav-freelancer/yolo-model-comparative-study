#!/usr/bin/env python
"""evaluate.py — full per-split evaluation for Cassava YOLO benchmark.

Usage:
  python evaluate.py --run_dir <path_to_run> --split val
  python evaluate.py --run_dir <path_to_run> --split test
  python evaluate.py --run_dir <path_to_run> --split test --out_csv results/metrics_test.csv

Outputs:
  - Appends a row to results/metrics_val.csv (or metrics_test.csv)
  - Saves PNG confusion matrices under <run_dir>/eval_{split}/
  - Prints a one-line summary string
  - Returns the summary string (when imported)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import yaml

CLASSES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]


def _load_images_and_labels(split_dir: Path) -> tuple[List[Path], List[int]]:
    """Walk split_dir/{class_name}/*.jpg and return (image_paths, label_indices)."""
    images, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            images.append(p)
            labels.append(idx)
    return images, labels


def run_inference(
    model_path: Path,
    image_paths: List[Path],
    imgsz: int = 224,
    batch_size: int = 32,
    device: int = 0,
) -> np.ndarray:
    """Run classification inference; returns softmax probability array [N, nc]."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.model.eval()

    all_probs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = [str(p) for p in image_paths[i : i + batch_size]]
        results = model.predict(
            batch_paths,
            imgsz=imgsz,
            device=device,
            verbose=False,
            stream=False,
        )
        for r in results:
            probs = r.probs.data.cpu().numpy()  # shape [nc]
            all_probs.append(probs)
    return np.array(all_probs)  # [N, nc]


def compute_full_metrics(
    true_labels: List[int],
    pred_labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute classification metrics using sklearn."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.array(true_labels)
    y_pred = pred_labels

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()

    # Log-loss
    ll = float(log_loss(y_true, probs))

    # ECE (Expected Calibration Error), 10 bins
    n_bins = 10
    confidences = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)

    # Macro OvR AUC
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "top1_acc": round(float(acc), 6),
        "balanced_acc": round(float(bal_acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "macro_precision": round(float(macro_p), 6),
        "macro_recall": round(float(macro_r), 6),
        "cohen_kappa": round(float(kappa), 6),
        "mcc": round(float(mcc), 6),
        "macro_ovr_roc_auc": round(float(auc), 6),
        "log_loss": round(ll, 6),
        "ece": round(float(ece), 6),
        "per_class_f1": {c: round(v, 4) for c, v in zip(class_names, per_class_f1)},
        "confusion_matrix": cm,
    }


def save_confusion_matrix_png(
    cm: List[List[int]],
    class_names: List[str],
    out_path: Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix as PNG using matplotlib/seaborn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm_arr = np.array(cm, dtype=float)
        if normalize:
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            cm_arr = cm_arr / np.maximum(row_sums, 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        fmt = ".2f" if normalize else "d"
        sns.heatmap(
            cm_arr, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] Could not save confusion matrix PNG: {e}")


def evaluate_run(
    run_dir: Path,
    split: str,
    cfg: Dict[str, Any],
    out_csv: Optional[Path] = None,
    imgsz: int = 224,
) -> str:
    """Evaluate best.pt from run_dir on the given split.

    Returns one-line summary string.
    Appends a row to out_csv (creates with header if missing).
    """
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        return f"[EVAL ERROR] No best.pt found in {run_dir}"

    # Load run metadata from args.yaml
    args_yaml = run_dir / "args.yaml"
    model_name = run_dir.name  # fallback
    version = "?"
    size = "?"
    route = "?"
    if args_yaml.exists():
        args_data = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
        model_name = args_data.get("name", run_dir.name)

    # Try to infer version/size/route from run name: e.g. "v8_n_pretrained_seed0"
    parts = run_dir.name.split("_")
    if len(parts) >= 2:
        version = parts[0]  # e.g. "v8"
        size = parts[1]      # e.g. "n"

    dataset_dir = ROOT / cfg["paths"]["dataset_dir"]
    split_dir = dataset_dir / split

    if not split_dir.exists():
        return f"[EVAL ERROR] Split dir not found: {split_dir}"

    image_paths, true_labels = _load_images_and_labels(split_dir)
    n = len(image_paths)
    if n == 0:
        return f"[EVAL ERROR] No images found in {split_dir}"

    print(f"  Evaluating {run_dir.name} on {split} ({n} images) ...")
    t0 = time.perf_counter()
    probs = run_inference(best_pt, image_paths, imgsz=imgsz, batch_size=32, device=0)
    pred_labels = probs.argmax(axis=1)
    elapsed = time.perf_counter() - t0

    metrics = compute_full_metrics(true_labels, pred_labels, probs, CLASSES)

    # Save confusion matrix PNGs
    eval_dir = run_dir / f"eval_{split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    save_confusion_matrix_png(
        cm, CLASSES, eval_dir / "confusion_matrix.png",
        normalize=False, title=f"{run_dir.name} — {split} Confusion Matrix"
    )
    save_confusion_matrix_png(
        cm, CLASSES, eval_dir / "confusion_matrix_normalized.png",
        normalize=True, title=f"{run_dir.name} — {split} Confusion Matrix (Normalized)"
    )

    # Determine best epoch from results.csv
    best_epoch = "?"
    results_csv_path = run_dir / "results.csv"
    if results_csv_path.exists():
        try:
            with open(results_csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # Find epoch with best val/metrics/top1
            col_candidates = ["metrics/accuracy_top1", "val/top1", "top1"]
            col = next((c for c in col_candidates if c in (rows[0] if rows else {})), None)
            if col:
                best_val = max(float(r[col]) for r in rows if r.get(col, ""))
                best_epoch = next(
                    r.get("epoch", "?") for r in rows
                    if r.get(col, "") and abs(float(r[col]) - best_val) < 1e-6
                )
        except Exception:
            pass

    # Find epochs_run from results.csv
    epochs_run = "?"
    if results_csv_path.exists():
        try:
            with open(results_csv_path, encoding="utf-8") as f:
                data_rows = [l for l in f if l.strip()]
            epochs_run = len(data_rows) - 1  # minus header
        except Exception:
            pass

    # Compute minutes (from run_dir timestamp metadata if available, else ?)
    minutes = "?"

    # Append to CSV
    if out_csv is None:
        out_csv = ROOT / cfg["paths"]["results_dir"] / f"metrics_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_dir.name,
        "version": version,
        "size": size,
        "route": route,
        "split": split,
        "top1_acc": metrics["top1_acc"],
        "balanced_acc": metrics["balanced_acc"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "cohen_kappa": metrics["cohen_kappa"],
        "mcc": metrics["mcc"],
        "macro_ovr_roc_auc": metrics["macro_ovr_roc_auc"],
        "log_loss": metrics["log_loss"],
        "ece": metrics["ece"],
        "per_class_f1_cbb": metrics["per_class_f1"].get("cbb", ""),
        "per_class_f1_cbsd": metrics["per_class_f1"].get("cbsd", ""),
        "per_class_f1_cgm": metrics["per_class_f1"].get("cgm", ""),
        "per_class_f1_cmd": metrics["per_class_f1"].get("cmd", ""),
        "per_class_f1_healthy": metrics["per_class_f1"].get("healthy", ""),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "inference_seconds": round(elapsed, 1),
    }

    fieldnames = list(row.keys())
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # Build one-line summary
    summary = (
        f"[EVAL-{split.upper()}] {run_dir.name} | {route} | "
        f"top1={metrics['top1_acc']:.4f} | macro_F1={metrics['macro_f1']:.4f} | "
        f"kappa={metrics['cohen_kappa']:.4f} | epochs={epochs_run} | best_ep={best_epoch}"
    )
    print(f"  {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to run directory containing weights/best.pt")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out_csv", default=None, help="CSV to append metrics to (default: results/metrics_{split}.csv)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--imgsz", type=int, default=224)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    out_csv = Path(args.out_csv) if args.out_csv else None

    summary = evaluate_run(run_dir, args.split, cfg, out_csv=out_csv, imgsz=args.imgsz)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
