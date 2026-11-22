#!/usr/bin/env python
"""final_report.py - reviewer-facing final-model evaluation for the cassava benchmark.

Selects the best model by validation macro-F1 (or --model), loads best.pt with the
same Ultralytics ClassificationModel loader used by train.py (works for R1 and for
R3 detection-backbone + Classify-head models), and evaluates ONCE on the internal
565-image test hold-out. Class-name-to-index mapping always comes from the loaded
model's own ``names`` attribute.

Usage:
  python src\\final_report.py                     # real hold-out evaluation (gated)
  python src\\final_report.py --model v8_n_pretrained_seed0
  python src\\final_report.py --force             # override the safety gates
  python src\\final_report.py --dry-run           # VAL subset, CPU only, no test set

Outputs:
  results/final/<run>_test/*        (metrics, reports, plots, confusion matrices)
  results/comparison_val.csv        (all finished models, route-tagged)
  results/comparison_val.png        (grouped bars, route-tagged)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_IS_DRY = "--dry-run" in sys.argv
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
if _IS_DRY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROGRESS_PATH = ROOT / "PROGRESS.md"
FALLBACK_PATH = ROOT / "results" / "progress_fallback.log"


def append_progress(message: str) -> None:
    """Append a short note to PROGRESS.md; never let a notes failure abort the run."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = f"\n<!-- final_report {stamp} -->\n{message.strip()}\n\n"
    try:
        with open(PROGRESS_PATH, "a", encoding="utf-8") as handle:
            handle.write(block)
    except Exception as exc:
        try:
            FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FALLBACK_PATH, "a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] PROGRESS.md write failed: {exc}\n{message.strip()}\n")
        except Exception:
            pass


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dataset_root(cfg: Dict[str, Any]) -> Path:
    primary = ROOT / cfg["paths"]["dataset_dir"]
    if primary.exists():
        return primary
    alternate = ROOT / "dataset"
    if alternate.exists():
        return alternate
    return primary


def results_root(cfg: Dict[str, Any]) -> Path:
    path = ROOT / cfg["paths"]["results_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_root(cfg: Dict[str, Any]) -> Path:
    return ROOT / cfg["paths"]["runs_dir"]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            return [row for row in csv.DictReader(handle) if any(v.strip() for v in row.values())]
    except Exception:
        return []


def as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def expected_benchmark_runs(res_dir: Path) -> List[Dict[str, str]]:
    path = res_dir / "available_runs.json"
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    ordered = ["v8_n", "v8_s", "v11_n", "v11_s"]
    selected = []
    for record in records:
        if record.get("version") == "v26-bonus" or record.get("size") == "std":
            continue
        if record.get("size") not in ("n", "s"):
            continue
        selected.append(record)
    selected.sort(key=lambda r: (ordered.index(f"{r['version']}_{r['size']}")
                                 if f"{r['version']}_{r['size']}" in ordered else 99,
                                 int(str(r["version"]).lstrip("v"))
                                 if str(r["version"]).lstrip("v").isdigit() else 99,
                                 r["size"]))
    return selected


def route_from_run_name(run_name: str) -> Optional[str]:
    token = run_name.split("_", 1)[0]
    if token in ("v8", "v11"):
        return "R1"
    if token in ("v3", "v5", "v9", "v10", "v12"):
        return "R3"
    return None


def route_for_run(run_name: str, val_rows: List[Dict[str, str]],
                  avail_runs: List[Dict[str, str]]) -> str:
    for row in val_rows:
        route = str(row.get("route") or "").strip()
        if row.get("run_name") == run_name and route not in ("", "?", "unknown"):
            return route
    by_name = route_from_run_name(run_name)
    if by_name:
        return by_name
    for record in avail_runs:
        route = str(record.get("route") or "").strip()
        if record.get("run_name") == run_name and route not in ("", "?", "unknown"):
            return route
    return "R1" if run_name.startswith(("v8_", "v11_")) else "R3"


def meta_for_run(run_name: str, val_rows: List[Dict[str, str]],
                 avail_runs: List[Dict[str, str]]) -> Dict[str, str]:
    meta = {"run_name": run_name, "route": route_for_run(run_name, val_rows, avail_runs),
            "version": "?", "size": "?", "model_name": run_name}
    for record in avail_runs:
        if record.get("run_name") == run_name:
            meta.update({k: str(record[k]) for k in ("version", "size", "model_name") if k in record})
            break
    for row in val_rows:
        if row.get("run_name") == run_name:
            for key in ("version", "size"):
                if row.get(key) and meta[key] == "?":
                    meta[key] = str(row[key])
            break
    if meta["version"] == "?" and "_" in run_name:
        meta["version"], meta["size"] = run_name.split("_", 1)[0], run_name.split("_", 1)[1]
    return meta


def resolve_run_dir(argument: str, cfg: Dict[str, Any]) -> Path:
    candidate = Path(argument)
    if candidate.exists():
        return candidate.resolve()
    for base in (runs_root(cfg), ROOT):
        probe = base / argument
        if probe.exists():
            return probe.resolve()
    raise FileNotFoundError(f"Could not resolve --model '{argument}' under {runs_root(cfg)} or {ROOT}")


def select_final_run(args: argparse.Namespace, cfg: Dict[str, Any],
                     val_rows: List[Dict[str, str]], res_dir: Path,
                     avail_runs: List[Dict[str, str]]) -> Path:
    if args.model:
        run_dir = resolve_run_dir(args.model, cfg)
        print(f"[select] --model override: {run_dir.name}")
        return run_dir
    scored = [(as_float(row.get("macro_f1")), row.get("run_name", "")) for row in val_rows]
    scored = [(score, name) for score, name in scored if score is not None and name]
    if scored:
        best_score, best_name = max(scored, key=lambda item: item[0])
        run_dir = runs_root(cfg) / best_name
        print(f"[select] best val macro-F1 = {best_score:.6f} -> {best_name}")
        return run_dir
    print("[select] metrics_val.csv has no usable macro-F1 yet; falling back to an existing run with best.pt")
    candidates = sorted(p for p in runs_root(cfg).iterdir()
                        if p.is_dir() and (p / "weights" / "best.pt").exists()) if runs_root(cfg).exists() else []
    if not candidates:
        raise RuntimeError("No finished run with weights/best.pt is available for selection")
    run_dir = candidates[0]
    print(f"[select] fallback run: {run_dir.name}")
    return run_dir


def load_model_and_names(run_dir: Path, device: Any) -> Tuple[Any, List[str], Dict[int, str]]:
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Missing weights: {best_pt}")
    model = YOLO(str(best_pt))
    model.to(device)
    model.model.eval()

    checkpoint = torch.load(str(best_pt), map_location="cpu", weights_only=False)
    saved_model = checkpoint.get("model")
    raw_names = getattr(saved_model, "names", None) or getattr(model, "names", None) or {}
    names = {int(k): str(v) for k, v in raw_names.items()}
    ordered = [names[i] for i in sorted(names)]
    print(f"[loader] {run_dir.name} | head={type(model.model.model[-1]).__name__} "
          f"| model.names={ordered}")
    return model, ordered, names


def fallback_class_names(split_dir: Path) -> List[str]:
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def collect_images(split_dir: Path, class_names: List[str], name_to_index: Dict[str, int],
                   limit: Optional[int] = None) -> Tuple[List[Path], List[int]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    images: List[Path] = []
    labels: List[int] = []
    for name in class_names:
        class_dir = split_dir / name
        if not class_dir.exists():
            continue
        if limit is None:
            chosen = sorted(class_dir.glob("*.jpg"))
        else:
            per_class = max(1, limit // max(1, len(class_names)))
            chosen = sorted(class_dir.glob("*.jpg"))[:per_class]
        for image in chosen:
            if name not in name_to_index:
                raise KeyError(f"Class folder '{name}' is absent from the model's own names {name_to_index}")
            images.append(image)
            labels.append(name_to_index[name])
    if limit is not None and len(images) > limit:
        images, labels = images[:limit], labels[:limit]
    if not images:
        raise RuntimeError(f"No images collected from {split_dir}")
    return images, labels


def predict_probs(model: Any, images: List[Path], imgsz: int, device: Any,
                  batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = [str(p) for p in images[start:start + batch_size]]
        results = model.predict(batch, imgsz=imgsz, device=device, verbose=False, stream=False)
        for result in results:
            chunks.append(result.probs.data.cpu().numpy())
    return np.vstack(chunks)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix,
        precision_recall_fscore_support, classification_report as sk_report,
    )
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                               average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    one_vs_rest = []
    for index in labels:
        tp = int(cm[index, index])
        fn = int(cm[index, :].sum() - tp)
        fp = int(cm[:, index].sum() - tp)
        tn = total - tp - fn - fp
        one_vs_rest.append((tp + tn) / total if total else 0.0)
    return {
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
        "f1": [float(v) for v in f1],
        "support": [int(v) for v in support],
        "one_vs_rest_accuracy": [float(v) for v in one_vs_rest],
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.astype(int).tolist(),
        "report_text": sk_report(y_true, y_pred, labels=labels, target_names=class_names,
                                 zero_division=0),
    }


def write_per_class_csv(path: Path, metrics: Dict[str, Any], class_names: List[str]) -> None:
    fields = ["metric", "precision", "recall", "f1", "one_vs_rest_accuracy", "support", "value"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(class_names):
            writer.writerow({
                "metric": name, "precision": f"{metrics['precision'][index]:.6f}",
                "recall": f"{metrics['recall'][index]:.6f}", "f1": f"{metrics['f1'][index]:.6f}",
                "one_vs_rest_accuracy": f"{metrics['one_vs_rest_accuracy'][index]:.6f}",
                "support": metrics["support"][index], "value": "",
            })
        for label, prefix in (("macro_avg", "macro"), ("weighted_avg", "weighted")):
            writer.writerow({
                "metric": label, "precision": f"{metrics[prefix + '_precision']:.6f}",
                "recall": f"{metrics[prefix + '_recall']:.6f}", "f1": f"{metrics[prefix + '_f1']:.6f}",
                "one_vs_rest_accuracy": f"{metrics['balanced_accuracy']:.6f}",
                "support": sum(metrics["support"]), "value": "",
            })
        writer.writerow({"metric": "overall_accuracy", "precision": "", "recall": "",
                         "f1": "", "one_vs_rest_accuracy": "", "support": "",
                         "value": f"{metrics['accuracy']:.6f}"})
        writer.writerow({"metric": "balanced_accuracy", "precision": "", "recall": "",
                         "f1": "", "one_vs_rest_accuracy": "", "support": "",
                         "value": f"{metrics['balanced_accuracy']:.6f}"})


def write_report_text(path: Path, metrics: Dict[str, Any], class_names: List[str],
                      header: Dict[str, str], n_images: int) -> None:
    lines = [
        "FINAL MODEL EVALUATION", f"Run: {header['run']}", f"Route: {header['route']}",
        f"Split: {header['split']}", f"Images: {n_images}",
        f"Class order (from model.names): {class_names}", "",
        metrics["report_text"], "",
        f"Overall accuracy   : {metrics['accuracy']:.6f}",
        f"Balanced accuracy  : {metrics['balanced_accuracy']:.6f}",
        f"Macro precision    : {metrics['macro_precision']:.6f}",
        f"Macro recall       : {metrics['macro_recall']:.6f}",
        f"Macro F1           : {metrics['macro_f1']:.6f}",
        f"Weighted F1        : {metrics['weighted_f1']:.6f}", "",
        "Per-class one-vs-rest accuracy:",
    ]
    lines += [f"  {name}: {metrics['one_vs_rest_accuracy'][i]:.6f}"
              for i, name in enumerate(class_names)]
    lines += ["", "Confusion matrix (rows=true, cols=predicted):",
              "true\\pred\t" + "\t".join(class_names)]
    for index, name in enumerate(class_names):
        lines.append(name + "\t" + "\t".join(str(v) for v in metrics["confusion_matrix"][index]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(out_dir: Path, metrics: Dict[str, Any], class_names: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.26
    ax.bar(x - width, metrics["precision"], width, label="Precision")
    ax.bar(x, metrics["recall"], width, label="Recall")
    ax.bar(x + width, metrics["f1"], width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("Per-class Precision / Recall / F1")
    fig.tight_layout(); fig.savefig(out_dir / "per_class_prf.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, metrics["one_vs_rest_accuracy"], color="#4c72b0")
    ax.axhline(metrics["accuracy"], color="#c44e52", linestyle="--",
               label=f"Overall accuracy = {metrics['accuracy']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("One-vs-rest accuracy"); ax.legend()
    ax.set_title("Per-class Accuracy with Overall Accuracy")
    fig.tight_layout(); fig.savefig(out_dir / "per_class_accuracy.png", dpi=140); plt.close(fig)

    overall = [metrics["accuracy"], metrics["macro_precision"],
               metrics["macro_recall"], metrics["macro_f1"]]
    labels = ["Accuracy", "Macro P", "Macro R", "Macro F1"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, overall, color=["#55a868", "#4c72b0", "#8172b3", "#c44e52"])
    for bar, value in zip(bars, overall):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.set_title("Overall Metrics")
    fig.tight_layout(); fig.savefig(out_dir / "overall_metrics.png", dpi=140); plt.close(fig)

    for normalized, filename in ((False, "confusion_matrix.png"),
                                 (True, "confusion_matrix_normalized.png")):
        data = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalized else cm
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(data, cmap="Blues")
        ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_yticks(x); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalized else ""))
        for row in range(len(class_names)):
            for col in range(len(class_names)):
                value = data[row, col]
                text = f"{value:.2f}" if normalized else str(int(value))
                ax.text(col, row, text, ha="center", va="center",
                        color="white" if value > data.max() * 0.6 else "black")
        fig.colorbar(image, ax=ax)
        fig.tight_layout(); fig.savefig(out_dir / filename, dpi=140); plt.close(fig)


def write_predictions(path: Path, images: List[Path], y_true: np.ndarray,
                      y_pred: np.ndarray, probs: np.ndarray, class_names: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "true", "predicted", "confidence", "correct"])
        for image, truth, pred, row in zip(images, y_true, y_pred, probs):
            writer.writerow([str(image.relative_to(ROOT)), class_names[truth],
                             class_names[pred], f"{float(row[pred]):.6f}",
                             int(truth == pred)])


def build_comparison(val_rows: List[Dict[str, str]], res_dir: Path,
                     avail_runs: List[Dict[str, str]]) -> Optional[Path]:
    usable = []
    for row in val_rows:
        score = as_float(row.get("macro_f1"))
        if score is None or not row.get("run_name"):
            continue
        meta = meta_for_run(row["run_name"], val_rows, avail_runs)
        usable.append({
            "run_name": row["run_name"], "route": meta["route"],
            "model_name": meta["model_name"], "version": meta["version"], "size": meta["size"],
            "val_accuracy": as_float(row.get("top1_acc")),
            "val_macro_precision": as_float(row.get("macro_precision")),
            "val_macro_recall": as_float(row.get("macro_recall")),
            "val_macro_f1": score,
        })
    if not usable:
        print("[comparison] no finished val rows available; skipping comparison table/chart")
        return None
    usable.sort(key=lambda item: item["val_macro_f1"], reverse=True)
    csv_path = res_dir / "comparison_val.csv"
    fields = ["rank", "run_name", "model_name", "route", "version", "size",
              "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(usable, 1):
            row = {"rank": rank, **item}
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fields})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [f"{item['version']}_{item['size']}\n[{item['route']}]" for item in usable]
    keys = [("val_accuracy", "Accuracy"), ("val_macro_precision", "Macro P"),
            ("val_macro_recall", "Macro R"), ("val_macro_f1", "Macro F1")]
    x = np.arange(len(names)); width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.1), 6))
    for offset, (key, label) in enumerate(keys):
        values = [(item[key] if item[key] is not None else 0.0) for item in usable]
        ax.bar(x + (offset - 1.5) * width, values, width, label=label)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Validation score")
    ax.set_title("Validation comparison (R1 = ImageNet cls, R3 = COCO backbone + Classify head)")
    ax.legend()
    fig.tight_layout()
    chart_path = res_dir / "comparison_val.png"
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)
    print(f"[comparison] wrote {csv_path.name} and {chart_path.name} for {len(usable)} models "
          f"(route tag present on every row/bar)")
    return csv_path


def training_processes() -> List[str]:
    matches = []
    try:
        import psutil
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            if process.pid == os.getpid():
                continue
            if not str(process.info.get("name") or "").lower().startswith("python"):
                continue
            cmdline = " ".join(process.info.get("cmdline") or [])
            if "run_benchmark.py" in cmdline or ("train.py" in cmdline and "--full" in cmdline):
                matches.append(f"pid={process.pid}: {cmdline[:160]}")
        return matches
    except Exception:
        pass
    try:
        import subprocess
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=20).stdout
        for line in output.splitlines():
            if "|" in line:
                pid, command = line.split("|", 1)
                if pid.strip() == str(os.getpid()):
                    continue
                if "run_benchmark.py" in command or ("train.py" in command and "--full" in command):
                    matches.append(f"pid={pid.strip()}: {command[:160]}")
    except Exception:
        pass
    return matches


def holdout_gates_ok(cfg: Dict[str, Any], res_dir: Path,
                     val_rows: List[Dict[str, str]]) -> bool:
    reasons = []
    processes = training_processes()
    if processes:
        reasons.append("training process still alive: " + "; ".join(processes))
    expected = expected_benchmark_runs(res_dir)
    finished = {row.get("run_name") for row in val_rows
                if as_float(row.get("macro_f1")) is not None}
    missing = [record["run_name"] for record in expected if record["run_name"] not in finished]
    if missing:
        reasons.append(f"{len(missing)}/{len(expected)} benchmark models unfinished: {', '.join(missing)}")
    if reasons:
        print("\n[HOLD-OUT REFUSED] " + "\n[HOLD-OUT REFUSED] ".join(reasons))
        print("[HOLD-OUT REFUSED] Re-run with --force only after confirming training is complete.")
        return False
    print(f"[gates] OK: {len(expected)} models finished, no training process detected")
    return True


def run_evaluation(args: argparse.Namespace, cfg: Dict[str, Any], res_dir: Path,
                   val_rows: List[Dict[str, str]], avail_runs: List[Dict[str, str]]) -> None:
    dry = args.dry_run
    run_dir = select_final_run(args, cfg, val_rows, res_dir, avail_runs)
    meta = meta_for_run(run_dir.name, val_rows, avail_runs)
    device: Any = "cpu" if dry else (args.device if args.device is not None else 0)
    imgsz = args.imgsz if args.imgsz is not None else int(cfg["training"]["imgsz"])
    split = "val" if dry else "test"
    split_dir = dataset_root(cfg) / split
    limit = args.max_images if dry else None

    model, class_names, names = load_model_and_names(run_dir, device)
    if dry:
        placeholders = set(class_names) <= set(str(i) for i in range(len(class_names)))
        if placeholders:
            class_names = fallback_class_names(split_dir)
            names = {index: name for index, name in enumerate(class_names)}
            print("[loader] WARNING: checkpoint names were numeric placeholders; "
                  "using sorted dataset class folders as fallback")
    name_to_index = {name: index for index, name in names.items()}
    images, y_true_list = collect_images(split_dir, class_names, name_to_index, limit=limit)
    y_true = np.array(y_true_list, dtype=int)

    print(f"[eval] run={run_dir.name} route={meta['route']} split={split} "
          f"n={len(images)} device={device} imgsz={imgsz}")
    started = time.perf_counter()
    probs = predict_probs(model, images, imgsz, device, args.batch_size)
    y_pred = probs.argmax(axis=1).astype(int)
    elapsed = time.perf_counter() - started
    metrics = compute_metrics(y_true, y_pred, class_names)

    out_dir = res_dir / "final" / f"{run_dir.name}_{'dryrun' if dry else 'test'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_class_csv(out_dir / "per_class_metrics.csv", metrics, class_names)
    header = {"run": run_dir.name, "route": meta["route"], "split": split}
    write_report_text(out_dir / "classification_report.txt", metrics, class_names, header, len(images))
    save_plots(out_dir, metrics, class_names)
    write_predictions(out_dir / "predictions.csv", images, y_true, y_pred, probs, class_names)
    summary = {
        "run": run_dir.name, "route": meta["route"], "split": split,
        "mode": "dry-run" if dry else "hold-out", "device": str(device),
        "n_images": len(images), "class_names": class_names,
        "name_mapping": {str(k): v for k, v in names.items()},
        "imgsz": imgsz, "inference_seconds": round(elapsed, 2),
        **{key: value for key, value in metrics.items()
           if key not in ("confusion_matrix", "report_text")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] accuracy={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"({elapsed:.1f}s) -> {out_dir}")

    append_progress(
        f"- final_report {'DRY-RUN (VAL, CPU)' if dry else 'HOLD-OUT'}: "
        f"`{run_dir.name}` [{meta['route']}] | split={split} | n={len(images)} | "
        f"acc={metrics['accuracy']:.4f} | macroF1={metrics['macro_f1']:.4f} | "
        f"artifacts={out_dir.relative_to(ROOT)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-model report for the cassava YOLO benchmark")
    parser.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--model", default=None, help="Override final run (folder name or path)")
    parser.add_argument("--dry-run", action="store_true", help="VAL subset on CPU; never touch test/GPU")
    parser.add_argument("--force", action="store_true", help="Bypass hold-out safety gates")
    parser.add_argument("--max-images", type=int, default=12, help="Dry-run image cap (default 12)")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Real run device (default 0 / NVIDIA)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    res_dir = results_root(cfg)
    metrics_path = res_dir / "metrics_val.csv"
    val_rows = read_csv_rows(metrics_path)
    avail_runs = expected_benchmark_runs(res_dir)

    print(f"[info] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[info] val rows={len(val_rows)} expected benchmark models={len(avail_runs)} "
          f"metrics_val.csv={metrics_path.exists()}")

    build_comparison(val_rows, res_dir, avail_runs)

    if not args.dry_run and not holdout_gates_ok(cfg, res_dir, val_rows):
        if not args.force:
            return
        print("[force] Safety gates overridden by --force; proceeding with hold-out evaluation")
        append_progress("- final_report WARNING: hold-out evaluation forced with --force "
                        "(gates had failed)")

    if args.dry_run:
        print("[dry-run] VAL split only, device forced to cpu, dataset/test never opened")
    run_evaluation(args, cfg, res_dir, val_rows, avail_runs)


if __name__ == "__main__":
    main()
