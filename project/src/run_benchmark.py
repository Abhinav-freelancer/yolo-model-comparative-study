#!/usr/bin/env python
"""Benchmark runner: orchestrates sequential full training runs.

Usage:
  python run_benchmark.py --smoke     # 2-epoch smoke (use run_new_smoke.py for new models)
  python run_benchmark.py --full      # 30-epoch full training, sequential, resumable

Outputs (--full):
  results/metrics_val.csv         (one row per model, appended after each run)
  results/full_run_summary.md     (updated after each model)
  PROGRESS.md                     (updated after each model)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml
from ultralytics.utils import LOGGER

import train
import evaluate as evaluator

CLASSES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]
LOCK_PATH = ROOT / "results" / "benchmark.lock"


# ---------------------------------------------------------------------------
# Duplicate-run lock
# ---------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            import subprocess
            output = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            return output == str(pid)
        except Exception:
            return False


def acquire_benchmark_lock(results_dir: Path) -> Optional[Path]:
    lock = results_dir / "benchmark.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except Exception:
            pid = -1
        if pid_alive(pid):
            print(f"REFUSING TO START: {lock} is held by live PID {pid}. "
                  f"Only one benchmark may run at a time.", flush=True)
            return None
        print(f"[lock] stale lock (PID {pid} not alive); overwriting", flush=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(f"[lock] acquired {lock} with PID {os.getpid()}", flush=True)
    return lock


def release_benchmark_lock(lock: Optional[Path]) -> None:
    try:
        if lock and lock.exists():
            pid = int(lock.read_text(encoding="utf-8").strip())
            if pid == os.getpid():
                lock.unlink()
                print(f"[lock] released {lock}", flush=True)
    except Exception as exc:
        print(f"[WARN] could not release lock: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def get_target_models(available_runs_path: Path) -> List[Dict[str, Any]]:
    """Return ordered list of models for the full benchmark.

    Order: R1 first (v8_n, v8_s, v11_n, v11_s), then R3 alphabetically.
    Excludes v26-bonus and v3_std (size="std").
    """
    with open(available_runs_path, encoding="utf-8") as f:
        runs = json.load(f)

    r1_order = ["v8_n", "v8_s", "v11_n", "v11_s"]
    r1 = []
    r3 = []
    for r in runs:
        if r.get("version") == "v26-bonus":
            continue
        if r.get("size") == "std":
            continue
        if r.get("size") not in ("n", "s"):
            continue
        key = f"{r['version']}_{r['size']}"
        if r.get("route") == "R1":
            r1.append(r)
        else:
            r3.append(r)

    # Sort R1 by preferred order, R3 by version/size
    def r1_rank(r: Dict) -> int:
        key = f"{r['version']}_{r['size']}"
        return r1_order.index(key) if key in r1_order else 99

    def r3_rank(r: Dict) -> tuple:
        # v3 < v5 < v9 < v10 < v12 (by version number, n before s)
        v = r["version"].lstrip("v")
        return (int(v) if v.isdigit() else 99, r["size"])

    r1.sort(key=r1_rank)
    r3.sort(key=r3_rank)
    return r1 + r3


# ---------------------------------------------------------------------------
# PROGRESS.md helpers
# ---------------------------------------------------------------------------

PROGRESS_PATH = ROOT / "PROGRESS.md"

def load_progress_state() -> Dict[str, Any]:
    """Load the structured state from PROGRESS.md (or return empty dict)."""
    if not PROGRESS_PATH.exists():
        return {}
    # We store a JSON block at the bottom of PROGRESS.md for machine reading
    text = PROGRESS_PATH.read_text(encoding="utf-8")
    marker = "<!-- STATE_JSON:"
    end_marker = ":STATE_JSON -->"
    if marker in text and end_marker in text:
        start = text.index(marker) + len(marker)
        end = text.index(end_marker)
        try:
            return json.loads(text[start:end].strip())
        except Exception:
            return {}
    return {}


def update_progress_md(
    models_done: List[Dict],
    models_pending: List[Dict],
    current_model: Optional[Dict] = None,
) -> None:
    """Rewrite PROGRESS.md with current status."""
    lines = [
        "# PROGRESS — Cassava YOLO Benchmark",
        "",
        f"**Last updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Training Status",
        "",
        "| # | Model | Version | Size | Route | Status | Val Top-1 | Val Macro-F1 | Epochs | Best Ep | Min |",
        "| --: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for i, m in enumerate(models_done, 1):
        status = m.get("train_status", "done")
        status_icon = "✅" if status not in ("failed", "skipped") else "❌"
        lines.append(
            f"| {i} | `{m['model_name']}` | {m['version']} | {m['size']} | "
            f"`{m['route']}` | {status_icon} {status} | "
            f"{m.get('val_top1', '?')} | {m.get('val_macro_f1', '?')} | "
            f"{m.get('epochs_run', '?')} | {m.get('best_epoch', '?')} | {m.get('minutes', '?')} |"
        )

    if current_model:
        i = len(models_done) + 1
        lines.append(
            f"| {i} | `{current_model['model_name']}` | {current_model['version']} | "
            f"{current_model['size']} | `{current_model['route']}` | ⏳ running | - | - | - | - | - |"
        )

    for i, m in enumerate(models_pending, len(models_done) + (1 if current_model else 0) + 1):
        lines.append(
            f"| {i} | `{m['model_name']}` | {m['version']} | {m['size']} | "
            f"`{m['route']}` | ⏳ pending | - | - | - | - | - |"
        )

    lines.extend([
        "",
        "## Notes",
        "- R1 models: ImageNet-pretrained classifiers (native Ultralytics *-cls.pt)",
        "- R3 models: COCO detection backbones + new Classify head (custom R3 route)",
        "- R1 vs R3 performance gaps are NOT purely architectural (pretraining domain differs)",
        "- Protocol: 30 epochs, patience 10, batch 16, imgsz 224, class-weighted CE, label smoothing 0.1, seed 0",
        "- Hold-out (565 images) evaluated ONCE at the end for final models only",
        "",
    ])

    state = {
        "models_done": models_done,
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    lines.append(f"<!-- STATE_JSON:{json.dumps(state)}:STATE_JSON -->")

    PROGRESS_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

def patch_route_in_metrics(out_csv: Path, run_name: str, route: str) -> bool:
    """Set the route cell of a metrics row without rewriting any other value."""
    if not out_csv.exists() or not route or route in ("?", "unknown"):
        return False
    try:
        with open(out_csv, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames or "route" not in fieldnames:
            return False
        changed = False
        for row in rows:
            current = str(row.get("route") or "").strip()
            if row.get("run_name") == run_name and current in ("", "?", "unknown"):
                row["route"] = route
                changed = True
        if not changed:
            return False
        temp_csv = out_csv.with_name(out_csv.name + ".tmp")
        with open(temp_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temp_csv.replace(out_csv)
        return True
    except Exception as exc:
        print(f"  [WARN] Could not patch route for {run_name}: {exc}")
        return False


def eval_after_training(
    run_dir: Path,
    model_meta: Dict[str, Any],
    cfg: Dict[str, Any],
    t_start: float,
    t_end: float,
) -> Dict[str, Any]:
    """Run val evaluation and return structured result dict."""
    minutes = round((t_end - t_start) / 60.0, 1)

    # Try evaluation
    out_csv = ROOT / cfg["paths"]["results_dir"] / "metrics_val.csv"
    try:
        summary_line = evaluator.evaluate_run(
            run_dir=run_dir,
            split="val",
            cfg=cfg,
            out_csv=out_csv,
            imgsz=cfg["training"]["imgsz"],
        )
        if patch_route_in_metrics(out_csv, run_dir.name, model_meta["route"]):
            print(f"  Patched route={model_meta['route']} in {out_csv.name} for {run_dir.name}")
    except Exception as e:
        summary_line = f"[EVAL ERROR] {e}"
        print(f"  Evaluation failed: {e}")

    # Read back last row of metrics_val.csv for structured data
    val_top1 = "?"
    val_macro_f1 = "?"
    epochs_run = "?"
    best_epoch = "?"
    if out_csv.exists():
        try:
            with open(out_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                val_top1 = last.get("top1_acc", "?")
                val_macro_f1 = last.get("macro_f1", "?")
                epochs_run = last.get("epochs_run", "?")
                best_epoch = last.get("best_epoch", "?")
        except Exception:
            pass

    return {
        "version": model_meta["version"],
        "size": model_meta["size"],
        "route": model_meta["route"],
        "model_name": model_meta["model_name"],
        "train_status": "done",
        "val_top1": val_top1,
        "val_macro_f1": val_macro_f1,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "minutes": minutes,
        "summary_line": summary_line,
    }


# ---------------------------------------------------------------------------
# GPU-hour re-estimate
# ---------------------------------------------------------------------------

def reestimate_total(first_result: Dict, remaining_models: List, target_epochs: int) -> float:
    """Re-estimate total GPU-hours using real sec/epoch from first completed model."""
    dataset_ratio = 10.0  # smoke vs full dataset
    smoke_json = ROOT / "results" / "smoke_results.json"
    smoke_map = {}
    if smoke_json.exists():
        try:
            recs = json.loads(smoke_json.read_text(encoding="utf-8"))
            smoke_map = {(r["version"], r["size"]): r.get("sec_per_epoch", 0) for r in recs}
        except Exception:
            pass

    # Actual sec/epoch for first model
    first_sec_epoch = first_result.get("sec_per_epoch", 0)
    # For remaining models, scale from smoke estimates or use first_sec_epoch as proxy
    total_seconds = first_sec_epoch * target_epochs
    for m in remaining_models:
        smoke_spe = smoke_map.get((m["version"], m["size"]), 0)
        if smoke_spe > 0:
            total_seconds += smoke_spe * dataset_ratio * target_epochs
        else:
            total_seconds += first_sec_epoch * target_epochs  # conservative proxy
    return round(total_seconds / 3600.0, 2)


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(cfg: Dict[str, Any]) -> None:
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    available_runs_path = results_dir / "available_runs.json"

    if not available_runs_path.exists():
        print(f"ERROR: {available_runs_path} not found. Run check_availability.py first.")
        sys.exit(1)

    models_to_run = get_target_models(available_runs_path)
    target_epochs = cfg["training"]["epochs"]
    budget_hours = cfg["budget"]["total_gpu_hours"]

    print(f"\n{'='*70}")
    print(f"FULL BENCHMARK — {len(models_to_run)} models | {target_epochs} epochs | budget {budget_hours}h")
    print(f"{'='*70}")
    for i, m in enumerate(models_to_run, 1):
        print(f"  {i:2d}. [{m['route']}] {m['version']}_{m['size']} -> {m['model_name']}")
    print()

    # Load progress state to support resumability
    state = load_progress_state()
    done_keys = {(r["version"], r["size"]) for r in state.get("models_done", [])}
    models_done = state.get("models_done", [])

    metrics_csv = results_dir / "metrics_val.csv"
    metric_rows = {}
    if metrics_csv.exists():
        try:
            with open(metrics_csv, newline="", encoding="utf-8") as handle:
                metric_rows = {row.get("run_name"): row for row in csv.DictReader(handle)
                               if row.get("run_name")}
        except Exception as exc:
            print(f"[WARN] could not read {metrics_csv}: {exc}", flush=True)

    runs_dir = ROOT / cfg["paths"]["runs_dir"]
    run_seed = cfg["training"]["seed"]

    for idx, m in enumerate(models_to_run):
        key = (m["version"], m["size"])
        run_name = f"{m['version']}_{m['size']}_pretrained_seed{run_seed}"
        if key in done_keys or run_name in metric_rows:
            print(f"[SKIP] {m['version']}_{m['size']} already has a metrics_val.csv row / "
                  f"PROGRESS.md entry; no duplicate row will be appended", flush=True)
            if key not in done_keys and run_name in metric_rows:
                row = metric_rows[run_name]
                models_done.append({
                    "version": m["version"], "size": m["size"], "route": m["route"],
                    "model_name": m["model_name"], "train_status": "done",
                    "val_top1": row.get("top1_acc", "?"),
                    "val_macro_f1": row.get("macro_f1", "?"),
                    "epochs_run": row.get("epochs_run", "?"),
                    "best_epoch": row.get("best_epoch", "?"), "minutes": "?",
                })
                done_keys.add(key)
            continue

        remaining = models_to_run[idx + 1:]
        current = m
        try:
            update_progress_md(models_done, remaining, current_model=current)
        except Exception as e:
            print(f"[WARN] Failed to update PROGRESS.md: {e}")
            fallback_path = ROOT / "results" / "progress_fallback.log"
            with open(fallback_path, "a", encoding="utf-8") as f:
                f.write(f"Failed update at model {current['model_name']}: {e}\n")

        run_dir = runs_dir / run_name
        had_partial = (run_dir / "weights" / "last.pt").exists() and (run_dir / "results.csv").exists()
        print(f"\n===== START {m['version']}_{m['size']} [{m['route']}] "
              f"{m['model_name']} ({'resume' if had_partial else 'fresh'}) =====", flush=True)

        t_start = time.perf_counter()
        result = train.run_training_safely(
            version=m["version"],
            size=m["size"],
            route=m["route"],
            model_name=m["model_name"],
            cfg=cfg,
            is_smoke=False,
        )

        if result.get("status") == "failed" and had_partial and run_dir.exists():
            renamed = runs_dir / f"{run_name}_partial_{time.strftime('%Y%m%d_%H%M%S')}"
            try:
                run_dir.rename(renamed)
                print(f"[resume-fallback] renamed {run_name} -> {renamed.name}; "
                      f"retraining from scratch", flush=True)
                t_start = time.perf_counter()
                result = train.run_training_safely(
                    version=m["version"], size=m["size"], route=m["route"],
                    model_name=m["model_name"], cfg=cfg, is_smoke=False,
                )
            except Exception as exc:
                print(f"[resume-fallback] rename/retry failed: {exc}", flush=True)
        t_end = time.perf_counter()

        sec_per_epoch = result.get("sec_per_epoch", 0)
        total_min = round((t_end - t_start) / 60.0, 1)

        print(f"\n--- Training done: {m['version']}_{m['size']} | "
              f"status={result['status']} | {sec_per_epoch}s/ep | {total_min} min ---")

        # --- First-model re-estimate ---
        if idx == 0 and result["status"] not in ("failed",):
            reest = reestimate_total(result, remaining, target_epochs)
            print(f"\n{'='*70}")
            print(f"⏱  FIRST MODEL DONE — Actual {sec_per_epoch}s/epoch on full dataset")
            print(f"⏱  Re-estimated total ({len(models_to_run)} models × {target_epochs} epochs): {reest} GPU-hours")
            if reest > budget_hours:
                print(f"⚠️  WARNING: Projection {reest}h EXCEEDS {budget_hours}h budget!")
                print(f"⚠️  Continuing as per approved protocol (30 epochs, all models).")
            else:
                print(f"✅ Projection {reest}h fits within {budget_hours}h budget.")
            print(f"{'='*70}")

        # --- Val evaluation ---
        done_entry = {
            "version": m["version"],
            "size": m["size"],
            "route": m["route"],
            "model_name": m["model_name"],
            "train_status": result["status"],
            "val_top1": "?",
            "val_macro_f1": "?",
            "epochs_run": result.get("epochs_done", "?"),
            "best_epoch": "?",
            "minutes": round((t_end - t_start) / 60.0, 1),
        }

        if result["status"] not in ("failed",) and run_dir.exists() and (run_dir / "weights" / "best.pt").exists():
            print(f"\n  Running val evaluation for {run_dir.name} ...")
            eval_result = eval_after_training(run_dir, m, cfg, t_start, t_end)
            done_entry.update({
                "val_top1": eval_result["val_top1"],
                "val_macro_f1": eval_result["val_macro_f1"],
                "epochs_run": eval_result["epochs_run"],
                "best_epoch": eval_result["best_epoch"],
            })
            summary_line = eval_result["summary_line"]
        else:
            summary_line = f"[SKIP EVAL] {run_name} training {result['status']}"

        models_done.append(done_entry)
        done_keys.add(key)

        # One-line summary printout
        print(f"\n{'*'*70}")
        print(f"[DONE] {m['model_name']} | {m['route']} | "
              f"val_top1={done_entry['val_top1']} | val_macro_F1={done_entry['val_macro_f1']} | "
              f"epochs={done_entry['epochs_run']} | best_ep={done_entry['best_epoch']} | "
              f"{done_entry['minutes']}min", flush=True)
        print(f"===== END {m['version']}_{m['size']} =====\n{'*'*70}\n", flush=True)

        # Update PROGRESS.md
        remaining_pending = [mm for mm in models_to_run
                              if (mm["version"], mm["size"]) not in done_keys]
        try:
            update_progress_md(models_done, remaining_pending, current_model=None)
        except Exception as e:
            print(f"[WARN] Failed to update PROGRESS.md: {e}")
            fallback_path = ROOT / "results" / "progress_fallback.log"
            with open(fallback_path, "a", encoding="utf-8") as f:
                f.write(f"Failed update after model {models_done[-1]['model_name'] if models_done else 'none'}: {e}\n")

    # --- Final summary ---
    print(f"\n{'='*70}")
    print("FULL BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'Route':<5} {'Val Top-1':>10} {'Val F1':>10} {'Epochs':>7} {'Min':>7}")
    print("-" * 70)
    for d in models_done:
        print(f"{d['model_name']:<22} {d['route']:<5} {str(d['val_top1']):>10} "
              f"{str(d['val_macro_f1']):>10} {str(d['epochs_run']):>7} {str(d['minutes']):>7}")
    print(f"{'='*70}")
    print(f"\nResults saved to: {ROOT / 'results' / 'metrics_val.csv'}")
    try:
        update_progress_md(models_done, [], current_model=None)
    except Exception as e:
        print(f"[WARN] Failed to update PROGRESS.md at end: {e}")
        fallback_path = ROOT / "results" / "progress_fallback.log"
        with open(fallback_path, "a", encoding="utf-8") as f:
            f.write(f"Failed final update after benchmark: {e}\n")


# ---------------------------------------------------------------------------
# Smoke mode (legacy, for the original 10 models only — use run_new_smoke.py for new)
# ---------------------------------------------------------------------------

def run_smoke_pipeline(cfg: Dict[str, Any]) -> None:
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    available_runs_path = results_dir / "available_runs.json"
    if not available_runs_path.exists():
        print(f"ERROR: {available_runs_path} not found.")
        sys.exit(1)

    models_to_run = get_target_models(available_runs_path)
    dataset_dir = ROOT / cfg["paths"]["dataset_dir"]
    smoke_dir = ROOT / cfg["paths"]["smoke_dataset_dir"]
    print(f"Checking/creating smoke dataset at {smoke_dir} ...")
    train.create_smoke_dataset(dataset_dir, smoke_dir, frac=cfg["smoke"]["subset_frac"], seed=cfg["smoke"]["seed"])

    smoke_records = []
    for idx, m in enumerate(models_to_run, 1):
        print(f"\n>>> Smoke ({idx}/{len(models_to_run)}): {m['version']}_{m['size']} ({m['route']} {m['model_name']})")
        res = train.run_training_safely(
            version=m["version"],
            size=m["size"],
            route=m["route"],
            model_name=m["model_name"],
            cfg=cfg,
            is_smoke=True,
        )
        smoke_records.append(res)
        print(f"--- {m['version']}_{m['size']}: status={res['status']}, "
              f"batch={res.get('batch_used')}, sec/ep={res.get('sec_per_epoch')}s ---")

    smoke_json = results_dir / "smoke_results.json"
    smoke_json.write_text(json.dumps(smoke_records, indent=2), encoding="utf-8")
    print(f"\nSaved smoke results to {smoke_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Cassava YOLO benchmark orchestrator")
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    if not args.smoke and not args.full:
        print("Please specify --smoke or --full")
        sys.exit(1)

    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_benchmark_lock(results_dir)
    if lock is None:
        sys.exit(2)
    try:
        if args.smoke:
            run_smoke_pipeline(cfg)
        elif args.full:
            run_full_pipeline(cfg)
    finally:
        release_benchmark_lock(lock)


if __name__ == "__main__":
    main()
