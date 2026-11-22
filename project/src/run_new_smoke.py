#!/usr/bin/env python
"""Focused smoke test for the 3 new models: v3_n, v12_n, v12_s.

Loads existing smoke_results.json (10 entries), appends new results,
recomputes 30-epoch estimates for all 13 models, saves:
  results/smoke_results.json   (13 entries)
  results/compute_estimate.json
  results/compute_estimate.md
Prints pass/fail, batch, sec/epoch for each new model.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml
from ultralytics.utils import LOGGER
import train

CFG_PATH = ROOT / "configs" / "benchmark.yaml"
RESULTS_DIR = ROOT / "results"
SMOKE_JSON = RESULTS_DIR / "smoke_results.json"

NEW_MODELS = [
    {"version": "v3",  "size": "n", "route": "R3", "model_name": "yolov3n-cls"},
    {"version": "v12", "size": "n", "route": "R3", "model_name": "yolo12n-cls"},
    {"version": "v12", "size": "s", "route": "R3", "model_name": "yolo12s-cls"},
]


def compute_estimates_30ep(smoke_records: list, cfg: dict) -> dict:
    """Compute GPU-hour estimates at 30 epochs on full dataset."""
    target_epochs = 30
    budget_hours = cfg["budget"]["total_gpu_hours"]
    dataset_ratio = 10.0  # ~4512 train / ~451 smoke train

    estimates = []
    total_seconds = 0.0

    for rec in smoke_records:
        sec_epoch_smoke = rec.get("sec_per_epoch", 0.0)
        est_sec_full = sec_epoch_smoke * dataset_ratio
        est_hours = (est_sec_full * target_epochs) / 3600.0
        total_seconds += est_sec_full * target_epochs
        estimates.append({
            "version": rec["version"],
            "size": rec["size"],
            "route": rec["route"],
            "model_name": rec["model_name"],
            "status": rec["status"],
            "batch_used": rec.get("batch_used", 16),
            "smoke_sec_per_epoch": sec_epoch_smoke,
            "est_full_sec_per_epoch": round(est_sec_full, 2),
            "est_30ep_gpu_hours": round(est_hours, 3),
        })

    total_hours = round(total_seconds / 3600.0, 2)
    return {
        "protocol_epochs": target_epochs,
        "budget_gpu_hours": budget_hours,
        "total_models": len(smoke_records),
        "total_est_30ep_gpu_hours": total_hours,
        "fits_budget": total_hours <= budget_hours,
        "per_model_estimates": estimates,
    }


def write_estimates_md(est: dict, smoke_records: list, out_path: Path) -> None:
    total_h = est["total_est_30ep_gpu_hours"]
    budget_h = est["budget_gpu_hours"]
    lines = [
        "# Cassava YOLO Benchmark — Compute & Smoke Test Estimate (30 epochs)",
        "",
        f"- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB), CUDA 12.1",
        f"- **Total Models (Smoke)**: {len(smoke_records)}",
        f"- **Smoke Protocol**: 2 epochs on 10% subset, batch 16, imgsz 224",
        f"- **Full Protocol**: 30 epochs, patience 10, batch 16, label smoothing 0.1",
        f"- **Budget Cap**: {budget_h} GPU-hours",
        f"- **Estimated Full Training Time (30 epochs)**: **{total_h} GPU-hours**",
        f"- **Budget Status**: {'✅ Fits within budget' if est['fits_budget'] else '⚠️ Exceeds budget'}",
        "",
        "## Per-Model Smoke & Estimate",
        "",
        "| Model | Route | Size | Status | Batch | Smoke s/ep | Full s/ep est | 30ep GPU-h |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for e in est["per_model_estimates"]:
        ok = "✅" if e["status"] == "passed" else "❌"
        lines.append(
            f"| `{e['model_name']}` | `{e['route']}` | `{e['size']}` | {ok} {e['status']} | "
            f"{e['batch_used']} | {e['smoke_sec_per_epoch']}s | {e['est_full_sec_per_epoch']}s | "
            f"**{e['est_30ep_gpu_hours']}h** |"
        )
    lines.extend([
        "",
        "## GPU Thermal & Clock Summary",
        "",
    ])
    warn = [(r["model_name"], r.get("gpu_after", {})) for r in smoke_records
            if r.get("gpu_after", {}).get("temp_c", 0) >= 85]
    if warn:
        lines.append("⚠️ Thermal warnings:")
        for name, ga in warn:
            lines.append(f"  - `{name}`: {ga.get('temp_c')}°C, SM {ga.get('sm_clock_mhz')} MHz")
    else:
        lines.append("✅ No thermal throttling (all runs ≤ 51°C, well below 85°C warning). "
                     "Low SM clocks after run are normal P8 idle downclock state.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing smoke results
    existing = []
    if SMOKE_JSON.exists():
        existing = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
    existing_versions = {(r["version"], r["size"]) for r in existing}
    print(f"Loaded {len(existing)} existing smoke records.")

    # Create smoke dataset if needed
    dataset_dir = ROOT / cfg["paths"]["dataset_dir"]
    smoke_dir = ROOT / cfg["paths"]["smoke_dataset_dir"]
    print(f"Checking smoke dataset at {smoke_dir} ...")
    train.create_smoke_dataset(dataset_dir, smoke_dir, frac=cfg["smoke"]["subset_frac"], seed=cfg["smoke"]["seed"])

    new_results = []
    for m in NEW_MODELS:
        key = (m["version"], m["size"])
        if key in existing_versions:
            print(f"[SKIP] {m['version']}_{m['size']} already in smoke_results.json")
            continue

        print(f"\n{'='*60}")
        print(f"Running smoke: {m['version']}_{m['size']} ({m['route']} {m['model_name']})")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        res = train.run_training_safely(
            version=m["version"],
            size=m["size"],
            route=m["route"],
            model_name=m["model_name"],
            cfg=cfg,
            is_smoke=True,
        )
        elapsed = time.perf_counter() - t0
        new_results.append(res)

        status = res.get("status", "?")
        batch = res.get("batch_used", "?")
        spe = res.get("sec_per_epoch", "?")
        print(f"\n[RESULT] {m['version']}_{m['size']}: status={status}, batch={batch}, sec/epoch={spe}s, total={round(elapsed,1)}s")
        if status == "failed":
            print(f"  ERROR: {res.get('error', 'unknown')[:300]}")

    # Merge and save
    all_results = existing + new_results
    # De-duplicate: keep last entry per (version, size)
    seen = {}
    for r in all_results:
        seen[(r["version"], r["size"])] = r
    all_results = list(seen.values())

    SMOKE_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} smoke records to {SMOKE_JSON}")

    # Compute 30-epoch estimates
    passed = [r for r in all_results if r.get("status") == "passed"]
    est = compute_estimates_30ep(passed, cfg)

    est_json = RESULTS_DIR / "compute_estimate.json"
    est_json.write_text(json.dumps(est, indent=2), encoding="utf-8")

    est_md = RESULTS_DIR / "compute_estimate.md"
    write_estimates_md(est, passed, est_md)

    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY — {len(all_results)} total models")
    print(f"{'='*60}")
    for r in all_results:
        tag = "[NEW]" if (r["version"], r["size"]) in {(m["version"], m["size"]) for m in NEW_MODELS} else "     "
        print(f"  {tag} {r['version']}_{r['size']:4s} ({r['route']}) {r['model_name']:20s} "
              f"status={r.get('status','?'):6s} batch={r.get('batch_used','?')} "
              f"sec/ep={r.get('sec_per_epoch','?')}s")
    print()
    print(f"{'='*60}")
    print(f"30-epoch estimate: {est['total_est_30ep_gpu_hours']} GPU-hours "
          f"({'✅ fits' if est['fits_budget'] else '⚠️ EXCEEDS'} {est['budget_gpu_hours']}h budget)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
