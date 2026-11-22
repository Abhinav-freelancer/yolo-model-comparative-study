#!/usr/bin/env python
"""Cassava YOLO classification training engine.

Supports:
  - R1 native models (yolov8n-cls, yolo11s-cls, etc.)
  - R3 custom models (yolov5n-cls, yolov9t-cls, yolov10s-cls, etc.) with detection-pretrained backbones
  - Class-weighted CrossEntropyLoss with label smoothing (0.1)
  - AdamW optimizer, cosine lr schedule, AMP on
  - Batch size 16 (halves to 8 on CUDA OOM)
  - Resumable runs, GPU temperature and clock monitoring
  - Strict device=0 (NVIDIA discrete GPU only)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force Ultralytics configuration directory inside workspace
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
# Force NVIDIA GPU (device 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import LOGGER, colorstr

import custom_heads


def query_gpu_stats() -> Dict[str, Any]:
    """Query current GPU temperature and SM clock via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            parts = [x.strip() for x in out.splitlines()[0].split(",")]
            temp_c = int(parts[0])
            sm_clock_mhz = int(parts[1])
            max_sm_clock_mhz = int(parts[2])
            mem_used_mib = int(parts[3])
            mem_total_mib = int(parts[4])
            clock_ratio = sm_clock_mhz / max(1, max_sm_clock_mhz)
            return {
                "temp_c": temp_c,
                "sm_clock_mhz": sm_clock_mhz,
                "max_sm_clock_mhz": max_sm_clock_mhz,
                "clock_ratio": round(clock_ratio, 3),
                "mem_used_mib": mem_used_mib,
                "mem_total_mib": mem_total_mib,
                "throttled": temp_c >= 85 or clock_ratio < 0.70
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


class CassavaClassificationLoss:
    """Class-weighted CrossEntropyLoss with label smoothing."""

    def __init__(self, model: nn.Module, label_smoothing: float = 0.1):
        w = getattr(model, "class_weights", None)
        self.class_weights = None if w is None else w.to(next(model.parameters()))
        self.label_smoothing = label_smoothing

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(
            preds,
            batch["cls"],
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return loss, {"loss": loss.detach()}


class CassavaClassificationTrainer(ClassificationTrainer):
    """Subclassed trainer implementing class-weighted CE + label smoothing + R3 injection."""

    def __init__(self, overrides: Optional[Dict] = None, custom_model: Optional[nn.Module] = None,
                 label_smoothing: float = 0.1):
        super().__init__(overrides=overrides)
        self._custom_model = custom_model
        self.label_smoothing = label_smoothing
        self.epoch_times: list[float] = []

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self._custom_model is not None:
            return self._custom_model
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

    def set_model_attributes(self):
        super().set_model_attributes()
        # Attach our custom loss criterion
        self.loss = CassavaClassificationLoss(self.model, label_smoothing=self.label_smoothing)

    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()
        super().on_train_epoch_start()

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        dt = time.perf_counter() - getattr(self, "_epoch_start_time", time.perf_counter())
        self.epoch_times.append(dt)


def create_smoke_dataset(dataset_dir: Path, smoke_dir: Path, frac: float = 0.1, seed: int = 42) -> None:
    """Create a 10% stratified subset of dataset_dir for smoke testing."""
    import random
    rng = random.Random(seed)

    if smoke_dir.exists():
        # verify it has files
        n_existing = sum(1 for _ in smoke_dir.rglob("*.jpg"))
        if n_existing > 100:
            return

    smoke_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_src = dataset_dir / split
        if not split_src.exists():
            continue
        for cls_dir in split_src.iterdir():
            if not cls_dir.is_dir():
                continue
            images = sorted(list(cls_dir.glob("*.jpg")))
            n_select = max(1, int(round(len(images) * frac)))
            selected = rng.sample(images, n_select)
            dst_cls_dir = smoke_dir / split / cls_dir.name
            dst_cls_dir.mkdir(parents=True, exist_ok=True)
            for p in selected:
                dst = dst_cls_dir / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)


def train_single_model(
    version: str,
    size: str,
    route: str,
    model_name: str,
    cfg: Dict[str, Any],
    is_smoke: bool = False,
    batch: int = 16,
    workers: int = 2,
) -> Dict[str, Any]:
    """Train a single model with automatic OOM retry (halves batch 16 -> 8)."""
    dataset_dir = ROOT / (cfg["paths"]["smoke_dataset_dir"] if is_smoke else cfg["paths"]["dataset_dir"])
    runs_dir = ROOT / (cfg["paths"]["smoke_runs_dir"] if is_smoke else cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    init_name = "pretrained"
    run_seed = cfg["training"]["seed"]
    run_name = f"{version}_{size}_{init_name}_seed{run_seed}"
    run_dir = runs_dir / run_name

    # Check resumability
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    target_epochs = cfg["smoke"]["epochs"] if is_smoke else cfg["training"]["epochs"]
    if run_dir.exists() and best_pt.exists() and results_csv.exists():
        try:
            with open(results_csv, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                # line 0 is header, lines 1.. are epochs
                if len(lines) - 1 >= target_epochs:
                    LOGGER.info(f"Skipping completed run {run_name} ({len(lines)-1} epochs done)")
                    return {
                        "run_name": run_name, "version": version, "size": size, "route": route,
                        "status": "completed", "epochs_done": len(lines) - 1, "run_dir": str(run_dir),
                        "batch": batch, "sec_per_epoch": 0.0
                    }
        except Exception:
            pass

    # Resume interrupted run from last.pt if available
    resume_from_last = False
    if run_dir.exists() and last_pt.exists() and results_csv.exists():
        try:
            with open(results_csv, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            epochs_done = len(lines) - 1  # header doesn't count
            if 0 < epochs_done < target_epochs:
                LOGGER.info(f"Resuming interrupted run {run_name} from last.pt ({epochs_done}/{target_epochs} epochs done)")
                resume_from_last = True
        except Exception:
            pass

    # Model instantiation
    custom_model = None
    model_arg = None
    if route == "R1":
        # Native pretrained model weights
        model_arg = f"{model_name}.pt" if not model_name.endswith(".pt") else model_name
    elif route == "R3":
        # Custom detection backbone + Classify head
        m, info = custom_heads.make_r3_model(version, size, nc=cfg["nc"], pretrained=True)
        if m is None:
            raise RuntimeError(f"R3 model build failed for {version}_{size}: {info.get('error', 'unknown error')}")
        custom_model = m
        model_arg = info["yaml"]
    else:
        raise ValueError(f"Unsupported route {route} for training")

    patience = cfg["smoke"]["patience"] if is_smoke else cfg["training"]["patience"]
    label_smoothing = cfg["training"]["label_smoothing"]

    augs = cfg["training"]["augs"]
    overrides = {
        "data": str(dataset_dir),
        "model": model_arg,
        "epochs": target_epochs,
        "patience": patience,
        "batch": batch,
        "imgsz": cfg["training"]["imgsz"],
        "workers": workers,
        "cache": cfg["training"]["cache"],
        "amp": cfg["training"]["amp"],
        "optimizer": cfg["training"]["optimizer"],
        "lr0": cfg["training"]["lr0"],
        "lrf": cfg["training"]["lrf"],
        "cos_lr": cfg["training"]["cos_lr"],
        "warmup_epochs": 0 if is_smoke else cfg["training"]["warmup_epochs"],
        "weight_decay": cfg["training"]["weight_decay"],
        "cls_pw": cfg["training"]["cls_pw"],
        "seed": run_seed,
        "device": 0,
        "project": str(runs_dir),
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "save_period": cfg["training"]["save_period"],
        "hsv_h": augs.get("hsv_h", 0.015),
        "hsv_s": augs.get("hsv_s", 0.4),
        "hsv_v": augs.get("hsv_v", 0.4),
        "degrees": augs.get("degrees", 15.0),
        "translate": augs.get("translate", 0.1),
        "scale": augs.get("scale", 0.5),
        "fliplr": augs.get("fliplr", 0.5),
        "flipud": augs.get("flipud", 0.5),
        "mixup": augs.get("mixup", 0.0),
        "cutmix": augs.get("cutmix", 0.0),
        "auto_augment": augs.get("auto_augment", None),
        "erasing": augs.get("erasing", 0.0),
        "verbose": True,
    }

    # If resuming from last.pt, override model to point at last.pt and set resume=True
    if resume_from_last:
        model_arg = str(last_pt)
        custom_model = None  # let Ultralytics load from checkpoint
        overrides["resume"] = True
        overrides["model"] = model_arg
        LOGGER.info(f"Resume mode: model={model_arg}")

    gpu_before = query_gpu_stats()
    start_time = time.perf_counter()

    trainer = CassavaClassificationTrainer(overrides=overrides, custom_model=custom_model,
                                           label_smoothing=label_smoothing)
    trainer.train()
    total_time = time.perf_counter() - start_time
    gpu_after = query_gpu_stats()

    epoch_times = trainer.epoch_times
    sec_per_epoch = (sum(epoch_times) / max(1, len(epoch_times))) if epoch_times else (total_time / max(1, target_epochs))

    return {
        "run_name": run_name,
        "version": version,
        "size": size,
        "route": route,
        "model_name": model_name,
        "status": "passed",
        "epochs_done": target_epochs,
        "batch_used": batch,
        "total_seconds": round(total_time, 2),
        "sec_per_epoch": round(sec_per_epoch, 2),
        "run_dir": str(run_dir),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def run_training_safely(
    version: str,
    size: str,
    route: str,
    model_name: str,
    cfg: Dict[str, Any],
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """Train with try/except, OOM retry (16 -> 8), and workers fallback (2 -> 0)."""
    batch = cfg["training"]["batch"]
    workers = cfg["training"]["workers"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    failures_csv = results_dir / "failures.csv"

    # Attempt 1: primary configuration
    try:
        LOGGER.info(f"\n{'='*70}\nStarting {version}_{size} ({route} {model_name}) | batch={batch}, workers={workers}\n{'='*70}")
        return train_single_model(version, size, route, model_name, cfg, is_smoke=is_smoke, batch=batch, workers=workers)
    except Exception as e:
        err_msg = str(e)
        LOGGER.warning(f"Attempt 1 failed for {version}_{size}: {err_msg}")
        torch.cuda.empty_cache()

        # Check if error is multiprocessing / worker related
        if "multiprocessing" in err_msg.lower() or "broken pipe" in err_msg.lower() or "pickle" in err_msg.lower():
            LOGGER.info(f"Retrying with workers=0 ...")
            try:
                return train_single_model(version, size, route, model_name, cfg, is_smoke=is_smoke, batch=batch, workers=0)
            except Exception as e2:
                err_msg = str(e2)
                torch.cuda.empty_cache()

        # Check if error is CUDA out of memory -> retry with halved batch
        if "out of memory" in err_msg.lower() or "cuda oom" in err_msg.lower() or "allocate" in err_msg.lower():
            half_batch = batch // 2
            LOGGER.warning(f"CUDA OOM detected! Retrying with halved batch {batch} -> {half_batch} ...")
            try:
                return train_single_model(version, size, route, model_name, cfg, is_smoke=is_smoke, batch=half_batch, workers=0)
            except Exception as e3:
                err_msg = str(e3)

        # Log failure
        LOGGER.error(f"FATAL: All attempts failed for {version}_{size}: {err_msg}")
        file_exists = failures_csv.exists()
        with open(failures_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "version", "size", "route", "model_name", "error"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": version,
                "size": size,
                "route": route,
                "model_name": model_name,
                "error": err_msg.replace("\n", " ")[:300]
            })
        return {
            "run_name": f"{version}_{size}_pretrained_seed{cfg['training']['seed']}",
            "version": version,
            "size": size,
            "route": route,
            "model_name": model_name,
            "status": "failed",
            "error": err_msg,
            "batch_used": batch,
            "sec_per_epoch": 0.0,
        }
