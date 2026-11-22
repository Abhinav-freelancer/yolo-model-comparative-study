#!/usr/bin/env python
"""Probe what Ultralytics 8.4.215 actually offers for YOLOv3..v12 classification.

Do NOT assume from memory: for every (version, size) we actually try to load
weights/configs. Routes (labeled in ALL tables/plots):
  R1 native   : Ultralytics *-cls.pt (ImageNet-pretrained) loaded successfully.
  R2 official : official-repo classification support (NOT used here - would need
                cloning external repos and risks the pinned CUDA torch env; any
                version not served by R1/R3 is documented, not silently dropped).
  R3 custom   : detection backbone (from ultralytics detection yaml) + standard
                Classify head (conv1x1->GAP->dropout->linear(5)), detection-
                pretrained backbone via intersect-load. See custom_heads.py.
  R4 skip     : infeasible; stated reason. NEVER fabricate results.

Outputs:
  results/availability_matrix.csv   (version,size,model_name,route,status,...)
  results/available_runs.json       (trainable runs consumed by run_benchmark.py)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SIZES = ["n", "s"]
VERSIONS = ["v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"]
BONUS = ["yolo26n-cls", "yolo26s-cls"]  # recorded, not part of v3..v12 comparison

# name candidates tried per version for R1 (all are actually attempted)
R1_CANDIDATES: dict[str, list[str]] = {
    "v3": [f"yolov3{s}-cls" for s in SIZES] + ["yolov3u-cls"],
    "v4": [f"yolov4{s}-cls" for s in SIZES],
    "v5": [f"yolov5{s}u-cls" for s in SIZES] + [f"yolov5{s}-cls" for s in SIZES],
    "v6": [f"yolov6{s}-cls" for s in SIZES],
    "v7": [f"yolov7{s}-cls" for s in SIZES],
    "v8": [f"yolov8{s}-cls" for s in SIZES],
    "v9": [f"yolov9{s}-cls" for s in SIZES],
    "v10": [f"yolov10{s}-cls" for s in SIZES],
    "v11": [f"yolo11{s}-cls" for s in SIZES],
    "v12": [f"yolo12{s}-cls" for s in SIZES],
}

R4_REASONS = {
    "v4": "no yolov4 code/weights in ultralytics 8.4.215 (no *-cls.pt, no detection yaml); "
          "official AlexeyAB repo has no integrated classifier; external clone would risk the pinned CUDA env",
    "v6": "ultralytics hosts no yolov6*.pt detection weights; random initialization rejected per benchmark protocol",
    "v7": "no yolov7 code/weights in ultralytics 8.4.215 (no *-cls.pt, no detection yaml); "
          "official WongKinYiu repo is detection-only (yolov7-e6e cls variant never released); "
          "external clone would risk the pinned CUDA env",
}


def _weights_file_mb(name: str) -> tuple[str, float]:
    """Ultralytics downloads weights into the CWD; return (path, MB) if present."""
    for cand in (Path(name + ".pt"), ROOT / (name + ".pt")):
        if cand.exists():
            return str(cand), round(cand.stat().st_size / 1e6, 2)
    return "", 0.0


def probe_r1(name: str) -> dict:
    """Try to load a native ultralytics *-cls model (downloads pretrained weights)."""
    row = dict(model_name=name, route="R1", status="failed", params_M="",
               weights_file="", weights_MB="", notes="")
    try:
        from ultralytics import YOLO
        m = YOLO(name + ".pt")
        if getattr(m, "task", "") != "classify":
            row["notes"] = f"loaded but task={getattr(m, 'task', '?')} (not classify)"
            return row
        row["status"] = "available"
        row["params_M"] = round(sum(p.numel() for p in m.model.parameters()) / 1e6, 3)
        wf, mb = _weights_file_mb(name)
        row["weights_file"], row["weights_MB"] = wf, mb
        row["notes"] = "native ultralytics *-cls.pt, ImageNet-pretrained"
    except Exception as e:
        row["notes"] = str(e)[:220].replace("\n", " ")
    return row


def main() -> None:
    rows: list[dict] = []

    # ---------- R1 native probes (real weight downloads) ----------
    for ver in VERSIONS:
        seen = set()
        for cand in R1_CANDIDATES[ver]:
            size = size_of_candidate(cand)
            r = probe_r1(cand)
            r.update(version=ver, size=size)
            rows.append(r)
            if r["status"] == "available":
                seen.add(size)
        print(f"[R1] {ver}: available sizes = {sorted(seen)}", flush=True)

    # ---------- bonus probes (recorded, not trained unless approved) ----------
    for name in BONUS:
        r = probe_r1(name)
        r.update(version="v26-bonus", size=name.replace("yolo", "").replace("-cls", ""))
        rows.append(r)

    # ---------- R3 custom probes ----------
    import custom_heads
    r3_rows = []
    for ver, specs in custom_heads.R3_MAP.items():
        # skip R3 entirely for sizes where R1 already works
        r1_ok = {(r["version"], r["size"]) for r in rows if r["route"] == "R1" and r["status"] == "available"}
        for spec in specs:
            size = spec["size"]
            if (ver, size) in r1_ok:
                continue
            key = f"{ver}_{size}"
            try:
                model, info = custom_heads.make_r3_model(ver, size, pretrained=True)
                row = dict(version=ver, size=size,
                           model_name=Path(info.get("yaml", "")).stem if info.get("yaml") else key,
                           route="R3", status="available" if model is not None else "failed",
                           params_M=info.get("params_M", ""),
                           weights_file=info.get("det_weights", ""), weights_MB="",
                           notes=str(info.get("note", ""))[:200] +
                                 (f" | init: {info.get('init', '')}" if model is not None else
                                  f" | {info.get('error', '')}"[:200]))
                if model is not None:
                    bl = info.get("backbone_load", {})
                    row["notes"] = (f"det-backbone intersect {bl.get('frac', '?')} "
                                    f"({bl.get('n_loaded', '?')}/{bl.get('n_total', '?')} keys); "
                                    f"{row['notes']}")[:300]
                del model
            except Exception as e:
                row = dict(version=ver, size=size, model_name=key, route="R3", status="failed",
                           params_M="", weights_file="", weights_MB="",
                           notes=str(e)[:220].replace("\n", " "))
            rows.append(row)
            r3_rows.append(row)
            print(f"[R3] {key}: {row['status']} - {row['notes'][:110]}", flush=True)

    # ---------- R4 skip rows ----------
    for ver in VERSIONS:
        has_any = any(r["version"] == ver and r["status"] == "available" for r in rows)
        if not has_any:
            for size in SIZES:
                if any(r["version"] == ver and r["size"] == size for r in rows):
                    continue
                rows.append(dict(version=ver, size=size, model_name=f"yolov{ver[1:]}{size}",
                                 route="R4", status="skipped", params_M="", weights_file="",
                                 weights_MB="", notes=R4_REASONS.get(ver, "infeasible")))

    # ---------- write outputs ----------
    cols = ["version", "size", "model_name", "route", "status", "params_M",
            "weights_file", "weights_MB", "notes"]
    rows.sort(key=lambda r: (VERSIONS.index(r["version"]) if r["version"] in VERSIONS else 99,
                             SIZES.index(r["size"]) if r["size"] in SIZES else 9,
                             r["route"]))
    with open(RESULTS / "availability_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # trainable runs for the orchestrator (pretrained only, per approved scope)
    runs = [dict(version=r["version"], size=r["size"], route=r["route"], model_name=r["model_name"],
                 run_name=f"{r['version']}_{r['size']}_pretrained_seed0")
            for r in rows if r["status"] == "available" and r["route"] in {"R1", "R3"}]
    (RESULTS / "available_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    print(f"\nWrote {RESULTS / 'availability_matrix.csv'} ({len(rows)} rows)")
    print(f"Wrote {RESULTS / 'available_runs.json'} ({len(runs)} trainable runs)")
    for r in runs:
        print(f"  RUN {r['run_name']:<32} {r['route']} {r['model_name']}")


def size_of_candidate(cand: str) -> str:
    """'yolov8n-cls' -> 'n'; 'yolov3u-cls' -> 'std'; fallback 'n'."""
    from re import match
    m = match(r"yolo[v]?\d+([nslmx])", cand)
    if m:
        return m.group(1)
    return "std"


if __name__ == "__main__":
    main()
