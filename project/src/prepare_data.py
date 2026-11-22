#!/usr/bin/env python
"""Prepare cassava dataset: inspect, clean, stratified split (80/10/10), save manifests/plots."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

CLASSES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]


def md5_file(path, chunk=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="../data set")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--dataset-dir", default="./data/dataset")
    ap.add_argument("--splits-dir", default="./splits")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    data_root = Path(args.data_root).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    figures_dir = results_dir / "figures"
    for d in [data_root, dataset_dir, splits_dir, results_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_root = raw_root / "train"
    test0 = raw_root / "test" / "0"
    if not test0.exists():
        # fallback
        test0 = raw_root / "test"

    # Collect train images
    records = []
    for cls in CLASSES:
        cls_dir = train_root / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.jpg"):
            if "__MACOSX" in p.parts or p.name.startswith("._"):
                continue
            records.append({"path": str(p), "cls": cls, "split": None})

    # Inspect
    counts = Counter(r["cls"] for r in records)
    print("Train counts:", dict(counts))
    sizes = []
    corrupt = []
    hashes = defaultdict(list)
    for r in records:
        p = Path(r["path"])
        if not is_valid_image(p):
            corrupt.append(r["path"])
            continue
        try:
            with Image.open(p) as im:
                sizes.append(im.size)  # (W,H)
                h = md5_file(p)
                hashes[h].append(r["path"])
        except Exception:
            corrupt.append(r["path"])

    if sizes:
        ws = np.array([s[0] for s in sizes])
        hs = np.array([s[1] for s in sizes])
        res_summary = {
            "min_wh": [int(ws.min()), int(hs.min())],
            "median_wh": [int(np.median(ws)), int(np.median(hs))],
            "max_wh": [int(ws.max()), int(hs.max())],
        }
    else:
        res_summary = {"min_wh": None, "median_wh": None, "max_wh": None}

    dup_groups = {k: v for k, v in hashes.items() if len(v) > 1}
    # Mark duplicates (keep first)
    dup_to_remove = set()
    for g in dup_groups.values():
        for p in g[1:]:
            dup_to_remove.add(str(Path(p).resolve()))

    print(f"Corrupt: {len(corrupt)} | Duplicate groups: {len(dup_groups)} | Extra dup files: {len(dup_to_remove)}")

    # Remove corrupt from records
    valid_records = []
    for r in records:
        rp = str(Path(r["path"]).resolve())
        if r["path"] in corrupt or rp in {str(Path(x).resolve()) for x in corrupt}:
            continue
        if rp in dup_to_remove:
            continue
        valid_records.append(r)

    # Stratified split 80/10/10
    y = np.array([r["cls"] for r in valid_records])
    X = np.arange(len(valid_records))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, temp_idx = next(sss1.split(X, y))
    y_temp = y[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)  # 0.1/0.1 of total
    val_rel, test_rel = next(sss2.split(np.arange(len(temp_idx)), y_temp))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    split_map = {}
    for i in tr_idx:
        split_map[i] = "train"
    for i in val_idx:
        split_map[i] = "val"
    for i in test_idx:
        split_map[i] = "test"

    for i, r in enumerate(valid_records):
        r["split"] = split_map[i]

    # Write split manifest
    manifest_path = splits_dir / "split_seed42.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src_path", "cls", "split", "file_hash"])
        w.writeheader()
        for r in valid_records:
            p = Path(r["path"])
            h = md5_file(p)
            w.writerow({"src_path": str(p), "cls": r["cls"], "split": r["split"], "file_hash": h})
    print(f"Saved split manifest: {manifest_path}")

    # Build ultralytics cls layout (copy)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)
    for r in valid_records:
        dst = dataset_dir / r["split"] / r["cls"] / Path(r["path"]).name
        shutil.copy2(r["path"], dst)

    # Test images (flatten test/0)
    test_out = data_root / "test"
    if test_out.exists():
        shutil.rmtree(test_out)
    test_out.mkdir(parents=True, exist_ok=True)
    test_files = []
    for p in sorted(test0.glob("*.jpg")):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        shutil.copy2(p, test_out / p.name)
        test_files.append(p.name)
    # Also include any jpg directly in raw_root/test
    for p in sorted((raw_root / "test").glob("*.jpg")):
        if p.name.startswith("._"):
            continue
        if not (test_out / p.name).exists():
            shutil.copy2(p, test_out / p.name)
            test_files.append(p.name)
    test_files = sorted(set(test_files))
    print(f"Test images copied: {len(test_files)} -> {test_out}")

    # Copy sample submission
    sub_src = raw_root / "sample_submission_file.csv"
    if sub_src.exists():
        shutil.copy2(sub_src, data_root / "sampleSubmission.csv")
        print(f"Copied sampleSubmission.csv")

    # Save report snippets
    summary = {
        "train_counts_raw": dict(counts),
        "counts_after_clean": {c: int(sum(1 for r in valid_records if r["cls"] == c)) for c in CLASSES},
        "corrupt_files": corrupt,
        "duplicate_groups": {k: v for k, v in list(dup_groups.items())[:50]},
        "n_duplicates_removed": len(dup_to_remove),
        "resolution_summary_wh": res_summary,
        "split_counts": {s: int(sum(1 for r in valid_records if r["split"] == s)) for s in ["train", "val", "test"]},
        "test_images_count": len(test_files),
    }
    with open(results_dir / "data_inspection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Class distribution plot
    plt.figure(figsize=(6, 4))
    names = CLASSES
    vals = [summary["counts_after_clean"][c] for c in names]
    plt.bar(names, vals)
    plt.title("Cassava class distribution (after cleaning)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close()
    print("Saved figures/class_distribution.png")

    print("DONE prepare_data")


if __name__ == "__main__":
    main()
