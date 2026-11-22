from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import final_report as fr

warnings.filterwarnings("ignore")

CFG = fr.load_config(ROOT / "configs" / "benchmark.yaml")
RES = fr.results_root(CFG)
RUNS = fr.runs_root(CFG)
DATA = fr.dataset_root(CFG)
VAL_DIR = DATA / "val"
TEST_DIR = DATA / "test"
KAGGLE_TEST = ROOT / "data" / "test"
SAMPLE = ROOT / "data" / "sampleSubmission.csv"
VAL_ROWS = fr.read_csv_rows(RES / "metrics_val.csv")
AVAIL = fr.expected_benchmark_runs(RES)
CLASSES = [str(item) for item in CFG["classes"]]
INDEX_TO_NAME = {index: name for index, name in enumerate(CLASSES)}
NAME_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
DEVICE = 0
IMGSZ = int(CFG["training"]["imgsz"])
STATE: Dict[str, Any] = {}


def note(stage: str, message: str) -> None:
    try:
        fr.append_progress(f"[{stage}] {message}")
    except Exception as exc:
        print(f"NOTE_FAILED {stage}: {exc!r}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def display_name(meta: Dict[str, str]) -> str:
    if meta.get("version") == "v3":
        return "YOLOv3-tiny"
    return str(meta.get("model_name", meta.get("run_name", "?")))


def rank_rows() -> List[Dict[str, str]]:
    usable = []
    for row in VAL_ROWS:
        score = fr.as_float(row.get("macro_f1"))
        if score is None or not row.get("run_name"):
            continue
        usable.append(row)
    usable.sort(key=lambda item: fr.as_float(item.get("macro_f1")) or 0.0, reverse=True)
    return usable


def step0() -> None:
    failures = RES / "failures.csv"
    if failures.exists():
        text = failures.read_text(encoding="utf-8").strip()
        if text:
            raise RuntimeError(f"results\\failures.csv is not empty: {text[:1000]}")
        failure_note = "results\\failures.csv exists and is empty"
    else:
        failure_note = "results\\failures.csv absent; no failed runs were recorded (all 13 passed)"

    check_runs = ["v10_n_pretrained_seed0", "v12_n_pretrained_seed0", "v12_s_pretrained_seed0"]
    images, labels = fr.collect_images(VAL_DIR, CLASSES, NAME_TO_INDEX)
    y_true = np.array(labels, dtype=int)
    out_dir = RES / "val_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []

    for run_name in check_runs:
        run_dir = RUNS / run_name
        weight_path = run_dir / "weights" / "best.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing weights: {weight_path}")
        model, names, _ = fr.load_model_and_names(run_dir, DEVICE)
        probs = fr.predict_probs(model, images, IMGSZ, DEVICE, 32)
        pred = probs.argmax(axis=1).astype(int)
        correct = int((pred == y_true).sum())
        pred_path = out_dir / f"{run_name}_val_predictions.csv"
        fr.write_predictions(pred_path, images, y_true, pred, probs, names)
        weights.append({"run_name": run_name, "path": str(weight_path), "sha256": sha256(weight_path)})
        predictions.append({"run_name": run_name, "path": str(pred_path),
                            "sha256": sha256(pred_path), "correct": correct,
                            "n_images": len(images)})
        del model
        torch.cuda.empty_cache()

    counts = {int(item["correct"]) for item in predictions}
    if counts != {457}:
        raise RuntimeError(f"Expected 457/564 for all three runs; got {sorted(counts)}")
    weight_hashes = {item["sha256"] for item in weights}
    pred_hashes = {item["sha256"] for item in predictions}
    if len(weight_hashes) != 3:
        raise RuntimeError("v10n/v12n/v12s do not use three distinct weights files")
    if len(pred_hashes) != 3:
        raise RuntimeError("v10n/v12n/v12s do not produce three distinct prediction files")

    payload = {"failures": failure_note, "weights": weights, "predictions": predictions,
               "all_correct_457_of_564": True, "distinct_weights": True,
               "distinct_prediction_files": True}
    write_json(RES / "val_checks_duplicate_report.json", payload)
    STATE["step0"] = payload
    note("STEP0", f"failures.csv: {failure_note}; v10n/v12n/v12s each 457/564 correct on shared val with 3 distinct best.pt files and 3 distinct saved prediction CSVs; v3 display label set to YOLOv3-tiny.")


def step1() -> None:
    final_runs = ["v11_s_pretrained_seed0", "v8_s_pretrained_seed0"]
    v11_dir = RES / "final" / "v11_s_pretrained_seed0_test"
    if not (v11_dir / "summary.json").exists():
        raise FileNotFoundError(f"Missing existing single hold-out evaluation: {v11_dir / 'summary.json'}")

    command = [sys.executable, str(ROOT / "src" / "final_report.py"),
               "--model", "v8_s_pretrained_seed0"]
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    log_path = RES / "final_report_v8s_out.log"
    log_path.write_text(f"COMMAND: {' '.join(command)}\n\n{result.stdout}\n{result.stderr}", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"final_report.py failed with exit code {result.returncode}. Exact stderr:\n{result.stderr}")

    records: Dict[str, Any] = {}
    for run_name in final_runs:
        out_dir = RES / "final" / f"{run_name}_test"
        summary_path = out_dir / "summary.json"
        per_class_path = out_dir / "per_class_metrics.csv"
        if not summary_path.exists() or not per_class_path.exists():
            raise FileNotFoundError(f"Incomplete hold-out artifacts in {out_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        per_class = list(csv.DictReader(per_class_path.open(encoding="utf-8")))
        pngs = [str(path) for path in sorted(out_dir.glob("*.png"))]
        records[run_name] = {"out_dir": str(out_dir), "summary": summary,
                             "per_class": per_class, "pngs": pngs,
                             "reused_existing_evaluation": run_name == "v11_s_pretrained_seed0"}

    write_json(RES / "holdout_results.json", records)
    STATE["step1"] = records
    for run_name, record in records.items():
        summary = record["summary"]
        note("STEP1", f"hold-out {run_name}: n={summary['n_images']} acc={summary['accuracy']:.6f} macroF1={summary['macro_f1']:.6f} balanced_acc={summary['balanced_accuracy']:.6f}; artifacts={record['out_dir']}; {'reused prior single eval' if record['reused_existing_evaluation'] else 'evaluated once now'}")


def step2() -> None:
    rows: List[Dict[str, Any]] = []
    for rank, source in enumerate(rank_rows(), 1):
        meta = fr.meta_for_run(source["run_name"], VAL_ROWS, AVAIL)
        name = display_name(meta)
        rows.append({
            "rank": rank,
            "run_name": source["run_name"],
            "model_name": name,
            "route": meta["route"],
            "version": meta["version"],
            "size": meta["size"],
            "chart_label": f"{name}\n[{meta['route']}]",
            "val_accuracy": fr.as_float(source.get("top1_acc")),
            "val_macro_precision": fr.as_float(source.get("macro_precision")),
            "val_macro_recall": fr.as_float(source.get("macro_recall")),
            "val_macro_f1": fr.as_float(source.get("macro_f1")),
        })

    fields = ["rank", "run_name", "model_name", "route", "version", "size",
              "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1"]
    csv_path = RES / "comparison_val.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fields})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [("val_accuracy", "Accuracy"), ("val_macro_precision", "Macro precision"),
            ("val_macro_recall", "Macro recall"), ("val_macro_f1", "Macro F1")]
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(16, len(rows) * 1.35), 6.5))
    for offset, (key, label) in enumerate(keys):
        values = [float(row[key] or 0.0) for row in rows]
        ax.bar(x + (offset - 1.5) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([row["chart_label"] for row in rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation score")
    ax.set_title("Validation comparison: all 13 models (R1=ImageNet classifier, R3=COCO backbone + Classify head)")
    ax.legend(loc="upper center", ncol=4)
    fig.tight_layout()
    chart_path = RES / "comparison_val.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    STATE["step2"] = {"csv": str(csv_path), "chart": str(chart_path), "rows": rows}
    note("STEP2", f"wrote route-tagged 13-model comparison table {csv_path.name} and one grouped bar chart {chart_path.name}; v3 labelled YOLOv3-tiny")


def measure_latency(model: Any, batch: int) -> float:
    dummy = torch.randn(batch, 3, IMGSZ, IMGSZ, device=DEVICE)
    with torch.inference_mode():
        for _ in range(20):
            model.model(dummy)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        elapsed: List[float] = []
        for _ in range(200):
            start.record()
            model.model(dummy)
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
    return float(sum(elapsed) / len(elapsed) / batch)


def step3() -> None:
    from thop import profile

    records: List[Dict[str, Any]] = []
    for source in VAL_ROWS:
        run_name = source["run_name"]
        meta = fr.meta_for_run(run_name, VAL_ROWS, AVAIL)
        run_dir = RUNS / run_name
        weight_path = run_dir / "weights" / "best.pt"
        model, _, _ = fr.load_model_and_names(run_dir, DEVICE)
        dummy = torch.randn(1, 3, IMGSZ, IMGSZ, device=DEVICE)
        with torch.no_grad():
            flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
        params_m = sum(parameter.numel() for parameter in model.model.parameters()) / 1e6
        size_mb = weight_path.stat().st_size / 1e6
        latency_b1 = measure_latency(model, 1)
        latency_b32 = measure_latency(model, 32)
        records.append({
            "run_name": run_name,
            "model_name": display_name(meta),
            "route": meta["route"],
            "params_M": round(params_m, 4),
            "gflops_224": round(float(flops) / 1e9, 4),
            "best_pt_MB": round(size_mb, 3),
            "latency_ms_per_img_b1": round(latency_b1, 4),
            "latency_ms_per_img_b32": round(latency_b32, 4),
        })
        del model, dummy
        torch.cuda.empty_cache()

    fields = ["run_name", "model_name", "route", "params_M", "gflops_224", "best_pt_MB",
              "latency_ms_per_img_b1", "latency_ms_per_img_b32"]
    csv_path = RES / "efficiency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    write_json(RES / "efficiency.json", records)
    STATE["step3"] = {"csv": str(csv_path), "records": records}
    note("STEP3", f"wrote efficiency.csv/json for 13 models: thop GFLOPs at 224, params, best.pt MB, GPU latency batch1/batch32 (warmup 20, mean over 200) on RTX 3050 Laptop 4GB")


def load_holdout_predictions(run_name: str) -> Tuple[List[Dict[str, str]], np.ndarray, np.ndarray]:
    path = RES / "final" / f"{run_name}_test" / "predictions.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No hold-out predictions in {path}")
    y = np.array([CLASSES.index(row["true"]) for row in rows], dtype=int)
    pred = np.array([CLASSES.index(row["predicted"]) for row in rows], dtype=int)
    return rows, y, pred


def step4() -> None:
    from scipy.stats import binomtest
    from sklearn.metrics import f1_score

    rows_a, y_a, pred_a = load_holdout_predictions("v11_s_pretrained_seed0")
    rows_b, y_b, pred_b = load_holdout_predictions("v8_s_pretrained_seed0")
    if [row["image"] for row in rows_a] != [row["image"] for row in rows_b]:
        raise RuntimeError("Hold-out prediction image order differs between v11s and v8s")
    if len(y_a) != 565:
        raise RuntimeError(f"Expected 565 hold-out predictions, got {len(y_a)}")

    rng = np.random.default_rng(0)
    labels = list(range(len(CLASSES)))

    def bootstrap(y: np.ndarray, pred: np.ndarray) -> Dict[str, List[float]]:
        accuracies: List[float] = []
        macro_f1s: List[float] = []
        for _ in range(1000):
            index = rng.integers(0, len(y), len(y))
            accuracies.append(float((y[index] == pred[index]).mean()))
            macro_f1s.append(float(f1_score(y[index], pred[index], labels=labels,
                                            average="macro", zero_division=0)))
        return {"accuracy_ci95": [float(v) for v in np.percentile(accuracies, [2.5, 97.5])],
                "macro_f1_ci95": [float(v) for v in np.percentile(macro_f1s, [2.5, 97.5])]}

    ci_a = bootstrap(y_a, pred_a)
    ci_b = bootstrap(y_b, pred_b)
    b = int(((pred_a == y_a) & (pred_b != y_b)).sum())
    c = int(((pred_a != y_a) & (pred_b == y_b)).sum())
    p_value = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    payload = {
        "method": "1000 bootstrap resamples, seed 0; exact two-sided McNemar via scipy.stats.binomtest",
        "alpha": 0.05,
        "v11_s_pretrained_seed0": {"accuracy": float((y_a == pred_a).mean()),
                                   "macro_f1": float(f1_score(y_a, pred_a, labels=labels, average="macro", zero_division=0)),
                                   **ci_a},
        "v8_s_pretrained_seed0": {"accuracy": float((y_b == pred_b).mean()),
                                  "macro_f1": float(f1_score(y_b, pred_b, labels=labels, average="macro", zero_division=0)),
                                  **ci_b},
        "mcnemar": {"v11s_correct_v8s_wrong": b, "v11s_wrong_v8s_correct": c,
                    "discordant_total": b + c, "p_value": p_value,
                    "significant_at_0_05": p_value < 0.05},
    }
    write_json(RES / "statistics.json", payload)
    STATE["step4"] = payload
    note("STEP4", f"bootstrap 95% CI and McNemar complete; p={p_value:.6f}; significant_at_0_05={p_value < 0.05}")


def flip_probs(model: Any, images: List[Path]) -> np.ndarray:
    arrays = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image for flip TTA: {path}")
        arrays.append(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
    chunks = []
    for start in range(0, len(arrays), 32):
        results = model.predict(arrays[start:start + 32], imgsz=IMGSZ,
                                device=DEVICE, verbose=False, stream=False)
        for result in results:
            chunks.append(result.probs.data.cpu().numpy())
    return np.vstack(chunks)


def write_submission(path: Path, probs: np.ndarray, ids: List[str], names: Dict[int, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Category", "Id"])
        for identifier, row in zip(ids, probs):
            writer.writerow([names[int(row.argmax())], identifier])


def validate_submission(path: Path, sample_ids: List[str]) -> Dict[str, Any]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    ids = [row.get("Id", "") for row in rows]
    labels = [row.get("Category", "") for row in rows]
    allowed = set(CLASSES)
    result = {
        "file": str(path),
        "rows": len(rows),
        "columns_ok": header == ["Category", "Id"],
        "row_count_ok": len(rows) == 3774,
        "no_duplicate_ids": len(ids) == len(set(ids)),
        "no_missing_ids": set(ids) == set(sample_ids),
        "ids_in_sample_order": ids == sample_ids,
        "filenames_have_extension": all(Path(identifier).suffix for identifier in ids),
        "labels_ok": all(label in allowed for label in labels),
    }
    result["passed"] = all(result[key] for key in (
        "columns_ok", "row_count_ok", "no_duplicate_ids", "no_missing_ids",
        "ids_in_sample_order", "filenames_have_extension", "labels_ok"))
    return result


def step5() -> None:
    top3 = [row["run_name"] for row in rank_rows()[:3]]
    with open(SAMPLE, newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    sample_ids = [row["Id"] for row in sample_rows]
    if len(sample_ids) != 3774:
        raise RuntimeError(f"Sample submission has {len(sample_ids)} rows, expected 3774")
    image_paths = [KAGGLE_TEST / identifier for identifier in sample_ids]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Kaggle test images, first: {missing[0]}")

    out_dir = RES / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_by_run: Dict[str, np.ndarray] = {}
    names_by_run: Dict[str, Dict[int, str]] = {}
    created: List[str] = []

    for run_name in top3:
        model, ordered, names = fr.load_model_and_names(RUNS / run_name, DEVICE)
        plain = fr.predict_probs(model, image_paths, IMGSZ, DEVICE, 32)
        flipped = flip_probs(model, image_paths)
        tta = (plain + flipped) / 2.0
        probs_by_run[run_name] = plain
        names_by_run[run_name] = names
        plain_path = out_dir / f"submission_{run_name}_plain.csv"
        tta_path = out_dir / f"submission_{run_name}_flip_tta.csv"
        write_submission(plain_path, plain, sample_ids, names)
        write_submission(tta_path, tta, sample_ids, names)
        created.extend([str(plain_path), str(tta_path)])
        del model
        torch.cuda.empty_cache()

    if len({tuple(sorted(names.items())) for names in names_by_run.values()}) != 1:
        raise RuntimeError("Top-3 models have different class-to-index mappings")
    ensemble = np.mean([probs_by_run[name] for name in top3], axis=0)
    ensemble_path = out_dir / "submission_top3_mean_softmax.csv"
    write_submission(ensemble_path, ensemble, sample_ids, names_by_run[top3[0]])
    created.append(str(ensemble_path))

    validations = [validate_submission(Path(path), sample_ids) for path in created]
    validation_path = RES / "submission_validation.csv"
    with open(validation_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validations[0].keys()))
        writer.writeheader()
        writer.writerows(validations)
    failed = [item["file"] for item in validations if not item["passed"]]
    if failed:
        raise RuntimeError(f"Submission validation failed for: {failed}")

    payload = {"top3": top3, "created": created, "validation_csv": str(validation_path),
               "validations": validations, "hidden_labels_checked": False}
    write_json(RES / "submissions.json", payload)
    STATE["step5"] = payload
    note("STEP5", f"created {len(created)} Kaggle submissions for top3 {top3} (plain, flip-TTA, top3 mean-softmax ensemble); all CSVs passed structural validation against sample; hidden labels not checkable")


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def step6() -> None:
    if "step0" not in STATE:
        STATE["step0"] = json.loads((RES / "val_checks_duplicate_report.json").read_text(encoding="utf-8"))
        STATE["step1"] = json.loads((RES / "holdout_results.json").read_text(encoding="utf-8"))
        comparison_rows = []
        for source in csv.DictReader((RES / "comparison_val.csv").open(encoding="utf-8")):
            row = dict(source)
            for key in ("rank", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1"):
                row[key] = int(row[key]) if key == "rank" else float(row[key])
            comparison_rows.append(row)
        STATE["step2"] = {"csv": str(RES / "comparison_val.csv"),
                          "chart": str(RES / "comparison_val.png"), "rows": comparison_rows}
        STATE["step3"] = {"csv": str(RES / "efficiency.csv"),
                          "records": json.loads((RES / "efficiency.json").read_text(encoding="utf-8"))}
        STATE["step4"] = json.loads((RES / "statistics.json").read_text(encoding="utf-8"))
        STATE["step5"] = json.loads((RES / "submissions.json").read_text(encoding="utf-8"))

    step0_data = STATE["step0"]
    step1_data = STATE["step1"]
    step2_data = STATE["step2"]
    efficiency = STATE["step3"]["records"]
    stats = STATE["step4"]
    submissions = STATE["step5"]

    v11 = step1_data["v11_s_pretrained_seed0"]["summary"]
    v8 = step1_data["v8_s_pretrained_seed0"]["summary"]
    per_class_v11 = step1_data["v11_s_pretrained_seed0"]["per_class"]
    per_class_v8 = step1_data["v8_s_pretrained_seed0"]["per_class"]

    availability_rows = [
        ["R1", "v8 n/s, v11 n/s", "native ImageNet-pretrained *-cls weights", "4 trained"],
        ["R3", "v3 n, v5 n/s, v9 n/s, v10 n/s, v12 n/s", "COCO detection backbone + new Classify head", "9 trained"],
        ["Skipped", "v4, v6, v7", "no usable pretrained classification/detection weights in scope", "0 trained"],
    ]

    protocol_rows = [
        ["Dataset", "iCassava 2019 / FGVCx, 5 classes"],
        ["Internal split", "4512 train / 564 val / 565 hold-out test; split seed 42"],
        ["Kaggle test", "3774 unlabeled images; submissions structurally validated only"],
        ["Training seed", "0"],
        ["Epochs / patience", "30 / 10"],
        ["Batch / image size", "16 / 224x224"],
        ["Optimizer / schedule", "AdamW lr 0.001, cosine to 1%"],
        ["Loss", "class-weighted cross-entropy + label smoothing 0.1"],
        ["AMP / workers / cache", "on / 2 / off"],
        ["Hold-out use", "opened once per final model after selection"],
    ]

    ranked_rows = []
    for row in step2_data["rows"]:
        ranked_rows.append([row["rank"], row["model_name"], row["route"],
                            f"{row['val_accuracy']:.6f}", f"{row['val_macro_precision']:.6f}",
                            f"{row['val_macro_recall']:.6f}", f"{row['val_macro_f1']:.6f}"])

    final_holdout_rows = []
    for label, summary in (("v11s (final)", v11), ("v8s (runner-up)", v8)):
        final_holdout_rows.append([label, summary["route"], summary["n_images"],
                                   f"{summary['accuracy']:.6f}", f"{summary['macro_f1']:.6f}",
                                   f"{summary['balanced_accuracy']:.6f}"])

    per_class_headers = ["Class/metric", "Precision", "Recall", "F1",
                         "One-vs-rest accuracy", "Support", "Value"]

    def per_class_table(rows: List[Dict[str, str]]) -> str:
        return markdown_table(per_class_headers, [
            [row["metric"], row["precision"], row["recall"], row["f1"],
             row["one_vs_rest_accuracy"], row["support"], row.get("value", "")]
            for row in rows
        ])

    per_class_v11_table = per_class_table(per_class_v11)
    per_class_v8_table = per_class_table(per_class_v8)

    efficiency_rows = []
    for record in efficiency:
        efficiency_rows.append([record["model_name"], record["route"],
                                f"{record['params_M']:.3f}", f"{record['gflops_224']:.3f}",
                                f"{record['best_pt_MB']:.2f}",
                                f"{record['latency_ms_per_img_b1']:.3f}",
                                f"{record['latency_ms_per_img_b32']:.3f}"])

    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    prediction_rows = list(csv.DictReader(
        (RES / "final" / "v11_s_pretrained_seed0_test" / "predictions.csv").open(encoding="utf-8")))
    y_true_cm = np.array([CLASSES.index(row["true"]) for row in prediction_rows], dtype=int)
    y_pred_cm = np.array([CLASSES.index(row["predicted"]) for row in prediction_rows], dtype=int)
    cm = sk_confusion_matrix(y_true_cm, y_pred_cm, labels=list(range(len(CLASSES))))
    confusions = []
    for true_index in range(len(CLASSES)):
        for pred_index in range(len(CLASSES)):
            if true_index != pred_index and cm[true_index, pred_index] > 0:
                confusions.append((int(cm[true_index, pred_index]),
                                   CLASSES[true_index], CLASSES[pred_index]))
    confusions.sort(reverse=True)
    confusion_rows = [[count, f"{true} -> {pred}"]
                      for count, true, pred in confusions[:8]]

    statistics_rows = [
        ["v11_s_pretrained_seed0", f"{stats['v11_s_pretrained_seed0']['accuracy']:.6f}",
         f"{stats['v11_s_pretrained_seed0']['accuracy_ci95'][0]:.6f}-{stats['v11_s_pretrained_seed0']['accuracy_ci95'][1]:.6f}",
         f"{stats['v11_s_pretrained_seed0']['macro_f1']:.6f}",
         f"{stats['v11_s_pretrained_seed0']['macro_f1_ci95'][0]:.6f}-{stats['v11_s_pretrained_seed0']['macro_f1_ci95'][1]:.6f}"],
        ["v8_s_pretrained_seed0", f"{stats['v8_s_pretrained_seed0']['accuracy']:.6f}",
         f"{stats['v8_s_pretrained_seed0']['accuracy_ci95'][0]:.6f}-{stats['v8_s_pretrained_seed0']['accuracy_ci95'][1]:.6f}",
         f"{stats['v8_s_pretrained_seed0']['macro_f1']:.6f}",
         f"{stats['v8_s_pretrained_seed0']['macro_f1_ci95'][0]:.6f}-{stats['v8_s_pretrained_seed0']['macro_f1_ci95'][1]:.6f}"],
    ]

    submission_rows = [[Path(item["file"]).name, item["rows"], item["passed"]]
                       for item in submissions["validations"]]

    png_lines = []
    for run_name, record in step1_data.items():
        png_lines.append(f"### {run_name} hold-out PNGs")
        png_lines.extend([f"- `{path}`" for path in record["pngs"]])
        png_lines.append("")

    report = f"""# Cassava YOLO Benchmark — Final Report

## Dataset and splits
- Dataset: iCassava 2019 / FGVCx, 5 classes: {', '.join(CLASSES)}.
- Internal split (seed 42): 4512 train / 564 validation / 565 hold-out test images.
- Kaggle test set: 3774 unlabeled images; submissions are checked structurally only, never against hidden labels.

## Availability and routes
{markdown_table(['Route', 'Models', 'Meaning', 'Status'], availability_rows)}
- v3 is labelled **YOLOv3-tiny** everywhere in this report and in all generated tables/charts.
- {step0_data['failures']}.
- Duplicate-score check: {', '.join(f"{item['run_name']} = {item['correct']}/{item['n_images']}" for item in step0_data['predictions'])}. The three best.pt files have distinct SHA-256 hashes and the three saved validation prediction CSVs have distinct SHA-256 hashes.

## Protocol
{markdown_table(['Item', 'Setting'], protocol_rows)}

## Ranked validation comparison (all 13 models)
{markdown_table(['Rank', 'Model', 'Route', 'Val accuracy', 'Val macro P', 'Val macro R', 'Val macro F1'], ranked_rows)}
- Route-tagged table: `{step2_data['csv']}`
- One grouped bar chart: `{step2_data['chart']}`

## Final hold-out results
{markdown_table(['Model', 'Route', 'Images', 'Accuracy', 'Macro F1', 'Balanced accuracy'], final_holdout_rows)}
- Final model was selected by validation macro-F1 before the hold-out was opened: `v11_s_pretrained_seed0` (0.829294), a near-tie with `v8_s_pretrained_seed0` (0.828997).
- The `v11_s_pretrained_seed0` hold-out evaluation was reused from its single prior evaluation; `v8_s_pretrained_seed0` was evaluated once in this run. No model was evaluated twice.

## Per-class results — final model v11s on hold-out
{per_class_v11_table}

## Per-class results — runner-up v8s on hold-out
{per_class_v8_table}

## Hold-out artifacts (full PNG paths)
{chr(10).join(png_lines)}
Per-image predictions and per-class CSVs are in `results\\final\\<run>_test\\`.

## Efficiency (all 13 models)
{markdown_table(['Model', 'Route', 'Params (M)', 'GFLOPs @224', 'best.pt (MB)', 'Latency b1 ms/img', 'Latency b32 ms/img'], efficiency_rows)}
- Params: `sum(p.numel())` from each loaded checkpoint.
- GFLOPs: `thop.profile` with a 1x3x224x224 input.
- Latency: GPU inference on **NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM**, warmup 20 iterations, mean over 200 timed iterations, batch 1 and batch 32. These are laptop-GPU numbers, not datacenter-GPU numbers.

## Statistics
{markdown_table(['Model', 'Accuracy', 'Accuracy 95% CI', 'Macro F1', 'Macro F1 95% CI'], statistics_rows)}
- Bootstrap: 1000 resamples, seed 0.
- McNemar (v11s vs v8s): v11s correct / v8s wrong = {stats['mcnemar']['v11s_correct_v8s_wrong']}; v11s wrong / v8s correct = {stats['mcnemar']['v11s_wrong_v8s_correct']}; discordant = {stats['mcnemar']['discordant_total']}; exact two-sided p = {stats['mcnemar']['p_value']:.6f}.
- Plain language: the v11s-vs-v8s difference is **{'statistically significant at alpha=0.05' if stats['mcnemar']['significant_at_0_05'] else 'NOT statistically significant at alpha=0.05'}**.

## Error analysis
Top off-diagonal hold-out confusions for v11s:
{markdown_table(['Count', 'True -> Predicted'], confusion_rows)}
- The main errors are between visually overlapping foliar symptom classes. In particular, cbsd and cmd both produce irregular brown/gray leaf lesions under field lighting; cbb and cgm can both present with yellowing and wilting. Class imbalance also matters: cmd has 266 hold-out images while healthy has 31, so minority-class F1 is weaker and errors concentrate in the larger lesion classes.

## Submissions
{markdown_table(['File', 'Rows', 'Structurally valid'], submission_rows)}
- Top 3 by validation macro-F1: {', '.join(submissions['top3'])}.
- Variants: plain, horizontal-flip TTA, and one mean-softmax ensemble of the three top models' plain probabilities.
- All CSVs were validated against `{SAMPLE}`: columns `Category,Id`, 3774 rows, no duplicate or missing Ids, filenames with extension, and labels within `{{cbb,cbsd,cgm,cmd,healthy}}`. Hidden-label accuracy cannot be checked.

## Recommendations
- Use the top-3 mean-softmax ensemble if Kaggle leaderboard latency is not binding; otherwise v11s is the single best validation model.
- Increase R3 training beyond 30 epochs with early stopping, since R3 models were still improving late in training.
- Repeat with multiple seeds/splits and add external-field images to reduce sensitivity to one split.
- Consider class-balanced sampling or focal loss for cbb/healthy, and collect harder cbsd/cmd examples.

## Limitations
- Single seed and single split.
- n/s sizes only; no m/l/x models.
- Latency measured on a 4 GB laptop GPU, not a datacenter GPU.
- R3 models ran all 30 epochs and may be undertrained.
- R1 vs R3 pretraining differs: R1 uses ImageNet classifier weights, R3 uses COCO detector backbones with a new Classify head; differences are not purely architectural.
- Kaggle hidden labels were not available, so submissions were checked structurally only.

## Citation
Mwebaze et al., "iCassava 2019 Fine-Grained Visual Categorization Challenge", https://arxiv.org/pdf/1908.02900.pdf
Ultralytics YOLO: https://github.com/ultralytics/ultralytics
"""
    report_path = RES / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    STATE["step6"] = {"report": str(report_path)}
    note("STEP6", f"wrote {report_path} with dataset, routes, protocol, ranked table, per-class results, efficiency, statistics, error analysis, submissions, recommendations, limitations and citations")


def main() -> None:
    if "--report-only" in sys.argv:
        step6()
        write_json(RES / "final_pipeline_DONE.json", {"status": "ok", "mode": "report-only", "state": STATE})
        print("FINAL_PIPELINE_DONE", flush=True)
        return
    step0()
    step1()
    step2()
    step3()
    step4()
    step5()
    step6()
    write_json(RES / "final_pipeline_DONE.json", {"status": "ok", "state": STATE})
    print("FINAL_PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        message = traceback.format_exc()
        (RES / "final_pipeline_error.log").write_text(message, encoding="utf-8")
        try:
            fr.append_progress("FINAL PIPELINE ERROR: see results\\final_pipeline_error.log")
        except Exception:
            pass
        print(message, flush=True)
        sys.exit(1)
