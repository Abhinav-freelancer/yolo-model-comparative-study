#!/usr/bin/env python
"""Route 3 (custom) classification models: detection backbone + YOLO Classify head.

For YOLO versions that have NO native classification weights in Ultralytics
(no `*-cls.pt`), we take the version's DETECTION yaml from the installed
ultralytics package (backbone section verbatim), replace the detection head
with the standard Ultralytics `Classify` head, and train with the same
protocol as every other model.

Exact head architecture (ultralytics.nn.modules.head.Classify, verified in
ultralytics 8.4.215):
    Conv1x1(c_backbone -> 1280) + BatchNorm + SiLU
    -> AdaptiveAvgPool2d(1)  (global average pool)
    -> Flatten
    -> Dropout(p=0.0 default; trainer's `dropout` arg can raise it)
    -> Linear(1280 -> nc=5)

i.e. conv -> GAP -> dropout -> linear(5), as specified by the benchmark plan.

Backbone initialization: detection-pretrained COCO weights of the SAME
version/size (e.g. yolov10n.pt for the yolov10n backbone), intersect-loaded
into the backbone (matching module indices because the backbone yaml section
is identical). If the detection weights cannot be downloaded or the
intersection covers <50% of backbone params, the model is marked FAILED -
never silently random-initialized (benchmark rule).

Route 3 models are labeled `R3` in every table/plot and never ranked
together with native (R1) models without the tag visible.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml as _yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "r3"  # generated R3 cls yamls (shipped for reproducibility)

# version -> list of dicts describing each buildable R3 variant.
# `size` uses the benchmark's n/s naming; approximations are noted.
# `src_yaml` is resolved inside the installed ultralytics package.
# `out_name` MUST contain the size char right after the version digits
# (ultralytics infers the scale from the filename stem via regex
#  r"yolo(e-)?[v]?\d+([nslmx])" - verified in 8.4.215).
R3_MAP: dict[str, list[dict]] = {
    "v3": [
        dict(size="n", src_yaml="v3/yolov3-tiny.yaml", det_weights="yolov3-tinyu.pt",
             out_name="yolov3n-cls.yaml",
             note="yolov3-tiny as n-scale proxy with yolov3-tinyu detection backbone"),
        dict(size="std", src_yaml="v3/yolov3.yaml", det_weights="yolov3.pt",
             out_name="yolov3-cls.yaml",
             note="yolov3 has no n/s scales in ultralytics; single 'standard' size (69M params)")
    ],
    "v5": [dict(size=s, src_yaml="v5/yolov5.yaml", det_weights=f"yolov5{s}u.pt",
                out_name=f"yolov5{s}-cls.yaml", note="")
           for s in ("n", "s")],
    "v6": [dict(size=s, src_yaml="v6/yolov6.yaml", det_weights=None,
                out_name=f"yolov6{s}-cls.yaml",
                note="yolov6 yaml has scales; ultralytics hosts no yolov6*.pt det weights "
                     "=> no detection-pretrained init available (scope allows pretrained only)")
           for s in ("n", "s")],
    "v9": [dict(size="n", src_yaml="v9/yolov9t.yaml", det_weights="yolov9t.pt",
                out_name="yolov9t-cls.yaml",
                note="ultralytics v9 sizes are t/s/m/c/e; t is the n-class equivalent"),
           dict(size="s", src_yaml="v9/yolov9s.yaml", det_weights="yolov9s.pt",
                out_name="yolov9s-cls.yaml", note="")],
    "v10": [dict(size=s, src_yaml=f"v10/yolov10{s}.yaml", det_weights=f"yolov10{s}.pt",
                 out_name=f"yolov10{s}-cls.yaml", note="")
            for s in ("n", "s")],
    "v12": [dict(size=s, src_yaml="12/yolo12.yaml", det_weights=f"yolo12{s}.pt",
                 out_name=f"yolo12{s}-cls.yaml", note="")
            for s in ("n", "s")],
}


def _find_ultralytics_pkg() -> Path:
    import ultralytics
    return Path(ultralytics.__file__).parent


def build_cls_yaml(version: str, size: str, nc: int = 5, out_dir: Path | None = None) -> dict:
    """Generate a custom classification yaml: <src backbone> + Classify head.

    Returns dict with keys: ok, yaml_path, error, note.
    """
    out_dir = out_dir or OUT_DIR
    try:
        spec = next(v for v in R3_MAP[version] if v["size"] == size)
    except KeyError:
        return dict(ok=False, error=f"no R3 spec for {version}{size}", note="")

    pkg = _find_ultralytics_pkg()
    src = pkg / "cfg" / "models" / spec["src_yaml"]
    if not src.exists():
        return dict(ok=False, error=f"source yaml missing: {src}", note=spec.get("note", ""))

    d = _yaml.safe_load(src.read_text())
    d["nc"] = nc
    d["head"] = [[-1, 1, "Classify", ["nc"]]]  # standard YOLO Classify head (see module docstring)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / spec["out_name"]
    out.write_text(_yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return dict(ok=True, yaml_path=str(out), error="", note=spec.get("note", ""))


def intersect_load_backbone(cls_model, det_model, min_frac: float = 0.5) -> dict:
    """Load detection-pretrained weights into cls_model where keys/shapes match.

    Returns dict(ok, n_loaded, n_total, frac).
    """
    import torch

    det_sd = det_model.model.state_dict() if hasattr(det_model, "model") else det_model.state_dict()
    cls_sd = cls_model.model.state_dict() if hasattr(cls_model, "model") else cls_model.state_dict()
    matched = {k: v for k, v in det_sd.items() if k in cls_sd and cls_sd[k].shape == v.shape}
    frac = len(matched) / max(1, len(cls_sd))
    if matched:
        cls_sd.update(matched)
        (cls_model.model if hasattr(cls_model, "model") else cls_model).load_state_dict(cls_sd)
    return dict(ok=frac >= min_frac, n_loaded=len(matched), n_total=len(cls_sd), frac=round(frac, 3))


def make_r3_model(version: str, size: str, nc: int = 5, pretrained: bool = True, verbose: bool = False):
    """Build the Route 3 ClassificationModel (CPU, no dataset needed).

    Returns (model, info_dict). model is a ClassificationModel ready for training.
    info_dict documents the exact route/architecture for logs and the report.
    """
    import torch

    from ultralytics.nn.tasks import ClassificationModel

    built = build_cls_yaml(version, size, nc=nc)
    if not built["ok"]:
        return None, built
    info = dict(route="R3", version=version, size=size,
                yaml=built["yaml_path"], note=built["note"],
                head="Classify: Conv1x1(c->1280)+BN+SiLU -> GAP -> Flatten -> Dropout(0.0) -> Linear(1280->5)")

    spec = next(v for v in R3_MAP[version] if v["size"] == size)
    model = ClassificationModel(built["yaml_path"], nc=nc, verbose=verbose)
    info["init"] = "random"
    info["det_weights"] = spec["det_weights"] or "none"

    if pretrained:
        if not spec.get("det_weights"):
            info["error"] = "no pretrained weights available in Ultralytics (random init rejected per protocol)"
            return None, info
        try:
            from ultralytics import YOLO
            det = YOLO(f"{spec['det_weights']}")  # downloads COCO det weights if needed
            res = intersect_load_backbone(model, det.model if hasattr(det, "model") else det)
            info["backbone_load"] = res
            if not res["ok"]:
                info["error"] = (f"detection-pretrained intersection too small "
                                 f"({res['n_loaded']}/{res['n_total']} keys)")
                return None, info
            info["init"] = f"detection-pretrained backbone ({spec['det_weights']}), head random"
        except Exception as e:  # download failure etc. -> FAILED, never silent random init
            info["error"] = f"det weights '{spec['det_weights']}' unavailable: {str(e)[:200]}"
            return None, info

    # determinism-friendly freeze check + param count
    info["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
    return model, info


if __name__ == "__main__":
    import json
    import sys

    out = {}
    for ver, specs in R3_MAP.items():
        for spec in specs:
            key = f"{ver}_{spec['size']}"
            try:
                model, info = make_r3_model(ver, spec["size"], pretrained=True)
                out[key] = info | dict(ok=model is not None)
                if model is not None:
                    del model
                torch_free = True
            except Exception as e:
                out[key] = dict(ok=False, error=str(e)[:300])
            print(key, "->", json.dumps(out[key])[:240], file=sys.stderr)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "r3_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote results/r3_probe.json")
