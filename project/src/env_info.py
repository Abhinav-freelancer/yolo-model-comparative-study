#!/usr/bin/env python
"""Record environment info to results/env_info.json. Run from project/ root."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project/


def gpu_vram_mib(name_filter: str = "NVIDIA") -> dict:
    """Query nvidia-smi for the discrete NVIDIA GPU (never the AMD iGPU)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        for line in out:
            if name_filter.lower() in line.lower():
                name, mem, drv = [x.strip() for x in line.split(",")]
                return {"gpu_name": name,
                        "vram_total_mib": int("".join(ch for ch in mem if ch.isdigit())),
                        "driver_version": drv}
    except Exception as e:
        return {"nvidia_smi_error": str(e)}
    return {"nvidia_smi_error": "no NVIDIA GPU found in nvidia-smi output"}


def main() -> None:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_exe": sys.executable,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu_name_torch"] = torch.cuda.get_device_name(0)  # device 0 forced everywhere
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:
        info["torch_error"] = str(e)
    info.update(gpu_vram_mib())
    for mod in ["ultralytics", "timm", "sklearn", "pandas", "numpy", "thop", "cv2", "PIL"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", getattr(m, "version", "unknown"))
        except Exception as e:
            info[f"{mod}_error"] = str(e)
    info["seed_note"] = "split seed=42 (prepare_data); run seed=0 (training)"

    out = ROOT / "results" / "env_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
