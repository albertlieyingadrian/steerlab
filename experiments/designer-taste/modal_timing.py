"""
Per-image compute time of the cheap tiers, for the router's cost column (The Taste Stack Episode 7).

COSTS MONEY (a few cents). Times, on 64 images from the Volume:
  formal  RapidOCR on a CPU container (2 cores) + the pixel statistics
  probe   SigLIP-so400m-384 image embedding on an A10G (batch 32, after warm-up)
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/designer-taste/modal_timing.py
Writes results/timing.json (first run, the registered costs) and appends every run to results/timing_runs.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modal_features import MODELS, hf, vol  # noqa: E402
from modal_features import image as _image  # noqa: E402

app = modal.App("taste-ep7-timing")
image = _image.add_local_python_source("modal_features")  # the container imports this module too
N = 64


@app.function(image=image, cpu=2.0, memory=4096, volumes={"/vol": vol}, timeout=1800)
def time_formal() -> dict:
    import time
    from pathlib import Path

    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    files = sorted(Path("/vol/taste/images").iterdir())[:N]
    ocr = RapidOCR()
    ocr(np.zeros((64, 64, 3), dtype=np.uint8))
    t = time.time()
    for f in files:
        im = Image.open(f).convert("RGB")
        w, h = im.size
        s = min(1.0, 1600 / max(w, h))
        if s < 1:
            im = im.resize((round(w * s), round(h * s)))
        ocr(np.asarray(im))
        im.resize((96, 96))
    return {"formal_s_per_image": (time.time() - t) / len(files), "cpu_cores": 2, "memory_gib": 4}


@app.function(image=image, gpu="A10G", volumes={"/vol": vol, "/hf": hf}, timeout=1800)
def time_probe() -> dict:
    import time
    from pathlib import Path

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    files = sorted(Path("/vol/taste/images").iterdir())[:N]
    model = AutoModel.from_pretrained(MODELS["so400m"], torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(MODELS["so400m"])
    ims = [Image.open(f).convert("RGB") for f in files[:8]]
    with torch.no_grad():
        model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half())
    torch.cuda.synchronize()
    t = time.time()
    for i in range(0, len(files), 32):
        ims = [Image.open(f).convert("RGB") for f in files[i : i + 32]]
        with torch.no_grad():
            model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half())
    torch.cuda.synchronize()
    full = (time.time() - t) / len(files)
    # the same model on the 640 px JPEGs the VLM judges read (post-hoc sensitivity: equal inputs for every GPU tier)
    small = sorted(Path("/vol/jpg640").iterdir())[:N]
    t = time.time()
    for i in range(0, len(small), 32):
        ims = [Image.open(f).convert("RGB") for f in small[i : i + 32]]
        with torch.no_grad():
            model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half())
    torch.cuda.synchronize()
    return {"probe_s_per_image": full, "probe_jpg640_s_per_image": (time.time() - t) / len(small)}


@app.local_entrypoint()
def main():
    from budget import check_modal

    check_modal(0.2, "timing (CPU + A10G, minutes)")
    out = {**time_formal.remote(), **time_probe.remote()}
    # results/timing.json holds the first run (the registered costs); every run is appended to timing_runs.json
    runs_path = HERE / "results" / "timing_runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    runs.append({"run": len(runs) + 1, "note": "repeat (post hoc)" if runs else "registered", **out})
    runs_path.write_text(json.dumps(runs, indent=1))
    if not (HERE / "results" / "timing.json").exists():
        (HERE / "results" / "timing.json").write_text(json.dumps(out, indent=1))
    print(out)
