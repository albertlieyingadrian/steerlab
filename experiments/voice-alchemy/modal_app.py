"""Voice Alchemy GPU stages on Modal H100s (COSTS MONEY; every call's GPU seconds go to spend.jsonl, and the run stops
before the $150 hard cap). Always run with MODAL_PROFILE=albertlieyingadrian.

    modal run experiments/voice-alchemy/modal_app.py --stage find                    # units -> results/units*
    modal run experiments/voice-alchemy/modal_app.py --stage gen --jobs data/jobs_dev1.jsonl --out results/gen_dev1.jsonl

Units live on the `voice-alchemy` volume under /vol/units/*.pt (find writes units.pt; circuits.py adds circuits.pt).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformer-lens==4.3.0", "networkx==3.6.1", "numpy<2.4", "hf_transfer==0.1.9")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_file(HERE / "va.py", "/root/va.py")
    .add_local_file(HERE / "gpu.py", "/root/gpu.py")
)
volume = modal.Volume.from_name("voice-alchemy", create_if_missing=True)
app = modal.App("voice-alchemy", image=image)
GPU = {
    "gpu": "H100",
    "timeout": 60 * 60,
    "volumes": {"/vol": volume},
    "secrets": [modal.Secret.from_name("huggingface-secret")],
}
CHUNK = 768


@app.function(**GPU)
def find_stage(tag: str = "gemma") -> dict:
    import os

    os.environ["VA_MODEL"] = tag
    sys.path.insert(0, "/root")
    import gpu
    import va

    t = time.time()
    out = Path(va.CFG["units_dir"])
    gpu.find(out)
    volume.commit()
    files = {n: (out / n).read_bytes() for n in ("units.pt", "units_meta.json", "f_replies.jsonl")}
    return {"files": files, "seconds": time.time() - t}


@app.function(**GPU)
def gen_stage(rows: list[dict], tag: str = "gemma") -> dict:
    import os

    os.environ["VA_MODEL"] = tag
    sys.path.insert(0, "/root")
    import gpu
    import va

    t = time.time()
    volume.reload()
    units: dict = {}
    for p in sorted(Path(va.CFG["units_dir"]).glob("*.pt")):
        units.update(gpu.load_units(p))
    model = gpu.boot()
    out = gpu.generate(model, rows, units)
    return {"rows": out, "seconds": time.time() - t}


@app.local_entrypoint()
def main(stage: str = "find", jobs: str = "", out: str = "") -> None:
    import va

    va.check_budget("modal", planned=2.0)
    if stage == "find":
        res = find_stage.remote(va.TAG)
        va.RESULTS.mkdir(parents=True, exist_ok=True)
        for name, data in res["files"].items():
            (va.RESULTS / name).write_bytes(data)
        va.record_spend("modal", (res["seconds"] + 90) * va.H100_PER_S, "find stage")
        print(f"find: {res['seconds']:.0f}s")
        return
    rows = [json.loads(line) for line in (HERE / jobs).read_text().splitlines() if line.strip()]
    (HERE / out).parent.mkdir(parents=True, exist_ok=True)
    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    va.check_budget("modal", planned=len(chunks) * 0.4)
    got, secs = [], 0.0
    for res in gen_stage.map(chunks, kwargs={"tag": va.TAG}):
        got += res["rows"]
        secs += res["seconds"] + 60
    va.record_spend("modal", secs * va.H100_PER_S, f"gen {jobs} ({len(rows)} rows, {len(chunks)} containers)")
    with (HERE / out).open("w") as f:
        for r in got:
            f.write(json.dumps(r) + "\n")
    print(f"gen: {len(got)} rows, {secs:.0f} GPU-s, total modal spend ${va.spent('modal'):.2f}")
