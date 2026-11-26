"""Run the guard eval on an open model with vLLM on one Modal H100.

    MODAL_PROFILE=albertlieyingadrian modal run experiments/injection-guard/modal_app.py \
        --model Qwen/Qwen3-8B --attacks PATH/attacks.jsonl --out results/qwen3-8b --transcripts PATH/t.jsonl

The whole agent loop runs inside the container (the database is in-memory SQLite), batched across every
conversation, so one call per model. Cost cap: one H100 for at most 40 minutes (about $2.70 at $3.95/h), no retries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("injection-guard")
cache = modal.Volume.from_name("injection-guard-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(HERE / "clinicguard", "/root/clinicguard")
)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": cache},
    timeout=40 * 60,
    retries=0,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_model(model: str, cases: list[dict], levels: list[str], samples: int, temperature: float) -> dict:
    sys.path.insert(0, "/root")
    import time

    from clinicguard.backends import VLLMBackend
    from clinicguard.runner import run_eval

    t0 = time.time()
    backend = VLLMBackend(model, temperature=temperature)
    cache.commit()
    records, transcripts = run_eval(cases, levels, backend, samples, model_name=model)
    return {"records": records, "transcripts": transcripts, "seconds": time.time() - t0}


@app.local_entrypoint()
def main(
    model: str, attacks: str, out: str, transcripts: str, samples: int = 3, temperature: float = 0.7, levels: str = ""
):
    sys.path.insert(0, str(HERE))
    from clinicguard.runner import DEFAULT_LEVELS, load_jsonl, load_legit, write_outputs

    cases = load_legit() + load_jsonl(attacks)
    lv = levels.split(",") if levels else DEFAULT_LEVELS
    res = run_model.remote(model, cases, lv, samples, temperature)
    rows = write_outputs(res["records"], res["transcripts"], Path(out), Path(transcripts))
    print(f"{model}: {len(res['records'])} conversations in {res['seconds']:.0f} s")
    for r in rows:
        print(r)
