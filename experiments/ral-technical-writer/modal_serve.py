"""Modal app `ral-writer-serve` (credits30): the user's humanizer's two models behind vLLM's OpenAI-compatible server,
so the private Humanize pipeline (private/slop-portal/humanize.py) can run on this study's base-model abstracts.

- `writer`: meta-llama/Llama-3.3-70B-Instruct on 2 H100s (the humanizer's rewriter, as in its 2x2x2 ablation)
- `gate`:   mistralai/Mistral-Small-24B-Instruct-2501 on 1 H100 (its in-loop fact check)

    MODAL_IMAGE_BUILDER_VERSION=2025.06 MODAL_PROFILE=credits30 modal deploy experiments/ral-technical-writer/modal_serve.py

Same pinned vLLM as the ablation's modal_llm.py. Bearer token from the Modal secret `ral-writer-llm` (LLM_TOKEN).
At most one container per server, scale to zero after 5 idle minutes; humanize_run.py stops the app when done.
"""

from __future__ import annotations

import subprocess

import modal

app = modal.App("ral-writer-serve")
volume = modal.Volume.from_name("ral-writer", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
MODELS = {"writer": "meta-llama/Llama-3.3-70B-Instruct", "gate": "mistralai/Mistral-Small-24B-Instruct-2501"}
COMMON = {"image": image, "volumes": {"/vol": volume}, "timeout": 2 * 60 * 60, "scaledown_window": 300,
          "max_containers": 1,
          "secrets": [modal.Secret.from_name("ral-writer-llm"), modal.Secret.from_name("huggingface-secret")]}  # fmt: skip


def _serve(name: str, tp: int = 1) -> None:
    import os

    cmd = (
        f"vllm serve {MODELS[name]} --served-model-name {MODELS[name]} --host 0.0.0.0 --port 8000 "
        f"--api-key {os.environ['LLM_TOKEN']} --max-model-len 8192 --gpu-memory-utilization 0.90 "
        f"--max-num-seqs 48 --tensor-parallel-size {tp}" + (" --disable-custom-all-reduce" if tp > 1 else "")
    )  # fmt: skip  (lesson C6)
    subprocess.Popen(cmd, shell=True)


@app.function(gpu="H100:2", **COMMON)
@modal.concurrent(max_inputs=48)
@modal.web_server(port=8000, startup_timeout=60 * 30)
def writer() -> None:
    _serve("writer", tp=2)


@app.function(gpu="H100", **COMMON)
@modal.concurrent(max_inputs=48)
@modal.web_server(port=8000, startup_timeout=60 * 20)
def gate() -> None:
    _serve("gate")
