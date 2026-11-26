"""The improver LLM for pilot 2 (FIXED; plan amendment 3, P2-6): Qwen3-Coder-30B-A3B-Instruct (FP8) under vLLM.

    MODAL_PROFILE=modal4 MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy experiments/slop-rsi/modal_llm.py

OpenRouter returned 402 on both keys on 2026-11-24, so the model both arms use is served here instead of Claude
Sonnet 5.5. One L40S container at most; it scales to zero after 4 idle minutes. The OpenAI-compatible endpoint needs
a key that is kept outside the repo (~/.config/slop-rsi/llmkey) and passed to the container as a secret.
"""

from __future__ import annotations

import os
import secrets
import subprocess
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
CONTEXT = 49152
GPU = os.environ.get("RSI_LLM_GPU", "L40S")


def llm_key() -> str:
    p = Path.home() / ".config" / "slop-rsi" / "llmkey"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(16))
        p.chmod(0o600)
    return p.read_text().strip()


app = modal.App("slop-rsi-llm")
hf = modal.Volume.from_name("slop-rsi-llm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
_secret = modal.Secret.from_dict({"LLM_KEY": llm_key()}) if modal.is_local() else modal.Secret.from_dict({})


@app.cls(image=image, gpu=GPU, volumes={"/hf": hf}, secrets=[_secret], timeout=8 * 3600, scaledown_window=240,
         max_containers=1)  # fmt: skip
@modal.concurrent(max_inputs=16)
class Server:
    @modal.web_server(8000, startup_timeout=1800)
    def serve(self) -> None:
        subprocess.Popen(
            [
                "vllm",
                "serve",
                MODEL,
                "--port",
                "8000",
                "--served-model-name",
                "improver",
                "--max-model-len",
                str(CONTEXT),
                "--gpu-memory-utilization",
                "0.92",
                "--max-num-seqs",
                "8",
                "--enable-prefix-caching",
                "--api-key",
                os.environ["LLM_KEY"],
                "--disable-log-requests",
            ]
        )
