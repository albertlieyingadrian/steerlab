"""The battery's judge: an open Qwen3 model behind vLLM's OpenAI-compatible server on Modal (COSTS MONEY: an A100-80GB or H100,
up to about $4.40/h with CPU and memory while a container is up; it scales to zero 5 minutes after the last request).

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/graft-monitor/battery/modal_judge.py
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal app stop graft-monitor-battery-judge     # when done

`battery.judge` deploys it on demand and talks to it at <url>/v1 with a bearer token kept in
~/.cache/graft-monitor/battery_judge_token (created on first deploy, never printed). It serves both the battery's
refusal and uncertainty labels and behavior-diff's judge, which in the published study was an OpenRouter model
(qwen3-235b-a22b-2507); here it is this open model on Modal instead. Thinking is turned off per request.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import threading
from pathlib import Path

import modal

APP_NAME = "graft-monitor-battery-judge"
# Qwen3-8B passed the canned known-answer check but mislabelled real small-model replies (27/40 on refusal), so the
# judge is the larger non-thinking Qwen3-30B-A3B-Instruct-2507. GM_BATTERY_JUDGE_MODEL overrides it (redeploy after).
JUDGE_MODEL = os.environ.get("GM_BATTERY_JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
GPUS = ["A100-80GB", "H100"]  # 61 GB of bf16 weights; priced at the dearer entry
TOKEN_FILE = Path.home() / ".cache" / "graft-monitor" / "battery_judge_token"


def token() -> str:
    if not TOKEN_FILE.exists():
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(secrets.token_urlsafe(32))
        TOKEN_FILE.chmod(0o600)
    return TOKEN_FILE.read_text().strip()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1")  # vLLM 0.11 breaks on transformers 5
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "GM_BATTERY_JUDGE_MODEL": JUDGE_MODEL})
)
cache = modal.Volume.from_name("graft-monitor-judge-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)
# Same object locally and in the container (where the key is already in the environment, so no file is read).
_secrets = [modal.Secret.from_dict({"VLLM_API_KEY": os.environ.get("VLLM_API_KEY") or token()})]


@app.function(
    gpu=GPUS,
    cpu=4.0,
    memory=65536,
    timeout=6 * 3600,
    scaledown_window=300,
    max_containers=1,
    volumes={"/cache": cache},
    secrets=_secrets,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=1200)
def serve() -> None:
    assert os.environ.get("VLLM_API_KEY"), "judge started without a token"
    proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            JUDGE_MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.90",
            "--enable-prefix-caching",
            "--served-model-name",
            JUDGE_MODEL,
        ]
    )

    def watchdog() -> None:  # a dead vLLM must not hold the GPU until the startup timeout
        proc.wait()
        os._exit(1)

    threading.Thread(target=watchdog, daemon=True).start()
