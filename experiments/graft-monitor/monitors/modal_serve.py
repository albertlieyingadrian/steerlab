"""OpenAI-compatible vLLM servers on Modal for the monitor judges (COSTS MONEY).

Two web servers in one app, `graft-monitor-serve`, both scale to zero (no keep-warm), one container each:

  safeguard  openai/gpt-oss-safeguard-20b  (MXFP4, harmony format; reasoning_effort per request) on L4|A10|L40S
  qwen       Qwen/Qwen3-8B (bf16, thinking off (template pinned), hermes tool calls for Petri's answer() tool) on an L40S

Deploy, use, stop:

    .venv/bin/python experiments/graft-monitor/monitors/modal_serve.py deploy     # checks the budget, logs the estimate
    # base URLs: https://<workspace>--graft-monitor-serve-<safeguard|qwen>.us-east.modal.run/v1
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal app stop graft-monitor-serve -y

The endpoints require the bearer key in ~/.config/graft-monitor/vllm_api_key (Modal secret graft-monitor-vllm).

Cost control, enforced here and in budget.py:
  - max_containers=1 per server (2 for safeguard during the confirmatory training runs), so at most three GPUs;
  - each container exits after MAX_LIFETIME_S whatever its traffic (a watchdog thread), and scales down after
    SCALEDOWN_S idle;
  - `deploy` refuses when logged spend + this deploy's worst case (one full lifetime per server) exceeds the cap,
    and the bench client logs another lifetime each time it sees a server cold-start.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = "graft-monitor-serve"
VOL = "/vol"
PORT = 8000
MAX_LIFETIME_S = 8 * 60 * 60  # confirmatory training runs call it for hours (was 45 min for the bench)
SCALEDOWN_S = 5 * 60
VLLM_VERSION = "0.11.0"

# list price per hour: GPU + CPU cores x $0.0473 + GiB x $0.0080
SERVERS = {
    # L4 first; A10 and L40S as capacity fallbacks, costed at the dearest of the list
    "safeguard": {
        "model": "openai/gpt-oss-safeguard-20b",
        "gpu": ["L40S", "A10", "L4"],
        "gpu_usd_h": 1.95,
        "cpu": 4,
        "max_containers": 1,
        "mem_gib": 32,
    },
    "qwen": {"model": "Qwen/Qwen3-8B", "gpu": "L40S", "gpu_usd_h": 1.95, "cpu": 4, "mem_gib": 32},
}


def usd_per_hour(name: str) -> float:
    s = SERVERS[name]
    return s["gpu_usd_h"] + s["cpu"] * 0.0473 + s["mem_gib"] * 0.0080


def lifetime_usd(name: str) -> float:
    return usd_per_hour(name) * MAX_LIFETIME_S / 3600


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "transformers==4.57.1")  # vLLM 0.11 breaks on transformers 5
    .env({"HF_HOME": f"{VOL}/hf", "VLLM_LOGGING_LEVEL": "INFO"})
)
volume = modal.Volume.from_name("graft-monitor", create_if_missing=True)
secret = modal.Secret.from_name("graft-monitor-vllm")
app = modal.App(APP_NAME, image=image)


def _watchdog(proc: subprocess.Popen) -> None:
    time.sleep(MAX_LIFETIME_S)
    print(f"lifetime cap {MAX_LIFETIME_S}s reached; stopping vLLM", flush=True)
    proc.terminate()
    time.sleep(10)
    os._exit(0)


def _launch(args: list[str]) -> None:
    cmd = ["vllm", "serve", *args, "--host", "0.0.0.0", "--port", str(PORT), "--api-key", os.environ["VLLM_API_KEY"]]
    proc = subprocess.Popen(cmd)
    threading.Thread(target=_watchdog, args=(proc,), daemon=True).start()


def _fn_kwargs(name: str) -> dict:
    s = SERVERS[name]
    return dict(
        gpu=s["gpu"],
        cpu=float(s["cpu"]),
        memory=s["mem_gib"] * 1024,
        timeout=MAX_LIFETIME_S + 300,
        scaledown_window=SCALEDOWN_S,
        max_containers=s.get("max_containers", 1),
        min_containers=0,
        volumes={VOL: volume},
        secrets=[secret],
    )


@app.function(**_fn_kwargs("safeguard"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def safeguard():
    # The chat template applies the harmony format; reasoning comes back as `reasoning_content`.
    _launch(
        [
            SERVERS["safeguard"]["model"],
            "--served-model-name",
            SERVERS["safeguard"]["model"],
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--max-num-seqs",
            "64",
        ]
    )


@app.function(**_fn_kwargs("qwen"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def qwen():
    _launch(
        [
            SERVERS["qwen"]["model"],
            "--served-model-name",
            SERVERS["qwen"]["model"],
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "24576",
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            "48",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--chat-template",
            _qwen_no_think_template(),
        ]
    )


def _qwen_no_think_template() -> str:
    """Qwen3's own chat template with enable_thinking pinned to false (vLLM 0.11 has no default-kwargs flag)."""
    import json

    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(SERVERS["qwen"]["model"], "tokenizer_config.json")).read_text())
    path = Path("/tmp/qwen3_no_think.jinja")
    path.write_text("{%- set enable_thinking = false %}\n" + cfg["chat_template"])
    return str(path)


def _deploy(servers: list[str]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import budget

    worst = sum(lifetime_usd(n) for n in servers)
    budget.check(worst, what=f"deploy {APP_NAME} ({', '.join(servers)})")
    env = {**os.environ, "MODAL_PROFILE": budget.MODAL_PROFILE}
    subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True, env=env)
    for n in servers:
        budget.log(
            lifetime_usd(n),
            f"deploy {APP_NAME}.{n} worst case one lifetime {MAX_LIFETIME_S}s "
            f"on {SERVERS[n]['gpu']} at ${usd_per_hour(n):.2f}/h",
        )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "deploy":
        _deploy(sys.argv[2:] or list(SERVERS))
    else:
        print(__doc__)
