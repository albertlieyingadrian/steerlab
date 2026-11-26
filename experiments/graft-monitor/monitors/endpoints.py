"""Where the Modal judge servers live and how to authenticate (the key is read, never printed).

Override a base URL with GRAFT_SAFEGUARD_BASE_URL / GRAFT_QWEN_BASE_URL and the key with GRAFT_VLLM_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = "albertlieyingadrian"
REGION = "us-east"
KEY_FILE = Path.home() / ".config" / "graft-monitor" / "vllm_api_key"

MODELS = {"safeguard": "openai/gpt-oss-safeguard-20b", "qwen": "Qwen/Qwen3-8B"}


def base_url(server: str) -> str:
    env = os.environ.get(f"GRAFT_{server.upper()}_BASE_URL")
    if env:
        return env.rstrip("/")
    return f"https://{WORKSPACE}--graft-monitor-serve-{server}.{REGION}.modal.run/v1"


def api_key() -> str:
    k = os.environ.get("GRAFT_VLLM_API_KEY")
    if k:
        return k
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    raise RuntimeError(f"no endpoint key: set GRAFT_VLLM_API_KEY or write {KEY_FILE}")


def client(server: str, timeout: float = 600.0):
    from openai import OpenAI

    return OpenAI(base_url=base_url(server), api_key=api_key(), timeout=timeout, max_retries=3)


def inspect_model_name(server: str) -> str:
    """An Inspect `openai-api/<service>/<model>` name, after exporting the service's BASE_URL and API_KEY."""
    service = f"graft{server}"
    os.environ[f"{service.upper()}_BASE_URL"] = base_url(server)
    os.environ[f"{service.upper()}_API_KEY"] = api_key()
    return f"openai-api/{service}/{MODELS[server]}"
