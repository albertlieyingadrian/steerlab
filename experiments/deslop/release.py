"""Publish the round-2 LoRA (merged weights + adapter, and the GGUF) to Hugging Face with the model card.

    python release.py --export   # merge, GGUF and llama.cpp smoke test on Modal -> results/export.json
    python release.py            # upload what --export wrote

The token is read from $HF_TOKEN or the Hugging Face cache (~/.cache/huggingface/token) and never printed.
Writes results/release.json with the repo URLs.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = "albertlieadrian/qwen3-4b-deslop"
VARIANTS = {
    REPO: ("transformers", "This repo holds the merged bfloat16 weights and, in `adapter/`, the LoRA adapter."),
    REPO + "-GGUF": ("gguf", "This repo holds the GGUF build (Q4_K_M, 2.5 GB) for llama.cpp and Ollama."),
}


def token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()


def main() -> None:
    if "--export" in sys.argv:
        out = modal.Function.from_name("deslop", "export").remote("deslop-r2b", release="qwen3-4b-deslop")
        (HERE / "results" / "export.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"Q4_K_M {out['q4_bytes'] / 1e9:.2f} GB, llama.cpp smoke test exit code {out['smoke_rc']}")
        return
    template = (HERE / "model_card.md").read_text(encoding="utf-8")
    cards = {
        repo: template.replace("{library}", lib).replace("{variant}", note) for repo, (lib, note) in VARIANTS.items()
    }
    upload = modal.Function.from_name("deslop", "upload")
    urls = upload.remote("deslop-r2b", REPO, token(), cards)
    stamp = datetime.now().isoformat(timespec="seconds")
    (HERE / "results" / "release.json").write_text(json.dumps({"repos": urls, "at": stamp}, indent=1) + "\n")
    print("\n".join(urls))


if __name__ == "__main__":
    main()
