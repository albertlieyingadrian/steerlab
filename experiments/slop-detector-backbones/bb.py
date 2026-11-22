"""Shared pieces for the backbone bake-off: arms, paths, eval sets and the spend ledger with its hard cap.

The slop-detector note's code is imported, never copied: its trainer (04_train.py), metrics (common.py), cut-off
fitting (diag_voi_gap.py), cross-fitting (windows.py), EditLens-val cleaning (tune_editlens_val.py) and the frozen
fresh set (fresh_test.py).
"""

from __future__ import annotations

import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
SD = HERE.parent / "slop-detector"
sys.path.insert(0, str(SD))

sd = import_module("common")  # the slop-detector note's common.py
RESULTS = HERE / "results"
RUNS = HERE / "runs"
# Data v2 relabelled with 4 buckets (the final 9B's training data). Text is never committed: it lives in the
# slop-detector checkout's gitignored data folder, or wherever SLOP_DATA points.
DATA = Path(os.environ.get("SLOP_DATA", SD / "data" / "v2b"))
LEDGER = RESULTS / "spend.jsonl"
CAP_USD = float(os.environ.get("BB_CAP_USD", "300"))  # $150 per round (round 2 is a pre-registered amendment)
VOLUME = "slop-detector"  # the slop-detector note's volume, in the albertlieyingadrian workspace
PROFILE = os.environ.get("MODAL_PROFILE", "albertlieyingadrian")

# Modal list prices per GPU-hour (2026-11 pricing page; the ledger is an estimate from wall time).
GPU_USD_H = {"H200": 4.54, "H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "A10G": 1.10, "L4": 0.80}

# New arms. kind: "lora" = the slop-detector recipe (LoRA r 16 on attention + MLP, head in full, bf16);
# "full" = full fine-tune (fp32 weights, bf16 autocast), the usual recipe for encoders.
ARMS = {
    "mistral-nemo-12b": {"base": "mistralai/Mistral-Nemo-Base-2407", "kind": "lora", "micro": 2048, "grad_ckpt": True},
    "llama-3.1-8b": {"base": "meta-llama/Llama-3.1-8B", "kind": "lora", "micro": 2048, "grad_ckpt": True},
    "modernbert-large": {"base": "answerdotai/ModernBERT-large", "kind": "full", "micro": 16384, "grad_ckpt": False},
    "deberta-v3-large": {"base": "microsoft/deberta-v3-large", "kind": "full", "micro": 16384, "grad_ckpt": False},
}
# Round 2, a pre-registered amendment to the plan, on H200s.
ARMS2 = {
    "gemma-3-12b": {
        "base": "google/gemma-3-12b-pt",
        "kind": "lora_mm",
        "micro": 2048,
        "grad_ckpt": True,
        "gpu": "H200",
    },
    "qwen3-14b-base": {"base": "Qwen/Qwen3-14B-Base", "kind": "lora", "micro": 2048, "grad_ckpt": True, "gpu": "H200"},
    "nemotron-30b-a3b": {
        "base": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "kind": "pool",
        "micro": 1024,
        "grad_ckpt": True,
        "gpu": "H200",
    },
}
ALL_ARMS = {**ARMS, **ARMS2}
# Learning-rate grids, chosen on validation only (sweep.py). The LoRA grid holds the card's 2e-4 that the Qwen
# reference used; the full fine-tune grid is the usual range for large encoders.
LR_GRID = {
    "lora": (1e-4, 2e-4, 4e-4),
    "lora_mm": (1e-4, 2e-4, 4e-4),
    "pool": (1e-4, 2e-4, 4e-4),
    "full": (1e-5, 2e-5, 5e-5),
}
SWEEP_ROWS = 16000  # rows of train for each sweep run (1 epoch); its val is the first 2,000 val rows
EPOCHS = 2.0  # as the Qwen3.5-9B reference
# Reference arms from the slop-detector note (adapters on the Modal volumes named here).
REFS = {"qwen3.5-9b": ("f9-9b", "albertlieyingadrian"), "qwen3.5-4b": ("b1x-4bk", "modal4")}

# Benchmark splits scored once per arm. "fresh" is the frozen one-look set; "ours_test" is data v2's own test split
# (for the generator-family check).
SPLITS = ("test", "test_enron", "test_llama", "human_detectors", "epoch", "editlens_val", "fresh", "ours_test")


def eval_texts(splits=SPLITS) -> dict[str, list[str]]:
    out = {}
    for s in splits:
        if s == "fresh":
            rows = import_module("fresh_test").fresh_rows()["text"]
        elif s == "ours_test":
            rows = [r["text"] for r in sd.read_jsonl(DATA / "test.jsonl")]
        else:
            rows = sd.benchmark(s)["text"]
        out[s] = [sd.empty_fences(str(t)) for t in rows]
    return out


def spent() -> float:
    return sum(r.get("usd", 0.0) for r in sd.read_jsonl(LEDGER))


def check_cap(planned_usd: float, what: str) -> None:
    """Refuse to start paid work that would take the estimated spend over the cap."""
    s = spent()
    if s + planned_usd > CAP_USD:
        raise SystemExit(f"cap: {what} needs ~${planned_usd:.0f}, spent ${s:.2f} of ${CAP_USD:.0f} (BB_CAP_USD)")


def log_spend(what: str, gpu: str, seconds: float, **extra) -> float:
    usd = round(GPU_USD_H.get(gpu, 3.95) * seconds / 3600, 3)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": what,
        "gpu": gpu,
        "seconds": round(seconds),
        "usd": usd,
        "profile": os.environ.get("MODAL_PROFILE", "default"),
    }
    rec.update(extra)
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return usd
