"""The supported models, their SAE releases and default layers, the judge, and approximate download sizes.

Light module: no torch. Everything here is a constant so `steerlab doctor` can describe the setup offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

QWEN_MODEL = "Qwen/Qwen3-1.7B"
GEMMA_MODEL = "google/gemma-3-1b-it"
JUDGE_MODEL = "Qwen/Qwen3-8B"
"""The judge for `sweep judge` and on-demand Qwen feature labels; only needed for those two jobs."""

MODELS: dict[str, str] = {"qwen": QWEN_MODEL, "gemma": GEMMA_MODEL}

# Larger models for directions, `chat`, `generate` and `serve` (no sweep or doctor row; EXTRA_SAES says which have
# an SAE): key -> Hub id, and the default layer, about halfway down (where Gemma Scope and Goodfire have SAEs).
EXTRA_MODELS: dict[str, str] = {
    "qwen8b": "Qwen/Qwen3-8B",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma2-9b": "google/gemma-2-9b-it",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
}
EXTRA_LAYERS: dict[str, int] = {"qwen8b": 18, "gemma2-2b": 12, "gemma2-9b": 20, "llama8b": 19}


class ExtraSAE(NamedTuple):
    """One SAE layer for an extra model: the Hub release, the layer, the file inside it, and Neuronpedia's names
    (None when Neuronpedia has no labels for it)."""

    release: str
    layer: int
    file: str
    neuronpedia_model: str | None = None
    neuronpedia_source: str | None = None


# Gemma Scope (v1, `params.npz`) at Neuronpedia's canonical L0 for the layer; Neuronpedia labels both.
# gemma-scope-2b-pt-res was trained on the base gemma-2-2b; gemma-scope-9b-it-res on gemma-2-9b-it itself.
# Qwen-Scope 8B (64K features, trained on the base Qwen3-8B) has no Neuronpedia labels. Goodfire's Llama-3.1-8B SAE
# (64K features, ReLU, trained on the instruct model at layer 19) is labelled on Goodfire's own platform, not here.
EXTRA_SAES: dict[str, ExtraSAE] = {
    "qwen8b": ExtraSAE("Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", 18, "layer18.sae.pt"),
    "llama8b": ExtraSAE("Goodfire/Llama-3.1-8B-Instruct-SAE-l19", 19, "Llama-3.1-8B-Instruct-SAE-l19.pth"),
    "gemma2-2b": ExtraSAE(
        "google/gemma-scope-2b-pt-res",
        12,
        "layer_12/width_16k/average_l0_82/params.npz",
        "gemma-2-2b",
        "gemmascope-res-16k",
    ),
    "gemma2-9b": ExtraSAE(
        "google/gemma-scope-9b-it-res",
        20,
        "layer_20/width_16k/average_l0_91/params.npz",
        "gemma-2-9b-it",
        "gemmascope-res-16k",
    ),
}


def extra_key(model: str) -> str | None:
    """The extra-model key for a key or Hub id (`gemma2-2b` or `google/gemma-2-2b-it`), else None."""
    if model in EXTRA_MODELS:
        return model
    return next((key for key, repo_id in EXTRA_MODELS.items() if repo_id == model), None)


GATED_REPOS: frozenset[str] = frozenset(
    {GEMMA_MODEL, EXTRA_MODELS["gemma2-2b"], EXTRA_MODELS["gemma2-9b"], EXTRA_MODELS["llama8b"]}
)
"""Repos behind a licence click on the Hub. steerlab refuses to download them until `auth_check` passes."""

# Qwen-Scope: TopK SAEs trained on the BASE model, one `layer{n}.sae.pt` per layer, 28 layers, k = 50
# (the "L0_50" in the release name; the files store no k).
QWEN_SCOPE_RELEASE = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
QWEN_SCOPE_LAYER = 14
# Qwen3-8B: the smallest Qwen3 Tinker trains that has Qwen-Scope SAEs (64K features
# per layer, top-k 50, all 36 layers, trained on the BASE model). The reconstruction check compares a quarter, half
# and three quarters of the depth; the two best are downloaded (about 2.1 GB each in float32).
QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
QWEN_SCOPE_8B_RELEASE = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
QWEN_SCOPE_8B_LAYERS = (9, 18, 27)
# Gemma Scope 2: JumpReLU SAEs trained on the instruct model, with Neuronpedia labels at layers 7/13/17/22.
GEMMA_SCOPE_RELEASE = "google/gemma-scope-2-1b-it"
GEMMA_SCOPE_LAYER = 13


def template_kwargs_for(model_name_or_type: str) -> dict[str, object]:
    """Extra chat-template variables per model, the one rule every caller shares.

    Qwen3 renders an empty, closed think block when `enable_thinking=False`, so the model answers at once; Gemma
    needs nothing. Accepts a Hub id ("Qwen/Qwen3-1.7B"), a config `model_type` ("qwen3") or a key ("qwen").
    """
    return {"enable_thinking": False} if "qwen3" in model_name_or_type.lower() else {}


_GB = 1024**3
_MB = 1024**2

APPROX_DOWNLOAD_BYTES: dict[str, int] = {
    QWEN_MODEL: int(4.0 * _GB),
    GEMMA_MODEL: int(2.0 * _GB),
    JUDGE_MODEL: int(16.4 * _GB),
    QWEN_SCOPE_RELEASE: 537 * _MB,
    GEMMA_SCOPE_RELEASE: 151 * _MB,
}
"""Rough sizes so `doctor` can say what a download would cost without asking the Hub."""


DIRECTIONS_ENV = "STEERLAB_DIRECTIONS"
"""Environment variable naming the folder of saved directions (`<model>/<concept>.npz/.json`)."""


def directions_dir() -> Path:
    """Where directions are built and looked up by default: $STEERLAB_DIRECTIONS, else `outputs/directions`.

    Relative paths resolve from the directory you run steerlab in.
    """
    return Path(os.environ.get(DIRECTIONS_ENV) or "outputs/directions")
