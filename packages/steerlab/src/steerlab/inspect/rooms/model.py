"""The Model room: which model is loaded, its shape, and what else steerlab has for it (lenses, catalog, SAEs).

    GET /api/model/info     id, family, layers, heads (query and key/value), widths, MLP type, parameters, and
                            availability of the logit/J/R/S lenses, the neuron-lens catalog and SAEs
    GET /api/model/cached   models in the local Hugging Face cache (restart with `--model <id>` to switch)

Switching models needs a restart: the loaded model is shared by every room under one lock.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

    from steerlab.inspect.rooms import RoomContext

ROOM = {"id": "model", "title": "Model", "order": 60, "key": "m"}

# Qwen3 models (post-trained and base) in the stdlogits/neuron-lens catalog
CATALOG_MODELS = frozenset(
    f"Qwen/Qwen3-{size}{suffix}" for size in ("0.6B", "1.7B", "4B", "8B", "14B") for suffix in ("", "-Base")
) | {"google/gemma-4-12B"}


def mlp_type(inspector: Any) -> str:
    """'SwiGLU' (gate and up projections, two signed poles per neuron) or a plain MLP, with its activation."""
    mlp = inspector.blocks[0].mlp
    act = str(getattr(inspector.model.config, "hidden_act", getattr(inspector.model.config, "hidden_activation", "")))
    gated = hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")
    if gated:
        return "SwiGLU" if act in ("silu", "swish") else f"gated ({act})"
    return f"MLP ({act})" if act else "MLP"


def saes_for(model_id: str) -> list[dict[str, Any]]:
    """The SAE releases steerlab can load for this model (`steerlab.saes`), from `steerlab.defaults`."""
    from steerlab import defaults

    rows: list[dict[str, Any]] = []
    if model_id.startswith("Qwen/Qwen3-1.7B"):
        rows.append({"release": defaults.QWEN_SCOPE_RELEASE, "layers": [defaults.QWEN_SCOPE_LAYER], "kind": "TopK"})
    if model_id.startswith("Qwen/Qwen3-8B"):
        layers = list(defaults.QWEN_SCOPE_8B_LAYERS)
        rows.append({"release": defaults.QWEN_SCOPE_8B_RELEASE, "layers": layers, "kind": "TopK"})
    if model_id == defaults.GEMMA_MODEL:
        layer = defaults.GEMMA_SCOPE_LAYER
        rows.append({"release": defaults.GEMMA_SCOPE_RELEASE, "layers": [layer], "kind": "JumpReLU"})
    key = defaults.extra_key(model_id)
    if key in defaults.EXTRA_SAES and not model_id.startswith("Qwen/Qwen3-8B"):
        sae = defaults.EXTRA_SAES[key]
        rows.append({"release": sae.release, "layers": [sae.layer], "kind": "JumpReLU" if "gemma" in key else "ReLU"})
    return rows


def lenses_for(ctx: RoomContext) -> dict[str, Any]:
    """Each lens and whether it is ready, downloadable or needs a fit. The logit lens is always ready."""
    try:
        jacobian = importlib.import_module("steerlab.lens.jacobian")
    except ImportError:
        return {"logit": {"status": "ready", "source": "the model's own final norm and unembedding"}}
    return jacobian.availability(ctx.model_id)  # type: ignore[no-any-return]


def describe(ctx: RoomContext) -> dict[str, Any]:
    inspector = ctx.inspector
    config = inspector.model.config
    n_kv = int(getattr(config, "num_key_value_heads", None) or inspector.n_heads)
    params = sum(p.numel() for p in inspector.model.parameters())
    first = next(inspector.model.parameters())
    gated = mlp_type(inspector).startswith(("SwiGLU", "gated"))
    return {
        "model": ctx.model_id,
        "family": str(getattr(config, "model_type", "")),
        "n_layers": inspector.n_layers,
        "n_heads": inspector.n_heads,
        "n_kv_heads": n_kv,
        "d_model": inspector.d_model,
        "d_head": inspector.d_head,
        "d_mlp": inspector.d_mlp,
        "vocab": inspector.vocab,
        "neurons": inspector.n_layers * inspector.d_mlp,
        "units": inspector.n_layers * inspector.d_mlp * (2 if gated else 1) + inspector.n_layers * inspector.n_heads,
        "heads_total": inspector.n_layers * inspector.n_heads,
        "mlp": mlp_type(inspector),
        "poles": 2 if gated else 1,
        "params": params,
        "dtype": str(first.dtype).removeprefix("torch."),
        "device": str(first.device),
        "tied_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "lenses": lenses_for(ctx),
        "catalog": {
            "source": "stdlogits/neuron-lens",
            "covered": ctx.model_id in CATALOG_MODELS,
            "models": sorted(CATALOG_MODELS),
        },
        "saes": saes_for(ctx.model_id),
        "rooms": ctx.rooms,
    }


def cached_models(hub: Path | None = None) -> list[str]:
    """Model ids in the Hugging Face hub cache (directory names only; nothing is opened)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    root = hub or Path(os.environ.get("HF_HUB_CACHE") or home / "hub")
    if not root.is_dir():
        return []
    dirs = (p.name for p in root.iterdir() if p.name.startswith("models--"))
    return sorted(name.removeprefix("models--").replace("--", "/", 1) for name in dirs)


def register(app: FastAPI, ctx: RoomContext) -> None:
    @app.get("/api/model/info")
    def info() -> dict[str, Any]:
        return ctx.view(lambda: describe(ctx))

    @app.get("/api/model/cached")
    def cached() -> dict[str, Any]:
        return {"loaded": ctx.model_id, "cached": cached_models()}
