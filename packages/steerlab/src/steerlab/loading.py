"""Load, unload and list models for steering, without exiting the process.

`steerlab chat` and `generate` load once and exit on any problem; `steerlab serve` keeps running and swaps models, so
it needs the same steps as plain functions: errors are `LoadError` (a `ValueError`), messages go to a callback, and
`unload` gives the memory back before the next model loads.

    backend, state = load_session("qwen8b", [18], None, discover_directions(root, "qwen8b"), on_message=print)
    ...
    unload(backend, state)

`catalog(root)` lists what a swap may pick: the models whose weights are already in the Hugging Face cache, each with
its saved directions and, when its SAE file is cached too, that SAE. With `local_only`, `load_session` loads from the
cached snapshot folder and refuses anything uncached, so the server can never start a download.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerlab import defaults, hub

if TYPE_CHECKING:
    from steerlab.control import ControlState


class LoadError(ValueError):
    """A model, SAE or direction could not be loaded. The message says what to do about it."""


def resolve_model(name: str) -> str:
    """`qwen`, `gemma` or an extra model key to its Hub id; anything else is used as given (a Hub id or a path)."""
    return defaults.MODELS.get(name) or defaults.EXTRA_MODELS.get(name, name)


def default_release(model_id: str) -> str | None:
    if model_id == defaults.QWEN_MODEL:
        return defaults.QWEN_SCOPE_RELEASE
    if model_id == defaults.GEMMA_MODEL:
        return defaults.GEMMA_SCOPE_RELEASE
    key = defaults.extra_key(model_id)
    return defaults.EXTRA_SAES[key].release if key in defaults.EXTRA_SAES else None


DEFAULT_SAE_LAYERS: dict[str, int] = {
    defaults.QWEN_MODEL: defaults.QWEN_SCOPE_LAYER,
    defaults.GEMMA_MODEL: defaults.GEMMA_SCOPE_LAYER,
    **{defaults.EXTRA_MODELS[key]: sae.layer for key, sae in defaults.EXTRA_SAES.items()},
}


def is_gemma_scope(release: str) -> bool:
    """Gemma Scope (v1 or 2), Goodfire and Qwen-Scope store their SAEs differently; the release name says which."""
    return "gemma-scope" in release.lower()


def is_goodfire(release: str) -> bool:
    return release.lower().startswith("goodfire/")


def sae_file(release: str, layer: int) -> str:
    """The file of one SAE layer inside its release, for checking the cache before loading."""
    from steerlab.saes import gemma_scope, goodfire, qwen_scope

    known = next((sae for sae in defaults.EXTRA_SAES.values() if sae.release == release and sae.layer == layer), None)
    if known is not None:
        return known.file
    if is_goodfire(release):
        return goodfire.filename(release)
    return gemma_scope.params_path(layer) if is_gemma_scope(release) else qwen_scope.filename(layer)


def fetch_sae(release: str, layer: int) -> Any:
    """One SAE layer of `release`, from the cache or the Hub, with the loader its format needs."""
    from steerlab.saes import gemma_scope, goodfire, qwen_scope

    if is_goodfire(release):
        return goodfire.fetch_goodfire(layer, release=release)
    if is_gemma_scope(release):
        return gemma_scope.fetch_gemma_scope(layer, release=release)
    return qwen_scope.fetch_qwen_scope(layer, release=release)


def discover_directions(root: str | Path, key: str) -> list[str]:
    """`NAME=PATH` specs for every saved direction under `<root>/<key>/` (an `.npz` with its `.json`), sorted."""
    folder = Path(root) / key
    if not folder.is_dir():
        return []
    return [
        f"{npz.stem}={npz.with_suffix('')}"
        for npz in sorted(folder.glob("*.npz"))
        if npz.with_suffix(".json").is_file()
    ]


def local_snapshot(model_id: str) -> Path | None:
    """The cached snapshot folder holding `model_id`'s config and weights, or None when it is not fully cached."""
    snapshots = hub.repo_cache_dir(model_id) / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        has_config = (snapshot / "config.json").is_file()
        has_weights = any(path.is_file() for path in snapshot.glob("*.safetensors"))
        if has_config and has_weights:
            return snapshot
    return None


def load_session(
    model: str,
    sae_layers: list[int],
    sae_release: str | None,
    direction_specs: list[str],
    *,
    on_message: Callable[[str], None],
    on_progress: Callable[[str], None] | None = None,
    local_only: bool = False,
) -> tuple[Any, ControlState]:
    """Load the model (refusing a gated download this account cannot make), its SAE layers and directions.

    Every SAE and direction must have the model's residual width: a vector built on another model would only fail
    inside the hook, mid-generation. With `local_only`, the weights load from the cached snapshot folder and an
    uncached model or SAE is refused rather than downloaded. `on_message` receives the control state's messages;
    `on_progress` the "loaded ..." lines (default: `on_message`).
    """
    from sandbagkit.backends.hf import HFBackend
    from sandbagkit.probing.hooks import num_layers
    from sandbagkit.probing.steering import SteeringVector

    from steerlab import directions as direction_lib
    from steerlab.control import ControlState
    from steerlab.engine import DirectionRegistry

    progress = on_progress or on_message
    model_id = resolve_model(model)
    source = model_id
    if local_only:
        snapshot = local_snapshot(model_id)
        if snapshot is None:
            raise LoadError(f"{model_id} is not in the Hugging Face cache; download it first")
        source = str(snapshot)
    elif model_id in defaults.GATED_REPOS and hub.cached_bytes(model_id) is None:
        try:
            hub.ensure_access(model_id)
        except hub.GatedRepoAccessError as error:
            raise LoadError(str(error)) from None
    backend = HFBackend.from_pretrained(source)
    width = int(backend.model.config.hidden_size)
    registry = DirectionRegistry(model=model_id)
    release = sae_release or default_release(model_id)
    for layer in sae_layers:
        if release is None:
            raise LoadError(f"{model_id} has no default SAE release; pass --sae-release")
        if local_only and hub.cached_file(release, sae_file(release, layer)) is None:
            raise LoadError(f"{release} layer {layer} is not in the Hugging Face cache; download it first")
        sae = fetch_sae(release, layer)
        if int(sae.d_model) != width:
            raise LoadError(
                f"{release} layer {layer} has width {sae.d_model} but {model_id} has width {width}; "
                "it was trained on another model"
            )
        registry.add_sae(sae.to(backend.device))
        progress(f"loaded SAE {release} layer {layer} ({sae.n_features} features)")
    for spec in direction_specs:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise LoadError(f"expected NAME=PATH, got {spec!r}")
        vector = SteeringVector.load(path)
        if vector.layer is None:
            raise LoadError(f"{path} records no layer; it cannot default its own")
        if vector.vector.shape[0] != width:
            built_on = vector.metadata.get("model") or "an unknown model"
            raise LoadError(
                f"{path} has width {vector.vector.shape[0]} (built on {built_on}) but {model_id} has width {width}"
            )
        registry.add_direction(name, direction_lib.to_feature_direction(vector, default_label=name))
        progress(f"loaded direction {name!r} (layer {vector.layer}, norm {vector.norm:.3g}) from {path}")
    state = ControlState(model_id, num_layers(backend.model), registry, on_message=on_message)
    return backend, state


def unload(backend: Any, state: ControlState | None = None) -> None:
    """Drop the model, its SAEs and directions, and give the device memory back (curation/runs.py's order).

    The caller must have left its `SteeringSession` first (that removes the hooks) and must drop its own references
    to `backend` and `state` afterwards; this clears what the two objects hold so nothing else keeps the weights.
    """
    import torch

    device = str(backend.device)
    if state is not None:
        state.clear()
        registry = state.resolver
        for attribute in ("saes", "directions", "_features"):
            held = getattr(registry, attribute, None)
            if isinstance(held, dict):
                held.clear()
    backend.model = None
    gc.collect()
    if device.startswith("mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class SAEInfo:
    release: str
    layer: int


@dataclass(frozen=True)
class CatalogEntry:
    """One model a swap may pick: its key and Hub id, its saved directions (`NAME=PATH`), and its cached SAE."""

    key: str
    model_id: str
    direction_specs: list[str] = field(default_factory=list)
    sae: SAEInfo | None = None

    @property
    def direction_names(self) -> list[str]:
        return [spec.partition("=")[0] for spec in self.direction_specs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model": self.model_id,
            "directions": self.direction_names,
            "sae": None if self.sae is None else {"release": self.sae.release, "layer": self.sae.layer},
        }


def demo_and_saved_directions(directions_root: str | Path, key: str) -> list[str]:
    """The tuned directions shipped with steerlab for `key` (`steerlab.demo`), plus any other concept saved under
    `directions_root`; a shipped direction wins over a saved one with the same name."""
    from steerlab import demo

    shipped = discover_directions(demo.DIRECTIONS_ROOT, key)
    names = {spec.partition("=")[0] for spec in shipped}
    saved = [spec for spec in discover_directions(directions_root, key) if spec.partition("=")[0] not in names]
    return sorted(shipped + saved)


def catalog(directions_root: str | Path) -> list[CatalogEntry]:
    """The models whose weights are cached, in `defaults` order, with their directions (shipped and saved) and SAE."""
    entries = []
    for key, model_id in {**defaults.MODELS, **defaults.EXTRA_MODELS}.items():
        if local_snapshot(model_id) is None:
            continue
        sae = None
        release, layer = default_release(model_id), DEFAULT_SAE_LAYERS.get(model_id)
        if release is not None and layer is not None and hub.cached_file(release, sae_file(release, layer)):
            sae = SAEInfo(release, layer)
        entries.append(CatalogEntry(key, model_id, demo_and_saved_directions(directions_root, key), sae))
    return entries
