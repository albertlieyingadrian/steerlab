"""The steering engine: forward hooks that add the control state's vectors at generated positions only.

A decoder layer's output is a tensor `[batch, seq, d_model]`. The hook on layer L adds the state's summed vector for L
to every position whose *absolute* index is at or past the prompt length. Absolute means "counted from the first prompt
token", which is what makes one rule serve two very different forward passes:

- the cached decode loop feeds one token per pass, so the pass's position 0 is really position `cache length`;
- a no-cache teacher-forced pass feeds the whole sequence at once, so position i really is position i.

`SteeringSession.position_offset` is the absolute index of the pass's first position (the decode loop sets it from the
cache's seen-token count; a no-cache pass uses 0) and `prompt_length` is where steering starts. Prompt positions,
including Gemma's BOS at position 0, are never touched, so the prompt's keys and values are exactly the unsteered
model's. Strength 0 adds a zero vector, which is bit-for-bit the unsteered output.

One hook per application layer, registered through sandbagkit's `register_hooks` (so they are always removed) and
re-registered whenever the control state's version changes. The hook returns a tensor because transformers 5 decoder
layers return tensors (`replace_layer_output` keeps the tuple shape for models that still return one).
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from contextlib import ExitStack
from types import ModuleType, TracebackType
from typing import Any

import torch
from sandbagkit.probing.hooks import layer_output_tensor, num_layers, register_hooks, replace_layer_output

from steerlab.control import Clamp, ControlState, DirectionSource, SAESource, SourceUnavailableError
from steerlab.saes.common import FeatureDirection, LoadedSAE

Hook = Callable[[Any, Any, Any], Any]


def on_cpu(direction: FeatureDirection) -> FeatureDirection:
    """The same direction with its unit on CPU in float32, so units from any source can be summed together.

    An SAE's decoder lives on the model's device while a saved `--direction` vector loads on CPU; `ControlState.apply`
    adds them and torch refuses to add across devices. `SteeringSession.sync` moves the summed delta back anyway.
    """
    return dataclasses.replace(direction, unit=direction.unit.detach().to(device="cpu", dtype=torch.float32))


class DirectionRegistry:
    """What is loaded and steerable: SAE layers (by release and layer) and named directions. A `control.Resolver`.

    Feature directions are cached once computed, so steering a known feature never touches the SAE weights again.
    With `model` set, a feature whose SAE is not loaded is looked up in the feature cache (`features contrast` and
    `features search` store what they touch), so `chat` can steer a known feature without 537 MB of SAE in memory.
    Every direction handed out is on CPU in float32, whatever device its source sits on.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.saes: dict[tuple[str, int], LoadedSAE] = {}
        self.directions: dict[str, FeatureDirection] = {}
        self._features: dict[tuple[str, int, int], FeatureDirection] = {}

    def add_sae(self, sae: LoadedSAE) -> None:
        self.saes[(sae.release, sae.layer)] = sae

    def add_direction(self, name: str, direction: FeatureDirection) -> None:
        self.directions[name] = on_cpu(direction)

    @property
    def sae_layers(self) -> list[int]:
        return sorted({layer for _, layer in self.saes})

    def __call__(self, source: SAESource | DirectionSource) -> FeatureDirection:
        if isinstance(source, DirectionSource):
            direction = self.directions.get(source.name)
            if direction is None:
                loaded = ", ".join(sorted(self.directions)) or "none"
                raise SourceUnavailableError(
                    f"direction {source.name!r} is not loaded (loaded: {loaded}); "
                    f"restart with --direction {source.name}=<path to its saved vector>"
                )
            return direction
        key = (source.release, source.layer, source.feature)
        if key not in self._features:
            self._features[key] = on_cpu(self._resolve_feature(source))
        return self._features[key]

    def clamp(self, source: SAESource) -> tuple[torch.Tensor, float, float, torch.Tensor, str]:
        """(encoder row, bias, threshold, raw decoder vector, label) of a feature, on CPU in float32, for clamping.

        Clamping reads the feature's activation, so it needs the SAE itself; the feature cache only keeps directions.
        """
        sae = self.saes.get((source.release, source.layer))
        if sae is None:
            raise SourceUnavailableError(
                f"clamping reads the feature's activation, so SAE layer {source.layer} of {source.release} must be "
                f"loaded; restart with --sae-layer {source.layer}"
            )
        if not 0 <= source.feature < sae.n_features:
            raise SourceUnavailableError(
                f"feature {source.feature} out of range for an SAE with {sae.n_features} features"
            )
        row, bias, threshold = sae.encoder_probe(source.feature)
        decoder = sae.decoder_vector(source.feature).detach().to("cpu", torch.float32)
        return row.detach().to("cpu", torch.float32), bias, threshold, decoder, source.describe()

    def _resolve_feature(self, source: SAESource) -> FeatureDirection:
        """The feature from its loaded SAE, else from the feature cache, else the message naming the flag to load it."""
        sae = self.saes.get((source.release, source.layer))
        if sae is not None:
            try:
                return sae.direction(source.feature)
            except IndexError as error:
                raise SourceUnavailableError(str(error)) from None
        cache = _feature_cache() if self.model is not None else None
        if cache is not None:
            cached: FeatureDirection | None = cache.direction(self.model, source.release, source.layer, source.feature)
            if cached is not None:
                return cached
        raise SourceUnavailableError(
            f"SAE layer {source.layer} of {source.release} is not loaded and feature {source.feature} is not in the "
            f"feature cache, so it cannot be resolved; restart with --sae-layer {source.layer} to load it"
        )


def _feature_cache() -> ModuleType | None:
    """`steerlab.features.cache` when this install has it, else None. A missing dependency inside it still raises."""
    try:
        return importlib.import_module("steerlab.features.cache")
    except ModuleNotFoundError as error:
        if error.name in ("steerlab.features", "steerlab.features.cache"):
            return None
        raise


class SteeringSession:
    """Hooks for one model and one ControlState. Use as a context manager; `sync()` after the state changes.

        with SteeringSession(model, tokenizer, state) as session:
            session.prompt_length = prompt_ids.shape[1]
            ...one forward per token, setting session.position_offset = cache.get_seq_length() first...

    `last_stats` holds, for the most recent forward pass, `{layer: (added_norm, median_residual_norm)}` over the
    positions that were steered (empty when none were, e.g. during prefill).
    """

    def __init__(self, model: Any, tokenizer: Any, state: ControlState, *, layers_path: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.state = state
        self.layers_path = layers_path
        self.model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
        self.n_layers = num_layers(model, layers_path)
        if state.n_layers != self.n_layers:
            raise ValueError(f"the control state expects {state.n_layers} layers but the model has {self.n_layers}")
        self.prompt_length = 0
        self.position_offset = 0
        self.last_stats: dict[int, tuple[float, float]] = {}
        self._stack: ExitStack | None = None
        self._synced_version = -1
        self._deltas: dict[int, torch.Tensor] = {}
        self._clamps: dict[int, list[Clamp]] = {}

    @property
    def device(self) -> torch.device:
        device: torch.device = next(self.model.parameters()).device
        return device

    def __enter__(self) -> SteeringSession:
        self._stack = ExitStack()
        self.sync()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._synced_version = -1

    def sync(self) -> bool:
        """Re-register the hooks if the state changed since the last sync. Returns True when it did."""
        if self._stack is None:
            raise RuntimeError("SteeringSession must be entered (`with session:`) before it steers")
        if self.state.version == self._synced_version:
            return False
        self._stack.close()
        self._stack = ExitStack()
        parameter = next(self.model.parameters())
        self._deltas = {
            layer: delta.to(device=parameter.device, dtype=parameter.dtype)
            for layer, delta in self.state.deltas.items()
        }
        self._clamps = {
            layer: [
                dataclasses.replace(
                    clamp,
                    encoder=clamp.encoder.to(parameter.device, torch.float32),
                    decoder=clamp.decoder.to(parameter.device, torch.float32),
                )
                for clamp in clamps
            ]
            for layer, clamps in self.state.clamps.items()
        }
        hooks = {layer: self._make_hook(layer) for layer in self.state.layers}
        if hooks:
            self._stack.enter_context(register_hooks(self.model, hooks, layers_path=self.layers_path))
        self._synced_version = self.state.version
        return True

    def _make_hook(self, layer: int) -> Hook:
        def hook(module: Any, inputs: Any, output: Any) -> Any:
            hidden = layer_output_tensor(output)
            first = self.prompt_length - self.position_offset  # first steered position within this pass
            if first >= hidden.shape[1]:
                return output  # every position in this pass belongs to the prompt
            first = max(first, 0)
            steered = hidden.clone()
            delta = self._deltas.get(layer)
            if delta is not None:
                steered[:, first:] = hidden[:, first:] + delta.to(device=hidden.device, dtype=hidden.dtype)
            if layer in self._clamps:
                steered[:, first:] = self._clamped(steered[:, first:], self._clamps[layer])
                added = float((steered[:, first:] - hidden[:, first:]).float().norm(dim=-1).median())
            else:
                added = float(delta.float().norm()) if delta is not None else 0.0
            self.last_stats[layer] = (
                added,
                float(hidden[:, first:].float().norm(dim=-1).median()),
            )
            return replace_layer_output(output, steered)

        return hook

    @staticmethod
    def _clamped(hidden: torch.Tensor, clamps: list[Clamp]) -> torch.Tensor:
        """Each clamped feature's activation moved to its target: h + (target - activation(h)) × decoder."""
        h = hidden.float()
        out = h.clone()
        for clamp in clamps:
            pre = h @ clamp.encoder + clamp.bias
            activation = pre * (pre > clamp.threshold).to(pre.dtype)
            out = out + (clamp.target - activation).unsqueeze(-1) * clamp.decoder
        return out.to(hidden.dtype)
