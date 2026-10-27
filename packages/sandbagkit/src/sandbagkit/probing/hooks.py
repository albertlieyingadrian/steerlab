"""Forward hooks: read (or change) what a transformer layer outputs, for many layers in ONE forward pass.

A forward hook is a callback PyTorch runs right after a module's forward():

    hook(module, inputs, output) -> None          just look (capture)
    hook(module, inputs, output) -> new_output    replace the output (steering)

    with capture_layers(model, [2, 14]) as captured:     one hook per requested layer
        model(**batch)                                   a single forward pass fills all of them
    captured[14]                                          tensor [batch, seq, hidden]
    # leaving the block removes every hook, even if the forward pass raised

Why guaranteed removal matters: a hook is global state on the model object. One left behind by a crashed
experiment keeps capturing (memory grows) or keeps steering (every later result is silently wrong). So every
hook here is registered inside a `try/finally` that removes it.

Layers return a tensor in transformers v5, but a tuple `(hidden, ...)` in v4 and some custom models, so
both are handled. Nothing here imports torch at module level: `import sandbagkit.probing` stays instant.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# Where common Hugging Face architectures keep their list of decoder layers, tried in order.
LAYER_PATHS: tuple[str, ...] = (
    "model.layers",  # Llama, Qwen2/3, Mistral, Gemma, Phi-3
    "model.language_model.layers",  # Gemma 3 and other multimodal wrappers
    "transformer.h",  # GPT-2, GPT-J, Falcon
    "gpt_neox.layers",  # Pythia, GPT-NeoX
    "model.decoder.layers",  # OPT
    "layers",  # a bare decoder stack
)

Hook = Callable[[Any, Any, Any], Any]


class LayerIndexError(IndexError, ValueError):
    """A requested layer doesn't exist in this model (or isn't a whole number)."""


def _get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def decoder_layers(model: Any, path: str | None = None) -> Any:
    """The model's decoder layers, in order (a `torch.nn.ModuleList`).

    `path` overrides the search, e.g. `"model.layers"`. Raises TypeError with the paths it tried when the
    architecture is unknown.
    """
    paths = (path,) if path is not None else LAYER_PATHS
    for candidate in paths:
        try:
            layers = _get_path(model, candidate)
        except AttributeError:
            continue
        if hasattr(layers, "__len__") and hasattr(layers, "__getitem__") and len(layers) > 0:
            return layers
    raise TypeError(
        f"can't find the decoder layers of {type(model).__name__}; tried {', '.join(paths)}. "
        "Pass layers_path='...' (the dotted attribute path of its ModuleList)."
    )


def num_layers(model: Any, path: str | None = None) -> int:
    return len(decoder_layers(model, path))


def resolve_layers(model: Any, layers: int | Sequence[int], path: str | None = None) -> list[int]:
    """Check layer indexes against the model and return them as a sorted, de-duplicated list.

    Negative indexes count from the end, as in Python lists (-1 is the last layer). Anything outside
    [-n, n - 1], or not an integer, raises LayerIndexError naming the valid range.
    """
    n = num_layers(model, path)
    requested = [layers] if isinstance(layers, int) else list(layers)
    if not requested:
        raise LayerIndexError("no layers requested")
    resolved: set[int] = set()
    for layer in requested:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise LayerIndexError(f"layer indexes must be integers, got {layer!r}")
        if not -n <= layer < n:
            raise LayerIndexError(
                f"layer {layer} doesn't exist: this model has {n} layers (valid: 0..{n - 1}, or -{n}..-1)"
            )
        resolved.add(layer % n)
    return sorted(resolved)


def layer_output_tensor(output: Any) -> torch.Tensor:
    """A layer's hidden states, whether it returned a tensor or a tuple `(hidden, ...)`."""
    hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
    return hidden


def replace_layer_output(output: Any, hidden: torch.Tensor) -> Any:
    """`output` with its hidden states swapped for `hidden`, keeping the same shape (tensor or tuple)."""
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def count_forward_hooks(model: Any) -> int:
    """How many forward hooks are registered anywhere in `model`. Tests use it to prove hooks were removed.

    Compare before and after, not with zero: transformers v5 registers some hooks of its own.
    """
    return sum(len(module._forward_hooks) for module in model.modules())


@contextmanager
def register_hooks(model: Any, hooks: dict[int, Hook], *, layers_path: str | None = None) -> Iterator[None]:
    """Register `{layer: hook}` on the decoder layers; remove ALL of them on exit, even after an error.

    Layer indexes are validated first (negative ones count from the end). If registering the third hook
    fails, the first two are still removed.
    """
    layer_modules = decoder_layers(model, layers_path)
    n = len(layer_modules)
    resolved = {}
    for layer, hook in hooks.items():
        resolved[resolve_layers(model, layer, layers_path)[0]] = hook
    if len(resolved) != len(hooks):
        raise LayerIndexError(f"two hooks map to the same layer (of {n}): {sorted(hooks)}")
    handles = []
    try:
        for layer, hook in resolved.items():
            handles.append(layer_modules[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_layers(
    model: Any, layers: int | Sequence[int], *, layers_path: str | None = None, detach: bool = True
) -> Iterator[dict[int, torch.Tensor]]:
    """Capture the outputs of `layers` during forward passes inside the block, as `{layer: [batch, seq, hidden]}`.

    Each forward pass overwrites the previous one's tensors, so read them after every call. Layer `i`'s output
    equals `model(..., output_hidden_states=True).hidden_states[i + 1]` (index 0 is the embeddings; the very
    last entry may have the final norm applied, depending on the architecture).
    """
    wanted = resolve_layers(model, layers, layers_path)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int) -> Hook:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            hidden = layer_output_tensor(output)
            captured[layer] = hidden.detach() if detach else hidden

        return hook

    with register_hooks(model, {layer: make_hook(layer) for layer in wanted}, layers_path=layers_path):
        yield captured
