"""Turn texts into activation vectors: batched, padding-aware, with optional chat templates and a disk cache.

    acts = extract_activations(model, tokenizer, texts, layers=[2, 14])      {2: [n, hidden], 14: [n, hidden]}

What happens for each batch:

    texts ──render──► token ids ──pad (right)──► ONE forward pass, hooks on every layer ──select──► vectors
                       │                            │                                    │
            chat template or raw text      attention_mask marks real tokens    "last" real token, "first",
            (+ an optional response)                                            "mean", or response tokens

Padding: a batch needs equal-length rows, so short texts get filler tokens. Fillers go on the RIGHT. With
causal attention a token only sees tokens before it, so fillers after the text can't change any real token,
and "last" means the last REAL token (from each text's length), not the last column. That's why a batched run
gives the same vectors as running texts one at a time.

Response tokens: the toolkit probed the last token of the prompt. To probe what the model does while
ANSWERING, pass `responses`: each text becomes prompt + response, and position "response_mean" averages the
response's tokens only. The response is tokenized on its own and appended, exactly as generated tokens would
follow the prompt. `skip_first_response_token=True` drops the first response token from that average (it is
often a formatting token rather than content).

Cache: extracting 64 prompts from a 0.5B model is quick, but from a 7B model, or 10,000 prompts, it isn't. With
`cache=ActivationCache(dir)`, each (model, token ids, layer, position) vector is stored as a `.npy` file and
reused. The key is the token ids themselves, so changing the template, the response or the tokenizer can't
return a stale vector.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import numpy.typing as npt

from sandbagkit.core.types import ModelInfo
from sandbagkit.probing.hooks import capture_layers, resolve_layers

if TYPE_CHECKING:
    import torch

Position = Literal["last", "first", "mean", "response_mean", "response_first"]
POSITIONS: tuple[str, ...] = get_args(Position)
_RESPONSE_POSITIONS = ("response_mean", "response_first")

Activations = dict[int, npt.NDArray[np.float32]]
"""`{layer: [n_texts, hidden]}`, what extract_activations returns."""


class NonFiniteActivationError(ValueError):
    """An activation came out NaN or infinite: usually float16 overflow, or a broken device kernel."""


# --- rendering texts into token ids -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoded:
    """One text as the model will see it. `response_start` is the index of the first response token, if any."""

    input_ids: tuple[int, ...]
    response_start: int | None = None

    def __len__(self) -> int:
        return len(self.input_ids)


def render_prompt(
    tokenizer: Any,
    text: str,
    *,
    chat_template: bool = False,
    system_prompt: str | None = None,
    template_kwargs: Mapping[str, object] | None = None,
) -> str:
    """The prompt string: `text` itself, or `text` as a user turn in the model's chat template.

    With the template, the string ends with the assistant header (`add_generation_prompt=True`), i.e. exactly
    where the model would start answering. `template_kwargs` are extra variables for the template, passed
    through unchanged (Qwen3: `{"enable_thinking": False}` turns its thinking block off).
    """
    if not chat_template:
        if system_prompt is not None:
            raise ValueError("system_prompt needs chat_template=True")
        if template_kwargs:
            raise ValueError("template_kwargs needs chat_template=True")
        return text
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("chat_template=True, but this tokenizer has no chat template")
    messages = [{"role": "user", "content": text}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return str(
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **dict(template_kwargs or {})
        )
    )


def encode(
    tokenizer: Any,
    text: str,
    *,
    response: str | None = None,
    chat_template: bool = False,
    system_prompt: str | None = None,
    max_length: int = 512,
    template_kwargs: Mapping[str, object] | None = None,
) -> Encoded:
    """Token ids for one text (and optional response), truncated to `max_length` tokens.

    Raw text gets the tokenizer's usual special tokens (`add_special_tokens=True`, like the toolkit). A chat
    template already contains them, so none are added on top.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    prompt = render_prompt(
        tokenizer, text, chat_template=chat_template, system_prompt=system_prompt, template_kwargs=template_kwargs
    )
    ids = list(tokenizer(prompt, add_special_tokens=not chat_template)["input_ids"])
    start = None
    if response is not None:
        start = len(ids)
        ids += list(tokenizer(response, add_special_tokens=False)["input_ids"])
        if start >= min(len(ids), max_length):
            raise ValueError(
                f"the response has no tokens left after truncating to max_length={max_length} "
                f"(the prompt alone is {start} tokens)"
            )
    if not ids:
        raise ValueError(f"text {text[:40]!r} tokenized to zero tokens")
    return Encoded(tuple(ids[:max_length]), start)


def _select_rows(
    hidden: torch.Tensor,
    lengths: Sequence[int],
    starts: Sequence[int | None],
    position: str,
    *,
    skip_first_response_token: bool = False,
) -> torch.Tensor:
    """One vector per row of `hidden` [batch, seq, dim], using only real (non-padding) tokens.

    With `skip_first_response_token`, "response_mean" averages from the SECOND response token on.
    """
    import torch

    rows = []
    for i, (length, start) in enumerate(zip(lengths, starts, strict=True)):
        if position == "last":
            rows.append(hidden[i, length - 1])
        elif position == "first":
            rows.append(hidden[i, 0])
        elif position == "mean":
            rows.append(hidden[i, :length].mean(dim=0))
        elif start is None:
            raise ValueError(f"position {position!r} needs responses")
        elif position == "response_first":
            rows.append(hidden[i, start])
        else:
            first = start + 1 if skip_first_response_token else start
            if first >= length:
                raise ValueError(
                    f"skip_first_response_token needs at least two response tokens, but text {i} of the batch "
                    f"has {length - start}"
                )
            rows.append(hidden[i, first:length].mean(dim=0))
    return torch.stack(rows)


# --- the cache --------------------------------------------------------------------------------------------------


class ActivationCache:
    """Activation vectors on disk, one `.npy` per (model, token ids, layer, position). Safe to delete any time.

    `hits` and `misses` count lookups since creation, so a script can show that a rerun did no model work.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model_id: str, encoded: Encoded, layer: int, position: str) -> str:
        payload = {
            "model": model_id,
            "ids": list(encoded.input_ids),
            "response_start": encoded.response_start,
            "layer": layer,
            "position": position,
        }
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.npy"

    def get(self, key: str) -> npt.NDArray[np.float32] | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        value: npt.NDArray[np.float32] = np.load(path, allow_pickle=False)
        return value

    def put(self, key: str, vector: npt.NDArray[np.float32]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, np.asarray(vector, dtype=np.float32), allow_pickle=False)
        os.replace(tmp, path)  # atomic: a reader never sees half a file

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*/*.npy")) if self.directory.exists() else 0


def model_revision(model: Any) -> str | None:
    """The commit of the loaded weights, if it can be known without the network.

    transformers v4 recorded it as `config._commit_hash`; v5 doesn't, so for a Hub model this reads it from the
    local cache's snapshot folder (`.../snapshots/<commit>/config.json`). None for local folders and
    config-built models.
    """
    config = model.config
    revision = getattr(config, "_commit_hash", None)
    if revision:
        return str(revision)
    name = getattr(config, "_name_or_path", "") or ""
    if not name or Path(name).exists():
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(name, "config.json")
    except Exception:  # an invalid repo id, or huggingface_hub missing: no revision, not an error
        return None
    if isinstance(cached, str) and Path(cached).parent.parent.name == "snapshots":
        return Path(cached).parent.name
    return None


def model_id_for(model: Any) -> str:
    """A cache key part naming the weights: `name@revision:dtype`. Empty for a model built from a config."""
    name = getattr(model.config, "_name_or_path", "") or ""
    if not name:
        return ""
    return f"{name}@{model_revision(model) or 'unknown'}:{str(model.dtype).removeprefix('torch.')}"


def model_info(model: Any) -> ModelInfo:
    """sandbagkit's ModelInfo for a loaded Hugging Face model, for run records."""
    config = model.config
    return ModelInfo(
        name=getattr(config, "_name_or_path", "") or type(model).__name__,
        provider="hf",
        revision=model_revision(model),
        dtype=str(model.dtype).removeprefix("torch."),
        device=str(model.device),
        num_parameters=sum(p.numel() for p in model.parameters()),
        supports_activations=True,
    )


# --- extraction -------------------------------------------------------------------------------------------------


def extract_activations(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    layers: int | Sequence[int],
    *,
    position: Position = "last",
    responses: Sequence[str] | None = None,
    chat_template: bool = False,
    system_prompt: str | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    cache: ActivationCache | None = None,
    model_id: str | None = None,
    layers_path: str | None = None,
    template_kwargs: Mapping[str, object] | None = None,
    skip_first_response_token: bool = False,
) -> Activations:
    """Activations at `layers` for every text, as `{layer: float32 array [len(texts), hidden]}`.

    - `position`: which token(s) become the vector. "last" (the toolkit's choice), "first", "mean" (all real
      tokens), or with `responses`, "response_mean" / "response_first".
    - `chat_template`: wrap each text as a user turn first (with an optional `system_prompt`, and optional
      `template_kwargs` passed to the template, e.g. `{"enable_thinking": False}` for Qwen3).
    - `skip_first_response_token`: with "response_mean", leave the first response token out of the average.
    - `batch_size`: texts per forward pass. The vectors don't depend on it (up to float rounding).
    - `cache` + `model_id`: reuse vectors computed before. `model_id` defaults to the model's name, revision
      and dtype, and must be given for a model built from a config (it has no name).

    Raises LayerIndexError for layers the model doesn't have, and NonFiniteActivationError on NaN/inf.
    """
    import torch

    if position not in POSITIONS:
        raise ValueError(f"unknown position {position!r}; choose from {', '.join(POSITIONS)}")
    if position in _RESPONSE_POSITIONS and responses is None:
        raise ValueError(f"position {position!r} needs responses=[...] (one per text)")
    if skip_first_response_token and position != "response_mean":
        raise ValueError(f"skip_first_response_token applies to position 'response_mean', not {position!r}")
    if responses is not None and len(responses) != len(texts):
        raise ValueError(f"{len(texts)} texts but {len(responses)} responses")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    wanted = resolve_layers(model, layers, layers_path)
    cache_model = ""
    if cache is not None:
        cache_model = model_id or model_id_for(model)
        if not cache_model:
            raise ValueError("this model has no name to key the cache on; pass model_id='...'")
    # The skip changes the vector, so it is part of the cache key; the plain position keeps its old key.
    cache_position = f"{position}:skip_first" if skip_first_response_token else position

    encoded = [
        encode(
            tokenizer,
            text,
            response=None if responses is None else responses[i],
            chat_template=chat_template,
            system_prompt=system_prompt,
            max_length=max_length,
            template_kwargs=template_kwargs,
        )
        for i, text in enumerate(texts)
    ]
    vectors: dict[int, list[npt.NDArray[np.float32]]] = {layer: [] for layer in wanted}
    slots: dict[int, dict[int, npt.NDArray[np.float32]]] = {layer: {} for layer in wanted}

    # 1. whatever the cache already has
    todo: list[int] = []
    for i, enc in enumerate(encoded):
        found = [
            None if cache is None else cache.get(cache.key(cache_model, enc, layer, cache_position)) for layer in wanted
        ]
        if any(value is None for value in found):
            todo.append(i)
            continue
        for layer, value in zip(wanted, found, strict=True):
            assert value is not None
            slots[layer][i] = value

    # 2. one forward pass per batch for the rest
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)  # its value never matters: padded positions are masked out
    device = model.device
    with capture_layers(model, wanted, layers_path=layers_path) as captured, torch.inference_mode():
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start : batch_start + batch_size]
            lengths = [len(encoded[i]) for i in batch]
            input_ids = torch.full((len(batch), max(lengths)), pad_id, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row, i in enumerate(batch):
                input_ids[row, : lengths[row]] = torch.tensor(encoded[i].input_ids, dtype=torch.long)
                attention_mask[row, : lengths[row]] = 1
            model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            starts = [encoded[i].response_start for i in batch]
            for layer in wanted:
                selected = (
                    _select_rows(
                        captured[layer], lengths, starts, position, skip_first_response_token=skip_first_response_token
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(selected).all():
                    raise NonFiniteActivationError(
                        f"NaN or inf in layer {layer} activations (texts {batch[0]}..{batch[-1]}); "
                        "if the model is in float16, load it in float32 or bfloat16"
                    )
                for row, i in enumerate(batch):
                    slots[layer][i] = selected[row]
                    if cache is not None:
                        cache.put(cache.key(cache_model, encoded[i], layer, cache_position), selected[row])

    result: Activations = {}
    for layer in wanted:
        vectors[layer] = [slots[layer][i] for i in range(len(encoded))]
        result[layer] = np.stack(vectors[layer]).astype(np.float32) if encoded else np.zeros((0, 0), np.float32)
    return result


def check_device_parity(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    layers: int | Sequence[int],
    *,
    reference_device: str = "cpu",
    **kwargs: Any,
) -> dict[int, float]:
    """Largest |difference| per layer between the model's own device and a copy on `reference_device`.

    Run it once per model: if a device's kernels misbehave, this shows it before any probe is trained. It
    copies the whole model, so it needs twice the memory.
    """
    reference = copy.deepcopy(model).to(reference_device)
    try:
        here = extract_activations(model, tokenizer, texts, layers, **kwargs)
        there = extract_activations(reference, tokenizer, texts, layers, **kwargs)
    finally:
        del reference
    return {layer: float(np.abs(here[layer] - there[layer]).max()) for layer in here}
