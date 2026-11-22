"""Generations and scored checks for an edit, measured before, after, and under the two random controls.

    generate(model, tok, "Who was the mother of Jesus?")                 # greedy, repetition_penalty=1.0
    word_counts(reply, ["mother", "mom"], ["father", "dad"])              # how often each side of the concept appears
    choice_margin(model, tok, "Mary or Joseph?", "Mary", "Joseph")        # log p(Mary) − log p(Joseph), in nats
    probe = AxisProbe.fit(model, tok, axes)                               # difference of means on the axis's own pairs
    probe.kept(model, tok)                                                # 1 intact, 0 fused, −1 swapped
    compare(editor, tok, axes, units, prompts=[...], checks=[...])         # the whole before / after / controls table

`compare` runs four columns: before (the model as it is), after (with the edit), random axis (the same units edited
along a random axis) and random units (as many units per layer, at random, along the real axis). An edit's effect is
only interesting where it beats both controls. Weights are restored after each column, and the function checks that
the logits of a probe prompt are bit-identical to the start before it returns (`restored`).

Importing this module does not import torch.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from steerlab.edits.reflection import (
    FRAMES,
    HEAD_PARTS,
    Axes,
    Editor,
    Sides,
    Units,
    capture,
    chat_ids,
    random_axes_like,
    random_units_like,
    word_positions,
)

if TYPE_CHECKING:
    import torch

MAX_NEW_TOKENS = 48
COLUMNS = ("before", "after", "random_axis", "random_units")


def _device(model: Any) -> Any:
    return next(model.parameters()).device


def prompt_ids(tokenizer: Any, prompt: str, chat: bool = True) -> list[int]:
    if chat and getattr(tokenizer, "chat_template", None):
        return chat_ids(tokenizer, prompt)[0]
    ids: list[int] = tokenizer(prompt, add_special_tokens=True).input_ids
    return ids


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    chat: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
) -> str:
    """One reply. Greedy by default, always with repetition_penalty=1.0 (Qwen's generation config sets 1.1, which
    greedy decoding alone leaves on). `sample=True` draws with `temperature` / `top_p` from a fixed `seed`, so the
    before and after columns draw from the same random stream."""
    import torch

    ids = prompt_ids(tokenizer, prompt, chat)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    kwargs: dict[str, Any] = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    if sample:
        torch.manual_seed(seed)
        kwargs = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": 0}
    device = _device(model)
    with torch.no_grad():
        out = model.generate(
            torch.tensor([ids], device=device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
            pad_token_id=pad,
            **kwargs,
        )
    return str(tokenizer.decode(out[0, len(ids) :], skip_special_tokens=True)).strip()


def word_counts(text: str, source: Sequence[str], target: Sequence[str]) -> dict[str, Any]:
    """Whole-word, case-insensitive counts of the source and target words in `text`, and a score in [−1, 1]:
    +1 every match is a source word, −1 every match is a target word, 0 balanced or none."""

    def count(words: Sequence[str]) -> int:
        total = 0
        for w in words:
            w = w.strip()
            if w:
                total += len(re.findall(rf"(?<!\w){re.escape(w)}(?!\w)", text, flags=re.IGNORECASE))
        return total

    a, b = count(source), count(target)
    return {"source": a, "target": b, "score": (a - b) / (a + b) if a + b else 0.0}


def sequence_logprob(model: Any, context: Sequence[int], answer: Sequence[int]) -> float:
    import torch

    ids = torch.tensor([list(context) + list(answer)], device=_device(model))
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].float()
    logp = torch.log_softmax(logits[len(context) - 1 : -1], -1)
    picked = logp[torch.arange(len(answer), device=logp.device), torch.tensor(list(answer), device=logp.device)]
    return float(picked.sum())


def choice_margin(model: Any, tokenizer: Any, question: str, a: str, b: str, chat: bool = True) -> float:
    """log p(answer a) − log p(answer b) as the start of the reply, in nats (positive: the model prefers a)."""
    context = prompt_ids(tokenizer, question, chat)
    ia = tokenizer(a, add_special_tokens=False).input_ids
    ib = tokenizer(b, add_special_tokens=False).input_ids
    if not ia or not ib:
        raise ValueError("both answers need at least one token")
    return sequence_logprob(model, context, ia) - sequence_logprob(model, context, ib)


@dataclass
class AxisProbe:
    """A difference-of-means probe on the axis's own sentence pairs at one layer's output. `kept` is the separation
    between the two sides under the current weights, over the separation when the probe was fitted: 1 the concept
    pair is intact, 0 fused, −1 swapped (neuron-reflection's 'kept')."""

    layer: int
    direction: torch.Tensor
    pairs: list[tuple[str, str]]
    baseline: float

    @staticmethod
    def sentences(axes: Axes) -> list[tuple[str, str]]:
        source = axes.source
        if source.get("kind") == "random":
            source = source.get("like") or {}
        pairs = [(str(p[0]), str(p[1])) for p in source.get("pairs", [])]
        if source.get("kind") == "pairs":
            return [(f.format(w=a), f.format(w=b)) for a, b in pairs for f in FRAMES]
        if source.get("kind") == "sentences":
            return pairs
        return []

    @classmethod
    def fit(cls, model: Any, tokenizer: Any, axes: Axes, layer: int | None = None) -> AxisProbe | None:
        pairs = cls.sentences(axes)
        if not pairs:
            return None
        layer = len(axes.layers()) // 2 if layer is None else layer
        a, b = _sides(model, tokenizer, pairs, layer)
        direction = a.mean(0) - b.mean(0)
        direction = direction / direction.norm().clamp_min(1e-12)
        sep = float((a @ direction).mean() - (b @ direction).mean())
        return cls(layer, direction, pairs, sep)

    def kept(self, model: Any, tokenizer: Any) -> float:
        a, b = _sides(model, tokenizer, self.pairs, self.layer)
        sep = float((a @ self.direction).mean() - (b @ self.direction).mean())
        return sep / self.baseline if self.baseline else 0.0


def _sides(model: Any, tokenizer: Any, pairs: Sequence[tuple[str, str]], layer: int) -> tuple[Any, Any]:
    import torch

    rows_a, rows_b = [], []
    for sa, sb in pairs:
        x = tokenizer(sa, add_special_tokens=True).input_ids
        y = tokenizer(sb, add_special_tokens=True).input_ids
        px, py = word_positions(x, y)
        rows_a.append(capture(model, x, px)[1][layer].mean(0))
        rows_b.append(capture(model, y, py)[1][layer].mean(0))
    return torch.stack(rows_a), torch.stack(rows_b)


def run_checks(
    model: Any,
    tokenizer: Any,
    replies: list[str],
    checks: Sequence[dict[str, Any]],
    probe: AxisProbe | None,
    chat: bool = True,
) -> list[dict[str, Any]]:
    """Score one column. A check is {"kind": "words", "source": [...], "target": [...]} (over the replies),
    {"kind": "choice", "question", "a", "b"} (a margin in nats) or {"kind": "probe"} (the axis probe's kept)."""
    out: list[dict[str, Any]] = []
    for check in checks:
        kind = check.get("kind")
        if kind == "words":
            counts = [word_counts(r, check.get("source", []), check.get("target", [])) for r in replies]
            src, tgt = sum(c["source"] for c in counts), sum(c["target"] for c in counts)
            value = (src - tgt) / (src + tgt) if src + tgt else 0.0
            out.append({"kind": "words", "source": src, "target": tgt, "value": value})
        elif kind == "choice":
            value = choice_margin(model, tokenizer, check["question"], check["a"], check["b"], chat)
            out.append({"kind": "choice", **{key: check[key] for key in ("question", "a", "b")}, "value": value})
        elif kind == "probe":
            kept = probe.kept(model, tokenizer) if probe else None
            out.append({"kind": "probe", "layer": probe.layer if probe else None, "value": kept})
        else:
            raise ValueError(f"unknown check kind {kind!r}; use words, choice or probe")
    return out


def fingerprint(model: Any, ids: Sequence[int]) -> torch.Tensor:
    import torch

    with torch.no_grad():
        return model(torch.tensor([list(ids)], device=_device(model)), use_cache=False).logits.detach().clone()  # type: ignore[no-any-return,unused-ignore]


class Edited(AbstractContextManager[dict[str, Any] | None]):
    """Apply one reflection (or nothing) for a block with its own saved copies, so edits already in place (the rest
    of the stack) stay; restore only what this block touched."""

    def __init__(
        self, editor: Editor, axes: Axes | None, units: Units | None, k: float, sides: Sides, parts: str
    ) -> None:
        self.editor, self.axes, self.units, self.k, self.sides, self.parts = editor, axes, units, k, sides, parts
        self.child: Editor | None = None

    def __enter__(self) -> dict[str, Any] | None:
        if self.axes is None or self.units is None:
            return None
        self.child = Editor(self.editor.model)
        self.child._gains = self.editor._gains
        return self.child.reflect(self.axes, self.units, self.k, self.sides, self.parts)

    def __exit__(self, *exc: object) -> None:
        if self.child is not None:
            self.child.restore()


def compare(
    editor: Editor,
    tokenizer: Any,
    axes: Axes,
    units: Units,
    *,
    prompts: Sequence[str],
    checks: Sequence[dict[str, Any]] = (),
    k: float = 2.0,
    sides: Sides = "both",
    parts: str = HEAD_PARTS,
    controls: bool = True,
    seed: int = 0,
    chat: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sample: bool = False,
    temperature: float = 0.8,
    base: Callable[[], AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    """The before / after / random-axis / random-units table for one edit. `base` (optional) applies other edits
    first in every column (the rest of the edit stack) and restores them afterwards."""
    editor.check(units)
    model = editor.model
    base_cm = base or nullcontext
    fingerprint_ids = prompt_ids(tokenizer, prompts[0] if prompts else "Hello", chat)[:32]
    start = fingerprint(model, fingerprint_ids)
    probe = None
    if any(c.get("kind") == "probe" for c in checks):
        with base_cm():
            probe = AxisProbe.fit(model, tokenizer, axes)

    conditions: list[tuple[str, Axes | None, Units | None]] = [("before", None, None), ("after", axes, units)]
    if controls:
        conditions.append(("random_axis", random_axes_like(axes, seed), units))
        conditions.append(("random_units", axes, random_units_like(editor, units, seed)))

    columns: dict[str, Any] = {}
    for name, ax, us in conditions:
        with base_cm(), Edited(editor, ax, us, k, sides, parts) as info:
            replies = [
                generate(
                    model,
                    tokenizer,
                    p,
                    chat=chat,
                    max_new_tokens=max_new_tokens,
                    sample=sample,
                    temperature=temperature,
                    seed=seed,
                )
                for p in prompts
            ]
            columns[name] = {
                "replies": replies,
                "checks": run_checks(model, tokenizer, replies, checks, probe, chat),
                "info": info,
                "units": us.ids()[:400] if us is not None and name == "random_units" else None,
            }
    restored = bool(torch_equal(fingerprint(model, fingerprint_ids), start))
    return {
        "prompts": list(prompts),
        "columns": columns,
        "order": [c for c in COLUMNS if c in columns],
        "restored": restored,
        "exact": editor.exact,
        "k": k,
        "sides": sides,
        "parts": parts,
        "units": units.count(),
        "seed": seed,
        "sample": sample,
    }


def torch_equal(a: Any, b: Any) -> bool:
    import torch

    return bool(torch.equal(a, b))
