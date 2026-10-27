"""What the harness needs from steerlab, in one place: load a model, steer one reply, and measure it.

Measuring is deliberately simple and readable: the share of Chinese characters in a reply, and whether the reply is
still fluent (long enough, and not repeating itself). Everything is greedy, so a trial gives the same reply every run.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

OUTPUTS = Path(os.environ.get("STEERLAB_OUTPUTS") or "outputs")
"""Where runs write: $STEERLAB_OUTPUTS, else `outputs/` (relative to where you run). Gitignored."""

DISTINCT_COLLAPSE = 0.3
"""Below this share of distinct character 3-grams, a reply counts as collapsed (repeating itself)."""


# --- Chinese characters ------------------------------------------------------------------------------------------


def is_cjk_char(char: str) -> bool:
    """Han ideographs (the characters Chinese is written in): CJK Unified, Extension A, and Compatibility."""
    code = ord(char)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF


def cjk_share(text: str) -> float:
    """Chinese characters as a share of all non-whitespace characters (0 when the text is empty)."""
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    return sum(1 for c in visible if is_cjk_char(c)) / len(visible)


# --- fluency -----------------------------------------------------------------------------------------------------


def distinct_n(items: Sequence[Hashable], n: int = 2) -> float:
    """Distinct n-grams over total n-grams. A sequence too short to hold one n-gram is trivially distinct (1.0)."""
    seq = list(items)
    if len(seq) < n:
        return 1.0
    grams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
    return len(set(grams)) / len(grams)


def fluent(text: str) -> bool:
    """Not collapsed: at least 20 characters and character 3-gram diversity of at least 0.3."""
    chars = [c for c in text if not c.isspace()]
    return len(chars) >= 20 and distinct_n(chars, 3) >= DISTINCT_COLLAPSE


# --- models and steering -----------------------------------------------------------------------------------------


def load_model(model_key: str) -> tuple[Any, Any, str, int]:
    """(model, tokenizer, model id, default layer) for a steerlab model key (`steerlab doctor` lists them)."""
    from steerlab.directions import load_model as _load

    return _load(model_key)


def make_session(model: Any, tokenizer: Any, model_id: str) -> tuple[Any, Any]:
    """A steerlab `SteeringSession` and its `ControlState`; directions are registered on `state.resolver` later."""
    from sandbagkit.probing.hooks import num_layers
    from steerlab.control import ControlState
    from steerlab.engine import DirectionRegistry, SteeringSession

    state = ControlState(
        model_id,
        num_layers(model),
        DirectionRegistry(),
        on_message=lambda _message: None,
    )
    return SteeringSession(model, tokenizer, state), state


def prompt_ids(session: Any, user_text: str, system: str | None = None) -> Any:
    """Chat-templated token ids for one user message (with the system prompt, when there is one)."""
    from steerlab.decode import encode_prompt, render_messages

    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
    text = render_messages(session.tokenizer, messages, model_type=session.model_type)
    return encode_prompt(session.tokenizer, text, session.device)


def generate(session: Any, ids: Any, *, max_new_tokens: int, events: Any = None) -> Any:
    """steerlab's greedy loop: one forward pass per token, the control state re-checked before each."""
    from steerlab.decode import greedy_generate

    return greedy_generate(session, ids, max_new_tokens=max_new_tokens, events=events)


def control_for(model_id: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """A control file's content: {model, entries}. `steerlab schema control` prints the schema."""
    return {"model": model_id, "entries": entries}


def direction_entry(name: str, strength: float) -> dict[str, Any]:
    """One control entry: the registered direction `name` at `strength` (in class gaps)."""
    return {"source": {"kind": "direction", "name": name}, "strength": strength}
