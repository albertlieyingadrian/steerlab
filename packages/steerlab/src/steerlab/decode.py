"""The manual decode loop: one forward pass per token, the control file checked before each one.

Why not `model.generate`? Because the demo needs to know, for every token, what was being steered when it was chosen,
and needs the control file to take effect on the *next* token. A hand-written loop with a KV cache makes that exact:

    prefill the prompt (unsteered)  ->  for each new token: apply events / poll the file, sync hooks, one forward,
                                        record the timeline entry, stream the token, feed it back

The timeline entry for token i says which entries were in force when token i was chosen, the norm each layer added and
the median residual norm at the steered positions (so the demo can print the ratio), and whether the state changed at
that token. `forced_logits` is the same computation without a cache (teacher forcing), used to prove the two agree.

Chat template rules: Qwen3's thinking block is off (`enable_thinking=False`, the one rule in
`defaults.template_kwargs_for`); Gemma has no system role, so a system prompt is folded into the first user turn.
Decoding is greedy with no repetition penalty at all, so strength 0 reproduces
`generate(do_sample=False, repetition_penalty=1.0)`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from transformers import DynamicCache

from steerlab import defaults
from steerlab.control import AppliedEntry, ControlValidationError, parse_control
from steerlab.engine import SteeringSession

Message = dict[str, str]


class Watcher(Protocol):
    """Called before every generated token: `ControlWatcher` re-reads a control file; the web worker's poller applies
    the admin's fader changes and may raise to cancel the reply."""

    def poll(self) -> bool: ...


def render_messages(tokenizer: Any, messages: Sequence[Message], *, model_type: str) -> str:
    """The chat template applied to `messages`, with the two model-specific rules from the module docstring."""
    rendered = [dict(message) for message in messages]
    if model_type.startswith("gemma") and rendered and rendered[0]["role"] == "system":
        system = rendered.pop(0)["content"]
        for message in rendered:
            if message["role"] == "user":
                message["content"] = f"{system}\n\n{message['content']}"
                break
        else:
            rendered.insert(0, {"role": "user", "content": system})
    kwargs = defaults.template_kwargs_for(model_type)
    text = tokenizer.apply_chat_template(rendered, tokenize=False, add_generation_prompt=True, **kwargs)
    return str(text)


def encode_prompt(tokenizer: Any, text: str, device: torch.device | str) -> torch.Tensor:
    """Token ids `[1, prompt_len]` of already-templated text (the template put the special tokens in)."""
    ids: torch.Tensor = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return ids.to(device)


@dataclass
class Event:
    """At generated token `token_index`, replace the control state with `control` (a control file's content)."""

    token_index: int
    control: dict[str, Any]


def load_events(path: str | Path) -> list[Event]:
    """Read a scripted events file: a JSON list of `{"token_index": int, "control": {model, entries}}`."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ControlValidationError(f"{path}: an events file is a JSON list of {{token_index, control}}")
    events = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not isinstance(item.get("token_index"), int) or item["token_index"] < 0:
            raise ControlValidationError(f"{path}: event {index} needs a non-negative integer token_index")
        if not isinstance(item.get("control"), dict):
            raise ControlValidationError(f"{path}: event {index} needs a control object ({{model, entries}})")
        parse_control(json.dumps(item["control"]))  # fail at load time, not at token 25
        events.append(Event(item["token_index"], item["control"]))
    return sorted(events, key=lambda event: event.token_index)


@dataclass
class TimelineEntry:
    """What was true when generated token `token_index` was chosen."""

    token_index: int
    token_id: int
    text: str
    applied: list[AppliedEntry]
    added_norm: dict[int, float]
    residual_norm: dict[int, float]
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "text": self.text,
            "applied": [entry.to_dict() for entry in self.applied],
            "added_norm": {str(layer): norm for layer, norm in self.added_norm.items()},
            "residual_norm": {str(layer): norm for layer, norm in self.residual_norm.items()},
            "changed": self.changed,
        }

    def describe(self) -> str:
        """One line for the terminal: token index, each applied entry, and the norm ratio where it is known."""
        if not self.applied:
            return f"[token {self.token_index}] steering off"
        parts = [entry.describe() for entry in self.applied]
        ratios = [
            f"L{layer}: added {self.added_norm[layer]:.3g} vs median residual {self.residual_norm[layer]:.3g}"
            f" (ratio {self.added_norm[layer] / self.residual_norm[layer]:.3g})"
            for layer in sorted(self.added_norm)
            if self.residual_norm.get(layer)
        ]
        return f"[token {self.token_index}] " + "; ".join(parts) + (f" | {'; '.join(ratios)}" if ratios else "")


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    timeline: list[TimelineEntry]
    step_logits: list[torch.Tensor] = field(default_factory=list, repr=False)
    finish_reason: str = "length"
    prompt_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "prompt_length": self.prompt_length,
            "finish_reason": self.finish_reason,
            "timeline": [entry.to_dict() for entry in self.timeline],
        }


def _eos_ids(session: SteeringSession) -> set[int]:
    ids: set[int] = set()
    for candidate in (
        getattr(session.model.generation_config, "eos_token_id", None),
        getattr(session.tokenizer, "eos_token_id", None),
    ):
        if isinstance(candidate, int):
            ids.add(candidate)
        elif isinstance(candidate, Iterable):
            ids.update(int(c) for c in candidate)
    return ids


def greedy_generate(
    session: SteeringSession,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    watcher: Watcher | None = None,
    events: Sequence[Event] | None = None,
    on_token: Callable[[int, str], None] | None = None,
    on_change: Callable[[TimelineEntry], None] | None = None,
) -> Generation:
    """Greedy decoding with a KV cache, re-checking the control state before every token.

    `on_token(index, text)` streams each token's text as soon as it is chosen. `on_change(entry)` fires when the state
    changed at that token (and at token 0 when something is already in force), before the token streams.
    """
    model = session.model
    eos = _eos_ids(session)
    pending = sorted(events or [], key=lambda event: event.token_index)
    cache = DynamicCache(config=model.config)
    prompt_ids = prompt_ids.to(session.device)
    session.prompt_length = int(prompt_ids.shape[1])
    input_ids = prompt_ids
    generated: list[int] = []
    timeline: list[TimelineEntry] = []
    step_logits: list[torch.Tensor] = []
    text_so_far = ""
    finish_reason = "length"
    with torch.no_grad():
        for index in range(max_new_tokens):
            while pending and pending[0].token_index <= index:
                try:
                    session.state.apply(pending.pop(0).control)
                except Exception as error:  # a bad event keeps the last valid state, like a bad file
                    session.state.say(f"event at token {index} ignored: {error}")
            if watcher is not None:
                watcher.poll()
            changed = session.sync()
            session.position_offset = cache.get_seq_length()
            session.last_stats = {}
            output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            logits = output.logits[0, -1].float().cpu()
            next_id = int(logits.argmax())
            stats = dict(session.last_stats)
            entry = TimelineEntry(
                token_index=index,
                token_id=next_id,
                text="",
                applied=list(session.state.applied),
                added_norm={layer: added for layer, (added, _) in stats.items()},
                residual_norm={layer: residual for layer, (_, residual) in stats.items()},
                changed=changed,
            )
            if on_change is not None and (changed or (index == 0 and entry.applied)):
                on_change(entry)
            if next_id in eos:
                finish_reason = "stop"
                break
            generated.append(next_id)
            full = session.tokenizer.decode(generated, skip_special_tokens=True)
            if not full.endswith("�"):  # an incomplete multi-byte character: wait for the next token
                entry.text = full[len(text_so_far) :]
                text_so_far = full
            timeline.append(entry)
            step_logits.append(logits)
            if on_token is not None:
                on_token(index, entry.text)
            input_ids = torch.tensor([[next_id]], device=session.device)
    return Generation(
        text=text_so_far,
        token_ids=generated,
        timeline=timeline,
        step_logits=step_logits,
        finish_reason=finish_reason,
        prompt_length=session.prompt_length,
    )


def forced_logits(session: SteeringSession, prompt_ids: torch.Tensor, generated_ids: Sequence[int]) -> torch.Tensor:
    """Teacher forcing without a cache: the logits that chose each generated token, `[len(generated), vocab]`.

    Position i's logits come from the forward over `prompt + generated[:i]`, exactly what the cached loop saw at step i,
    so the two must agree for a fixed control (the KV-parity invariant).
    """
    prompt_ids = prompt_ids.to(session.device)
    session.prompt_length = int(prompt_ids.shape[1])
    session.position_offset = 0
    session.sync()
    tail = torch.tensor([list(generated_ids)], device=session.device, dtype=prompt_ids.dtype)
    full = torch.cat([prompt_ids, tail], dim=1) if tail.shape[1] else prompt_ids
    with torch.no_grad():
        logits = session.model(input_ids=full, use_cache=False).logits[0]
    start = session.prompt_length - 1
    forced: torch.Tensor = logits[start : start + len(generated_ids)].float().cpu()
    return forced


def _stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def chat(
    session: SteeringSession,
    *,
    max_new_tokens: int,
    system: str | None = None,
    watcher: Watcher | None = None,
    input_fn: Callable[[str], str] = input,
    write: Callable[[str], None] = _stdout,
    on_change: Callable[[TimelineEntry], None] | None = None,
) -> list[Message]:
    """A multi-turn terminal chat. Each turn re-prefills the whole history unsteered, then steers the new reply.

    Commands: `r` restarts the current reply (the last assistant turn is dropped and regenerated), `c` clears the
    history (a fresh conversation, same model and control file), `q` quits. Returns the history. Steered text from
    earlier turns stays in the context; `r` gives a clean baseline for the current reply, `c` for the next question.
    """
    history: list[Message] = []
    write(
        "steerlab chat: type a message and press Enter. "
        "`r` restarts the current reply, `c` clears the history, `q` quits.\n"
    )

    def default_on_change(entry: TimelineEntry) -> None:
        write(f"\n{entry.describe()}\n  (earlier steered text stays in the context; `r` restarts this reply)\n")

    report = on_change or default_on_change
    while True:
        try:
            line = input_fn("you> ").strip()
        except (EOFError, StopIteration):
            break
        if line in ("q", "quit", "exit"):
            break
        if line in ("c", "clear"):
            history.clear()
            write("history cleared: the next message starts a fresh conversation\n")
            continue
        if line == "r":
            if len(history) < 2 or history[-1]["role"] != "assistant":
                write("nothing to restart yet\n")
                continue
            history.pop()
            write(
                "restarting the reply: the history is prefilled again unsteered; "
                "the new reply uses the current control\n"
            )
        elif not line:
            continue
        else:
            history.append({"role": "user", "content": line})
        messages = ([{"role": "system", "content": system}] if system else []) + history
        text = render_messages(session.tokenizer, messages, model_type=session.model_type)
        prompt = encode_prompt(session.tokenizer, text, session.device)
        write("model> ")
        generation = greedy_generate(
            session,
            prompt,
            max_new_tokens=max_new_tokens,
            watcher=watcher,
            on_token=lambda _index, piece: write(piece),
            on_change=report,
        )
        write("\n")
        history.append({"role": "assistant", "content": generation.text})
    return history
