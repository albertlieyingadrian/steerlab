"""An Antislop-style backtracking sampler for Hugging Face causal LMs (Paech et al., arXiv 2510.15061).

The model samples token by token. After each token the text so far is checked with the habit detectors; when a
banned pattern has just completed, generation steps back to the token where the pattern began, bans that token at
that position, and samples again from there. Nothing about the model changes: only which of its own continuations
are kept. Token banning alone cannot do this for multi-token patterns ("not just X, but Y"); backtracking can.

    from deslop.sampler import antislop_generate
    out = antislop_generate(model, tokenizer, messages, max_new_tokens=400, temperature=0.7)
    out.text, out.backtracks

Differences from the reference sampler: the banned set is this package's habit detectors (a pattern counts once it is
followed by another character, so "robust" is not banned inside "robustness"), the first token of a banned span is
removed outright rather than down-weighted, and after `max_backtracks` the sampler keeps what it has.

Needs the `models` extra (torch, transformers).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from deslop.habits import DETECTORS, Span

DEFAULT_BANNED = (
    "em_dash",
    "not_x_but_y",
    "signpost",
    "stock_phrase",
    "slop_word",
    "formatting",
    "invented_claim",
    "which_chain",
)
"""Habits the sampler steps back from. Lists of three, fragments and answered questions are left alone: they are
judgements about whole sentences, and people use them too."""


@dataclass
class SamplerOutput:
    text: str
    token_ids: list[int]
    backtracks: int
    finished: bool
    accepted: list[Span] = field(default_factory=list)
    """Banned spans kept because the backtrack budget ran out."""


def violations(text: str, habits: Sequence[str], *, final: bool) -> list[Span]:
    """Banned spans in `text`. Unless `final`, a span touching the end of the text is not yet confirmed."""
    spans = [span for habit in habits for span in DETECTORS[habit](text)]
    return [s for s in spans if final or s.end < len(text.rstrip())]


def _sample(logits: Any, *, temperature: float, top_p: float, top_k: int, generator: Any) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(0, order, sorted_probs)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator).item())


def _crop(cache: Any, length: int) -> None:
    """Keep the first `length` positions of a DynamicCache (transformers 5 takes a negative count to remove)."""
    current = int(cache.get_seq_length())
    if current <= length:
        return
    try:
        cache.crop(length - current)
    except ValueError:
        cache.crop(length)


def antislop_generate(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]] | None = None,
    *,
    prompt_ids: Sequence[int] | None = None,
    habits: Sequence[str] = DEFAULT_BANNED,
    max_new_tokens: int = 400,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_backtracks: int = 300,
    seed: int | None = None,
    logits_hook: Callable[[Any], Any] | None = None,
) -> SamplerOutput:
    """Generate one reply with backtracking. Pass chat `messages` (the chat template is applied) or `prompt_ids`.

    `temperature=0` is greedy decoding. No repetition penalty is applied (the same as repetition_penalty=1.0).
    """
    import torch
    from transformers import DynamicCache

    unknown = [h for h in habits if h not in DETECTORS]
    if unknown:
        raise KeyError(f"unknown habit(s): {', '.join(unknown)}")
    if prompt_ids is None:
        if messages is None:
            raise ValueError("pass messages or prompt_ids")
        rendered = tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    prompt = torch.tensor([list(prompt_ids)], device=device)
    n_prompt = prompt.shape[1]
    eos_ids = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]
    config_eos = getattr(model.generation_config, "eos_token_id", None) or []
    eos = set(eos_ids) | set(config_eos if isinstance(config_eos, list) else [config_eos])
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    cache = DynamicCache()
    with torch.no_grad():
        logits = model(prompt, past_key_values=cache, use_cache=True).logits[0, -1]
    tokens: list[int] = []
    bans: dict[int, set[int]] = {}
    accepted: list[Span] = []
    accepted_keys: set[tuple[int, str, str]] = set()
    backtracks = 0
    finished = False

    def step(token: int) -> Any:
        with torch.no_grad():
            out = model(torch.tensor([[token]], device=device), past_key_values=cache, use_cache=True)
        return out.logits[0, -1]

    while len(tokens) < max_new_tokens:
        position = len(tokens)
        scores = logits if logits_hook is None else logits_hook(logits)
        if position in bans:
            scores = scores.clone()
            scores[list(bans[position])] = float("-inf")
        token = _sample(scores, temperature=temperature, top_p=top_p, top_k=top_k, generator=generator)
        tokens.append(token)
        done = token in eos
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        bad = [s for s in violations(text, habits, final=done) if (s.start, s.habit, s.text) not in accepted_keys]
        if bad and backtracks < max_backtracks:
            first = min(bad, key=lambda s: s.start)
            # The token where the span begins: the last prefix whose decoded text ends at or before span.start.
            back = position
            while back > 0 and len(tokenizer.decode(tokens[:back], skip_special_tokens=True)) > first.start:
                back -= 1
            bans.setdefault(back, set()).add(tokens[back])
            for later in [p for p in bans if p > back]:
                del bans[later]
            tokens = tokens[:back]
            backtracks += 1
            # Rebuild the logits for position `back`: crop the cache to just before it and re-feed one token.
            _crop(cache, n_prompt + back - 1)
            logits = step(int(prompt[0, -1]) if back == 0 else tokens[back - 1])
            continue
        for s in bad:  # out of budget: keep them, and stop checking them
            accepted.append(s)
            accepted_keys.add((s.start, s.habit, s.text))
        if done:
            finished = True
            break
        logits = step(token)

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return SamplerOutput(text=text, token_ids=tokens, backtracks=backtracks, finished=finished, accepted=accepted)
