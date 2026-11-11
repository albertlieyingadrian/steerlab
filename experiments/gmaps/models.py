"""Tinker chat backend for bases other than Qwen3-8B (the shared `fin-llm/backends.py` is Qwen3-8B only).

    nemo                     Nemotron-3.5-Lightning-30B-A3B (MoE, 3B active), reasoning off
    nemo@tinker://...        a LoRA sampler checkpoint on that base

Reasoning off means the chat template's `enable_thinking=False`, which opens the answer with an empty
`<think></think>`. Prices: Tinker's public model list (tinker-docs.thinkingmachines.ai/tinker/models.json, read
2026-11-11), USD per million tokens.
"""

from __future__ import annotations

import os
import re
import threading
import time

import run

BASES = {
    "nemo": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",  # MoE 30B, 3B active
    "super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",  # MoE 120B, 12B active
    "q36": "Qwen/Qwen3.6-35B-A3B",  # MoE 35B, 3B active
    "q397": "Qwen/Qwen3.5-397B-A17B",  # MoE 397B, 17B active (also the open-weight teacher)
}
PRICE = {
    "nemo": {"prefill": 0.184, "sample": 0.495, "train": 0.44},
    "super": {"prefill": 0.57, "sample": 1.44, "train": 1.276},
    "q36": {"prefill": 0.45, "sample": 1.335, "train": 1.177},
    "q397": {"prefill": 1.65, "sample": 7.50, "train": 6.60},
}
_TOK: dict = {}
_LOCK = threading.Lock()


def tokenizer(base: str):
    with _LOCK:
        if base not in _TOK:
            from transformers import AutoTokenizer

            _TOK[base] = AutoTokenizer.from_pretrained(BASES[base])
    return _TOK[base]


def chat_ids(base: str, messages: list[dict], thinking: bool = False) -> list[int]:
    tok = tokenizer(base)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    return tok.encode(text, add_special_tokens=False)


def stop_ids(base: str) -> list[int]:
    return [tokenizer(base).convert_tokens_to_ids("<|im_end|>")]


class TinkerChat:
    provider = "tinker"

    def __init__(self, budget, base: str, model_path: str | None = None, thinking: bool = False):
        run.fin.load_env()
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY is not set; nothing was sent")
        import tinker

        self.budget, self.base, self.thinking = budget, base, thinking
        self.name = base + ("-think" if thinking else "") + (f"@{model_path.split('/')[-1]}" if model_path else "")
        svc = tinker.ServiceClient()
        self.client = (
            svc.create_sampling_client(model_path=model_path)
            if model_path
            else svc.create_sampling_client(base_model=BASES[base])
        )

    def submit(self, messages, max_tokens=400, temperature=0.0, seed=0, n=1):
        """Send one sampling request without waiting; `finish` turns the handle into replies."""
        from tinker import types

        self.budget.check()
        ids = chat_ids(self.base, messages, self.thinking)
        params = types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, seed=seed if n == 1 else None, stop=stop_ids(self.base)
        )
        return self.client.sample(types.ModelInput.from_ints(ids), n, params), ids, n, time.time()

    def finish(self, handle):
        future, ids, n, start = handle
        result = future.result()
        seconds = time.time() - start
        stops = stop_ids(self.base)
        p = PRICE[self.base]
        out = []
        for seq in result.sequences:
            toks = [t for t in seq.tokens if t not in stops]
            raw = tokenizer(self.base).decode(toks, skip_special_tokens=False).split("</think>")[-1]
            text = re.sub(r"<\|[^|]*\|>|<think>", "", raw).strip()
            out.append(run.backends.Reply(text, len(ids), len(seq.tokens), 0.0, seconds))
        usd = (len(ids) * p["prefill"] + sum(r.tokens_out for r in out) * p["sample"]) / 1e6
        self.budget.add("tinker", self.name, usd, prefill=len(ids), sample=sum(r.tokens_out for r in out))
        for r in out:
            r.usd = usd / n
        return out

    def chat(self, messages, max_tokens=400, temperature=0.0, seed=0, n=1):
        return self.finish(self.submit(messages, max_tokens, temperature, seed, n))


def map_chat(model, items, make_messages, workers=4, window=48, **kw):
    """Yield (item, replies or exception) in input order. Tinker models keep up to `window` requests in flight from
    one thread (asynchronous futures, not extra local workers); other backends use a pool of at most 4 threads."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    if hasattr(model, "submit"):
        queue = deque()
        for it in items:
            queue.append((it, model.submit(make_messages(it), **kw)))
            if len(queue) >= window:
                it0, h = queue.popleft()
                yield it0, _safe(model.finish, h)
        while queue:
            it0, h = queue.popleft()
            yield it0, _safe(model.finish, h)
        return

    def one(it):
        return it, _safe(lambda: model.chat(make_messages(it), **kw))

    with ThreadPoolExecutor(min(workers, 4)) as ex:
        yield from ex.map(one, items)


def _safe(fn, *args):
    for attempt in range(3):
        try:
            return fn(*args)
        except SystemExit:
            raise
        except Exception as e:
            err = e
            time.sleep(2 * (attempt + 1))
            if args:  # a Tinker handle cannot be resubmitted from here
                break
    return err


# Anthropic API list prices, USD per million tokens (claude-api skill table, cached 2026-11-03): input, output,
# cache read. Cache writes cost 1.25x input. Fable 5.1's cache-read price is not in the table; 10% of input is used.
CLAUDE_PRICE = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-fable-5-1": (10.0, 50.0, 1.00),
    "claude-fable-5-2": (8.0, 40.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


class Claude:
    """Anthropic API (the default provider since 2026-11-11). Spec `claude-opus-5-5:low` = model + effort.

    Thinking is always on for these models (the `thinking` field is omitted); `effort` sets its depth. No temperature
    (the current API rejects sampling parameters), so repeated calls are samples, not greedy decodes. The system
    prompt and the few-shot turns are a cached prefix. Claude outputs are never used as training data."""

    provider = "anthropic"

    def __init__(self, budget, model: str, effort: str | None):
        run.fin.load_env()
        import anthropic

        # Anthropic spend sits outside the pilot cap: its own ledger article
        budget = run.fin.Budget("gmaps-claude", budget.tag)
        self.budget, self.model, self.effort = budget, model, effort or "medium"
        self.name = f"{model}-{self.effort}"
        self.client = anthropic.Anthropic(max_retries=6, timeout=300)

    def chat(self, messages, max_tokens=16000, temperature=None, seed=0, n=1):
        self.budget.check()
        system = [{"type": "text", "text": messages[0]["content"], "cache_control": {"type": "ephemeral"}}]
        turns = [dict(m) for m in messages[1:]]
        if len(turns) > 1:  # cache through the last few-shot turn; the transmission itself varies
            last = turns[-2]
            last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
        out = []
        for _ in range(n):
            start = time.time()
            r = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=turns,
                output_config={"effort": self.effort},
            )
            seconds = time.time() - start
            u = r.usage
            pin, pout, pread = CLAUDE_PRICE[self.model]
            cw, cr = u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0
            usd = (u.input_tokens * pin + cw * pin * 1.25 + cr * pread + u.output_tokens * pout) / 1e6
            self.budget.add(
                "anthropic", self.name, usd, input=u.input_tokens, cache_write=cw, cache_read=cr, output=u.output_tokens
            )
            text = "" if r.stop_reason == "refusal" else "".join(b.text for b in r.content if b.type == "text")
            out.append(run.backends.Reply(text, u.input_tokens + cw + cr, u.output_tokens, usd, seconds))
        return out


def make(spec: str, budget):
    base, _, path = spec.partition("@")
    think = base.endswith("-think")
    base = base.removesuffix("-think")
    if base in BASES:
        return TinkerChat(budget, base, path or None, think)
    if spec.startswith("claude-"):
        model, _, effort = spec.partition(":")
        return Claude(budget, model, effort or None)
    return run.backends.make(spec, budget)
