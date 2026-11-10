"""Chat backends with per-call cost: Qwen3-8B (base or a LoRA checkpoint) sampled on Tinker, and OpenAI / OpenRouter.

Every backend has `chat(messages, max_tokens, temperature, seed) -> Reply`; the cost of each call is recorded in the
article's ledger (`fin.Budget`). Prices are USD per million tokens: Tinker's pinned table (steerlab, 2026-10-25) and
the providers' list prices on 2026-11-10 (OpenRouter's model list; OpenAI's direct prices are the same; cached
input on OpenAI is 10% of the input price).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import fin

QWEN = "Qwen/Qwen3-8B"
TINKER_PRICE = {"prefill": 0.195, "sample": 0.6, "train": 0.44}
API_PRICE = {  # (input, output) USD per million tokens
    "gpt-5.4-nano": (0.2, 1.25),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.5": (5.0, 30.0),
    "openai/gpt-5.5": (5.0, 30.0),
}


@dataclass
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    usd: float
    seconds: float
    tokens: list[int] | None = None  # sampled ids (Tinker), for training
    logprobs: list[float] | None = None
    prompt_ids: list[int] | None = None


_TOK = None
_TOK_LOCK = threading.Lock()


def tokenizer():
    global _TOK
    with _TOK_LOCK:  # transformers' lazy imports are not thread-safe
        if _TOK is None:
            from transformers import AutoTokenizer

            _TOK = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    return _TOK


def chat_ids(messages: list[dict], thinking: bool) -> list[int]:
    text = tokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    return tokenizer().encode(text, add_special_tokens=False)


def stop_ids() -> list[int]:
    return [tokenizer().convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


def strip_think(text: str) -> str:
    return text.split("</think>")[-1].strip()


class Tinker:
    """Qwen3-8B on Tinker; `model_path` is a LoRA sampler checkpoint (tinker://...) or None for the base model."""

    provider = "tinker"

    def __init__(self, budget: fin.Budget, model_path: str | None = None, thinking: bool = False):
        fin.load_env()
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY is not set; nothing was sent")
        import tinker

        self.budget, self.thinking, self.model_path = budget, thinking, model_path
        tokenizer()
        self.name = f"qwen3-8b{'-think' if thinking else ''}" + (f"@{model_path.split('/')[-1]}" if model_path else "")
        svc = tinker.ServiceClient()
        if model_path:
            self.client = svc.create_sampling_client(model_path=model_path)
        else:
            self.client = svc.create_sampling_client(base_model=QWEN)

    def chat(self, messages, max_tokens=1024, temperature=0.0, seed=0, n=1) -> list[Reply]:
        from tinker import types

        self.budget.check()
        ids = chat_ids(messages, self.thinking)
        stops = stop_ids()
        # a seed makes all n samples identical on Tinker, so several samples are drawn unseeded
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed if n == 1 else None,
                                      stop=stops)  # fmt: skip
        start = time.time()
        result = self.client.sample(types.ModelInput.from_ints(ids), n, params).result()
        seconds = time.time() - start
        out = []
        for seq in result.sequences:
            toks = list(seq.tokens)
            out.append(Reply(tokenizer().decode([t for t in toks if t not in stops], skip_special_tokens=True),
                             len(ids), len(toks), 0.0, seconds, toks,
                             list(seq.logprobs) if seq.logprobs is not None else None, ids))  # fmt: skip
        # prefill is billed once per call, however many samples
        total = (len(ids) * TINKER_PRICE["prefill"] + sum(r.tokens_out for r in out) * TINKER_PRICE["sample"]) / 1e6
        self.budget.add(
            "tinker",
            self.name,
            total,
            prefill=len(ids),
            sample=sum(r.tokens_out for r in out),
        )
        for r in out:
            r.usd = total / n
        return out


class API:
    """`gpt-*` through OpenAI direct; a model name with a '/' through OpenRouter.

    OpenAI was used direct, then through OpenRouter while its key had no credit (2026-11-10, ~16:00-20:00), then direct
    again; every ledger row names its provider. Cached input tokens (OpenAI reports them) cost 10% of the input price;
    calls through OpenRouter were recorded at the full input price."""

    def __init__(self, budget: fin.Budget, model: str, effort: str | None = None):
        fin.load_env()
        from openai import OpenAI

        self.budget, self.model, self.effort = budget, model, effort
        self.name = model + (f"-{effort}" if effort else "")
        if "/" in model:
            self.provider, self.route = "openrouter", model
            self.client = OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
                timeout=180,
                max_retries=4,
            )
        else:
            self.provider, self.route = "openai", model
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=6)

    def chat(self, messages, max_tokens=4096, temperature=None, seed=0, n=1) -> list[Reply]:
        self.budget.check()
        kwargs = {
            "model": self.route,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        start = time.time()
        r = self.client.chat.completions.create(**kwargs)
        seconds = time.time() - start
        pin, pout = API_PRICE[self.model]
        tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
        details = getattr(r.usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if self.provider == "openai" else 0
        usd = ((tin - cached) * pin + cached * pin * 0.1 + tout * pout) / 1e6
        self.budget.add(self.provider, self.name, usd, input=tin, cached=cached, output=tout)
        return [Reply(r.choices[0].message.content or "", tin, tout, usd, seconds)]


def make(spec: str, budget: fin.Budget):
    """`qwen`, `qwen-think`, `qwen@tinker://...`, `gpt-5.4-mini`, `gpt-5.4-mini:low` (reasoning effort)."""
    if spec.startswith("qwen"):
        name, _, path = spec.partition("@")
        return Tinker(budget, path or None, thinking=name.endswith("think"))
    model, _, effort = spec.partition(":")
    return API(budget, model, effort or None)
