"""Language-model clients for labelling, with an on-disk cache and cost accounting.

`AnthropicLLM` (Claude Haiku 5.5 by default) and `OpenAILLM` call the vendors' SDKs, which are optional extras
(`pip install "steerlab-circuitkit[anthropic]"` or `[openai]`). Keys come from the usual environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); circuitkit never reads or prints them itself. `CachedLLM` wraps any client
so that a repeated prompt costs nothing and a rerun of a notebook gives the same labels. `ScriptedLLM` answers from
a function, for tests and offline demos.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# USD per million tokens (input, output), list prices; used only to report spend.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.00),
}


@dataclass
class Usage:
    """Running token and dollar totals for one client."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def add(self, model: str, inp: int, out: int) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        pin, pout = PRICES.get(model, (0.0, 0.0))
        self.usd += (inp * pin + out * pout) / 1e6


class LLM(Protocol):
    """One completion: a system prompt and a user message in, text out."""

    @property
    def model(self) -> str: ...

    @property
    def usage(self) -> Usage: ...

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


@dataclass
class AnthropicLLM:
    """Claude through the Anthropic SDK. Thinking is off and effort low: labelling is a short, cheap task."""

    model: str = "claude-haiku-5-5"
    effort: str = "low"
    usage: Usage = field(default_factory=Usage)
    _client: Any = None

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic()
        kwargs: dict[str, Any] = {}
        if self.model.startswith("claude-haiku-5"):
            kwargs = {"thinking": {"type": "disabled"}, "output_config": {"effort": self.effort}}
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        self.usage.add(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")


@dataclass
class OpenAILLM:
    """An OpenAI chat model through the OpenAI SDK."""

    model: str = "gpt-5-mini"
    usage: Usage = field(default_factory=Usage)
    _client: Any = None

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        import openai

        if self._client is None:
            self._client = openai.OpenAI()
        extra: dict[str, Any] = {"reasoning_effort": "minimal"} if self.model.startswith("gpt-5") else {}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            **extra,
        )
        u = resp.usage
        self.usage.add(self.model, u.prompt_tokens if u else 0, u.completion_tokens if u else 0)
        return resp.choices[0].message.content or ""


@dataclass
class ScriptedLLM:
    """Answers with `fn(system, prompt)`: for tests and offline demos. Counts calls, costs nothing."""

    fn: Callable[[str, str], str]
    model: str = "scripted"
    usage: Usage = field(default_factory=Usage)

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.usage.add(self.model, 0, 0)
        return self.fn(system, prompt)


def default_cache_dir() -> Path:
    return Path(os.environ.get("CIRCUITKIT_CACHE", Path.home() / ".cache" / "circuitkit" / "llm"))


@dataclass
class CachedLLM:
    """Wraps an LLM with a JSON-lines cache keyed by (model, system, prompt, max_tokens).

    The cache is append-only and safe to commit next to a notebook, so readers can rerun it without a key.
    """

    inner: Any
    path: Path | str | None = None
    _table: dict[str, str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def model(self) -> str:
        return str(self.inner.model)

    @property
    def usage(self) -> Usage:
        u: Usage = self.inner.usage
        return u

    def _file(self) -> Path:
        p = Path(self.path) if self.path is not None else default_cache_dir() / "cache.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load(self) -> dict[str, str]:
        with self._lock:  # threads share one client: fill the table fully before anyone reads it
            if self._table is None:
                table: dict[str, str] = {}
                f = self._file()
                if f.exists():
                    for line in f.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            row = json.loads(line)
                            table[row["key"]] = row["text"]
                self._table = table
            return self._table

    @staticmethod
    def key(model: str, system: str, prompt: str, max_tokens: int) -> str:
        h = hashlib.sha256(json.dumps([model, system, prompt, max_tokens]).encode("utf-8"))
        return h.hexdigest()

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        k = self.key(self.model, system, prompt, max_tokens)
        table = self._load()
        if k in table:
            self.usage.cached_calls += 1
            return table[k]
        text = str(self.inner.complete(system, prompt, max_tokens))
        with self._lock:
            table[k] = text
            with self._file().open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "model": self.model, "text": text}) + "\n")
        return text


def make_llm(provider: str = "anthropic", model: str | None = None, cache: Path | str | None = None) -> CachedLLM:
    """`make_llm("anthropic")` -> cached Claude Haiku 5.5; `make_llm("openai", "gpt-5-mini")` -> cached OpenAI."""
    if provider == "anthropic":
        inner: Any = AnthropicLLM(model=model or "claude-haiku-5-5")
    elif provider == "openai":
        inner = OpenAILLM(model=model or "gpt-5-mini")
    else:
        raise ValueError(f"unknown provider {provider!r} (anthropic or openai)")
    return CachedLLM(inner, cache)
