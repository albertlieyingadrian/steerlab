"""A minimal OpenAI-compatible chat client (OpenRouter, vLLM, Ollama, ...) with retries, a disk cache and a spend cap.

The API key is read from an environment variable whose *name* is passed in; it is never logged.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

Chat = Callable[[list[dict[str, str]]], str]


class BudgetExceeded(RuntimeError):
    pass


class Client:
    """`client(messages) -> text`. Every call is cached by (model, messages, params) in `cache` (a JSONL file)."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        cache: str | Path | None = None,
        max_usd: float = 5.0,
        **params: Any,
    ) -> None:
        self.model, self.base_url, self.params = model, base_url.rstrip("/"), {"temperature": 0.0, **params}
        self.key = os.environ.get(api_key_env, "")
        self.max_usd, self.usd, self.calls = max_usd, 0.0, 0
        self.cache_path = Path(cache) if cache else None
        self.cache: dict[str, str] = {}
        self.lock = threading.Lock()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["text"]
                self.usd += row.get("usd", 0.0)

    def key_for(self, messages: list[dict[str, str]], params: dict[str, Any] | None = None) -> str:
        blob = json.dumps([self.model, messages, params or self.params], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def __call__(self, messages: list[dict[str, str]], **extra: Any) -> str:
        """`extra` overrides parameters for this call only (e.g. a per-prompt seed)."""
        params = {**self.params, **extra}
        k = self.key_for(messages, params)
        if k in self.cache:
            return self.cache[k]
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.usd:.2f} of ${self.max_usd:.2f}")
        body = {"model": self.model, "messages": messages, **params, "usage": {"include": True}}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    out = json.loads(r.read())
                text = out["choices"][0]["message"]["content"] or ""
                usd = float((out.get("usage") or {}).get("cost") or 0.0)
                break
            except (OSError, http.client.HTTPException, KeyError, IndexError, json.JSONDecodeError) as e:
                if attempt == 5:
                    raise RuntimeError(f"{self.model}: {type(e).__name__} after 6 tries") from e
                time.sleep(2**attempt)
        with self.lock:
            self.cache[k], self.usd, self.calls = text, self.usd + usd, self.calls + 1
            if self.cache_path:
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": k, "text": text, "usd": usd}) + "\n")
        return text


def pmap(fn: Callable[[Any], Any], xs: Iterable[Any], workers: int = 16) -> list[Any]:
    """`[fn(x) for x in xs]` on a thread pool, order kept."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, xs))


def parse_json(text: str) -> Any:
    """The first JSON object in a reply, tolerating code fences and prose around it."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj
