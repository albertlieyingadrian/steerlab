"""
The independent frontier judge: Claude Haiku 5.5 (anthropic/claude-haiku-5.5 on OpenRouter, reasoning disabled),
given the same instructions as the open judge (common.judge_messages) and asked for one digit. Key:
OPENROUTER_API_KEY_2 (the first account ran dry on 2026-11-19). With reasoning on and a small max_tokens, Haiku 5.5
spent the budget thinking and returned empty text, hence reasoning off. Used only for (a) agreement with the open
judge on base replies and (b) quality at the final and drift-matched checkpoints. Hard cap $5 (budget.py), checked
before every call; answers are cached by content so nothing is paid for twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget import charge_openrouter, check_openrouter
from common import CACHE, judge_messages, load_env

MODEL = "anthropic/claude-haiku-5.5"
CACHE_FILE = CACHE / "frontier_cache.jsonl"
_lock = threading.Lock()


def _key(task, prompt, reply):
    return hashlib.sha1(json.dumps([MODEL, task, prompt, reply]).encode()).hexdigest()


def _load_cache():
    if not CACHE_FILE.exists():
        return {}
    out = {}
    for line in CACHE_FILE.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = r
    return out


def _call(messages):
    """One scoring call. Returns (text, cost in $ from OpenRouter's usage.cost)."""
    import os
    import urllib.request

    load_env()
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 16,
            "reasoning": {"enabled": False},
            "usage": {"include": True},
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY_2']}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 402, 403) or attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    text = resp["choices"][0]["message"].get("content") or ""
    return text, float(resp.get("usage", {}).get("cost", 0.0) or 0.0)


def score(items, purpose, workers=8):
    """items: {task, prompt, reply}. Returns a list of scores (1-9, or None if unparseable), in order."""
    cache = _load_cache()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def one(it):
        k = _key(it["task"], it["prompt"], it["reply"])
        if k in cache:
            return cache[k]["score"]
        check_openrouter()
        text, cost = _call(judge_messages(it["task"], it["prompt"], it["reply"][:6000]))
        m = re.search(r"[1-9]", text)
        s = int(m.group(0)) if m else None
        with _lock:
            charge_openrouter(MODEL, cost, purpose)
            with CACHE_FILE.open("a") as f:
                f.write(json.dumps({"key": k, "score": s, "raw": text}) + "\n")
        return s

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, items))


if __name__ == "__main__":
    from budget import openrouter_spent

    smoke = [
        {"task": "creative", "prompt": "Write a one-line story about a lighthouse.", "reply": "I can't help with that."}
    ]
    print(score(smoke, "smoke"))
    print(f"OpenRouter spent: ${openrouter_spent():.4f}")
