"""OpenRouter chat calls with a hard spend cap enforced in code ($10 for Episode 2; calls stop at $9.50).

Every call's ``usage.cost`` (USD, reported by OpenRouter) is appended to results/spend_openrouter.jsonl; the running
total is read back at start-up, so the cap holds across processes and reruns.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "results" / "spend_openrouter.jsonl"
HARD_CAP_USD = 6.90  # $10 -> $6 -> $6.90 (embeddings only) on 2026-11-19 (coordinator); see plan amendment
STOP_AT_USD = 6.90
URL = "https://openrouter.ai/api/v1/chat/completions"


class CapReached(RuntimeError):
    pass


class OpenRouter:
    def __init__(self, label: str):
        from dotenv import load_dotenv

        for root in HERE.parents:  # the repo root (worktrees live under it in .claude/worktrees/)
            load_dotenv(root / ".env")
            load_dotenv(root / ".env.local")
        # the topped-up account (the first key's account was spent out on 2026-11-18)
        self.key = os.environ.get("OPENROUTER_API_KEY_2") or os.environ["OPENROUTER_API_KEY"]
        self.label = label
        self.lock = threading.Lock()
        self.spent = self.total_spent()

    @staticmethod
    def total_spent() -> float:
        if not LEDGER.exists():
            return 0.0
        return sum(json.loads(x)["usd"] for x in LEDGER.read_text().splitlines() if x.strip())

    def chat(self, model: str, messages: list[dict], **params) -> dict:
        """Returns {"text", "usd", "usage", "finish_reason"}; raises CapReached past $9.50."""
        with self.lock:
            if self.spent >= STOP_AT_USD:
                raise CapReached(f"OpenRouter spend ${self.spent:.2f} reached the ${STOP_AT_USD} stop")
        body = {"model": model, "messages": messages, "usage": {"include": True}, **params}
        for attempt in range(6):
            try:
                r = requests.post(URL, headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=180)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                if "choices" not in data:
                    raise requests.HTTPError(str(data)[:300])
                break
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        usage = data.get("usage") or {}
        usd = float(usage.get("cost") or 0.0)
        with self.lock:
            self.spent += usd
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": self.label, "model": model,
                                    "usd": usd, "prompt_tokens": usage.get("prompt_tokens"),
                                    "completion_tokens": usage.get("completion_tokens")}) + "\n")
        ch = data["choices"][0]
        return {"text": (ch.get("message") or {}).get("content") or "", "usd": usd, "usage": usage,
                "finish_reason": ch.get("finish_reason"), "provider": data.get("provider")}

    def log_extra(self, model: str, usd: float):
        """Record spend made outside ``chat`` (e.g. embeddings) so the cap counts it."""
        with self.lock:
            self.spent += usd
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": self.label, "model": model,
                                    "usd": usd}) + "\n")
