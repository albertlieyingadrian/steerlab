"""Shared pieces: prompts, the answer parser, the metrics, and the capped OpenRouter caller.

One readout for every model: the same prompt text, at least 50 sampled answers per question and condition at
temperature 1, each reply parsed to one option letter (or marked invalid). Chat models get the prompt as one user
message; base models get the same text followed by "\nAnswer:".
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEFGHIJ"
COUNTRIES = ["United States", "Philippines", "Singapore", "Malaysia", "Indonesia", "Taiwan", "Brazil", "South Korea"]
CONDITIONS = ["default"] + COUNTRIES
N_SAMPLES = 50
LIVE_IN = {"United States": "the United States", "Philippines": "the Philippines"}


def load_questions(path: Path | None = None) -> list[dict]:
    return json.loads((path or HERE / "data" / "questions.json").read_text())


def prompt_text(q: dict, condition: str) -> str:
    """The user message. `condition` is "default" or a country name ("I live in <country>." comes first)."""
    text = q["question"].replace("[your country]", "your country")
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    body = f"{text}\n\nOptions:\n{opts}\n\nAnswer with the letter of one option only."
    if condition == "default":
        return body
    return f"I live in {LIVE_IN.get(condition, condition)}.\n\n{body}"


def base_text(q: dict, condition: str) -> str:
    """The same text for a base model, ending where the answer letter goes."""
    return prompt_text(q, condition) + "\nAnswer:"


_LEAD = re.compile(r"^\s*(?:\*\*)?(?:answer\s*[:\-]?\s*)?(?:option\s+)?\(?([A-J])\b(?![a-z'])", re.IGNORECASE)
_ANSWER_IS = re.compile(
    r"\b(?:answer|choose|pick|select|option)\s*(?:is|:)?\s*\(?\**([A-J])\b(?![a-z'])", re.IGNORECASE
)


def parse(reply: str, n_options: int) -> int:
    """Index of the chosen option, or -1 if none is named at the start or in an "answer is X" phrase (refusals, out-of-range letters)."""
    if reply is None:
        return -1
    r = reply.strip()
    r = re.sub(r"^<think>.*?</think>", "", r, flags=re.DOTALL).strip()
    m = _LEAD.match(r)
    if m and m.group(1).isupper():
        i = LETTERS.index(m.group(1))
        return i if i < n_options else -1
    m = _ANSWER_IS.search(r)
    if m and m.group(1).isupper():
        i = LETTERS.index(m.group(1))
        return i if i < n_options else -1
    return -1


def counts(choices: list[int], k: int) -> np.ndarray:
    c = np.zeros(k)
    for x in choices:
        if 0 <= x < k:
            c[x] += 1
    return c


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (square root of the divergence, base 2): 0 = identical, 1 = no overlap."""
    p = np.asarray(p, float) / np.sum(p)
    q = np.asarray(q, float) / np.sum(q)
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    return math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits, divided by log2(number of options): 0 = one answer, 1 = spread evenly."""
    p = np.asarray(p, float)
    p = p / p.sum()
    nz = p > 0
    k = len(p)
    return float(-np.sum(p[nz] * np.log2(p[nz])) / math.log2(k)) if k > 1 else 0.0


def floor_draws(human: list[float], n: int, rng: np.random.Generator, reps: int = 200) -> float:
    """Mean distance between a country's split and n answers drawn from that split: a perfect sampler's score."""
    h = np.asarray(human, float)
    return float(np.mean([jsd(rng.multinomial(n, h), h) for _ in range(reps)]))


def bootstrap(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling questions."""
    v = np.asarray(values, float)
    idx = rng.integers(0, len(v), (reps, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- OpenRouter, capped
API_CAP_USD = 10.0
LEDGER = HERE / "results" / "openrouter_spend.jsonl"
_lock = threading.Lock()
_spent = [None]
_read_at = [0.0]


def spent() -> float:
    """Total logged spend. Re-read from the shared ledger every 20 s, so parallel samplers see each other's spend."""
    if _spent[0] is None or time.time() - _read_at[0] > 20:
        _read_at[0] = time.time()
        _spent[0] = (
            sum(json.loads(x)["cost"] for x in LEDGER.read_text().splitlines() if x.strip()) if LEDGER.exists() else 0.0
        )
    return _spent[0]


class CapReached(RuntimeError):
    pass


def openrouter(model: str, text: str, key: str, *, max_tokens: int, extra: dict | None = None) -> str | None:
    """One sampled reply at temperature 1 (or the model's fixed default, 1, where temperature is not accepted).

    Stops (raises CapReached) once API_CAP_USD has been spent, counted from OpenRouter's own per-call cost."""
    import requests

    with _lock:
        if spent() >= API_CAP_USD:
            raise CapReached(f"OpenRouter cap ${API_CAP_USD} reached")
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "usage": {"include": True}, **(extra or {})}  # fmt: skip
    for attempt in range(6):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=60,
            )
            if r.status_code == 200:
                d = r.json()
                cost = float(d.get("usage", {}).get("cost") or 0.0)
                with _lock:
                    _spent[0] = spent() + cost
                    with LEDGER.open("a") as f:
                        f.write(json.dumps({"model": model, "cost": cost, "t": round(time.time())}) + "\n")
                ch = d.get("choices") or [{}]
                return (ch[0].get("message") or {}).get("content")
            if r.status_code in (402,):
                raise CapReached(f"OpenRouter refused: {r.status_code}")
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    return None
