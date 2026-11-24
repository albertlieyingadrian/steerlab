"""Paths, corpus loading, windows, the train/test split and the bootstrap shared by every step.

Texts stay in the local cache (EditLens is research-only and gated); `results/` holds only counts and rates.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path.insert(0, str(LAB / "packages" / "deslop" / "src"))
RESULTS = HERE / "results"
CACHE = Path.home() / ".cache" / "steerlab" / "slop-rules"
EDITLENS = Path.home() / ".cache" / "steerlab" / "editlens" / "data"
HC3 = CACHE / "hc3" / "all.jsonl"
CORPUS = CACHE / "corpus.jsonl"  # every text, human and AI, with its window
GEN = CACHE / "gen"  # new open-model generations, one jsonl per model
CLASSIC = LAB / "experiments" / "deslop" / "data" / "human" / "passages.jsonl"
H2H = LAB / "experiments" / "deslop-h2h" / "results" / "gen" / "original.jsonl"

SEED = 20261124
TEST_SHARE = 0.4
WINDOW_MIN, WINDOW_CAP, TEXT_MIN = 150, 250, 60

# The open-weights models generated for this note (OpenRouter ids), the decision set.
OPEN_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "z-ai/glm-5.3",
    "moonshotai/kimi-k3",
    "qwen/qwen3.8-27b",
    "google/gemma-4-31b-it",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "mistralai/mistral-large-2512",
    "meta-llama/llama-3.3-70b-instruct",
]
SHORT = {m: m.split("/")[1] for m in OPEN_MODELS}

EDITLENS_GENRE = {
    "news": "news article",
    "reddit_writing_prompts": "short story posted in reply to a Reddit writing prompt",
    "fineweb_edu": "educational web article",
    "amazon_reviews": "Amazon product review",
    "google_reviews": "Google review of a local business",
    "enron_email": "work email",
}
HC3_SOURCES = ["reddit_eli5", "open_qa", "wiki_csai", "finance", "medicine"]

WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
SENT = re.compile(r"[^.!?\n]+(?:[.!?]+[\"”’)]*|\n|$)")


def n_words(text: str) -> int:
    return len(WORD.findall(text))


def split_first_sentence(text: str) -> tuple[str, str]:
    """(first sentence, rest). The first sentence ends at . ! ? or a line break."""
    text = text.strip()
    m = re.search(r"[.!?]+[\"”’)]*\s+|\n", text)
    if not m or m.end() > 400:
        return "", text
    return text[: m.end()].strip(), text[m.end() :].strip()


def window(text: str) -> str | None:
    """Sentences from the start until at least WINDOW_MIN words (cut at WINDOW_CAP); None if under TEXT_MIN words."""
    text = text.strip()
    if n_words(text) < TEXT_MIN:
        return None
    out, n = [], 0
    for m in SENT.finditer(text):
        piece = m.group(0)
        w = n_words(piece)
        if n + w > WINDOW_CAP and n >= TEXT_MIN:
            break
        out.append(piece)
        n += w
        if n >= WINDOW_MIN:
            break
    return "".join(out).strip()


_DETOK = [
    (r" n't\b", "n't"),
    (r" '(s|re|ve|ll|d|m)\b", r"'\1"),
    (r" ([,.;:!?%)])", r"\1"),
    (r"\( ", "("),
    (r"\$ (?=\d)", "$"),
    (r'" ([^"]*?) "', r'"\1"'),
    (r"[ \t]+", " "),
]


def detok(text: str) -> str:
    """Undo HC3's tokenisation ("does n't", "you 're", "word ,") so contractions and punctuation rules can match."""
    for a, b in _DETOK:
        text = re.sub(a, b, text)
    return text.strip()


def group_split(group: str) -> str:
    """Deterministic 60/40 split by prompt group."""
    h = int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < TEST_SHARE else "train"


def load_corpus(split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in CORPUS.read_text().split("\n") if line.strip()]
    return [r for r in rows if split is None or r["split"] == split]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def bootstrap_groups(groups: np.ndarray, n: int = 1000, seed: int = SEED) -> list[np.ndarray]:
    """Index arrays for `n` bootstrap draws that resample prompt groups (all windows of a group move together)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws.append(np.concatenate([members[i] for i in pick]))
    return draws
