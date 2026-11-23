"""Shared paths, splits and checks for the detector-evasion study (Model voice, Episode 5).

A rewriter (Qwen3-8B + LoRA) is trained with GRPO to rewrite AI-written passages so that our 9B slop detector scores
them human. Episode 3 (`../deslop-h2h/`) supplies the source passages and the rulers; Episode 1 (`../deslop/`) supplies
the prompts, the 318 human passages and the habit scorer. Both are imported, never copied.

Splits (fixed before any rewrite was generated):
- train: Qwen3-8B's untouched replies to the 264 training prompts of topics 4-47 (Episode 3's `original-train`), plus
  two more samples per prompt drawn by `prep.py` (792 passages);
- dev: the same model's replies to the 24 dev prompts (topics 0-3), used for every setting and threshold;
- held-out: Episode 3's 96 untouched Qwen3-8B replies to the 48 held-out prompts (`original`), every detector flags
  all 96; and, as a second generator, Episode 1's 48 Qwen3-4B replies (`base`). Never used for a choice.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EP1 = HERE.parent / "deslop"
EP3 = HERE.parent / "deslop-h2h"
RESULTS = HERE / "results"
CONFIGS = HERE / "configs"
for p in (EP1, HERE.parents[1] / "packages" / "deslop" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MODAL_PROFILE", "albertlieyingadrian")

from prompts import held_out, training  # Episode 1's prompts

APP = "detector-evasion"
SEED = 0
B = 10_000
# Thresholds that flag 1% of the 318 human passages, from Episode 3's summary (same passages, same detectors).
EP3_THRESHOLDS = json.loads((EP3 / "results" / "summary.json").read_text())["thresholds_1pct"]
REWRITE_PROMPT = (
    "Rewrite the passage below so that it reads as if a person wrote it. Keep every fact, claim and point it makes, "
    "and add none. Reply with the rewritten passage only.\n\nPassage:\n{source}"
)

# Judge prompts (gemma-3-27b-it). CLAIM_CHECK is the held-out content ruler (one question per key claim); FIDELITY is a
# secondary 1-5 rating; HUMAN is Episode 3's question, word for word.
CLAIM_CHECK = (
    "Here is a passage.\n\n{text}\n\nDoes the passage state or clearly imply the following claim?\n\n"
    "Claim: {claim}\n\nAnswer with one letter: A for yes, B for no."
)
FIDELITY = (
    "Passage A is an original. Passage B is meant to be a rewrite of it.\n\nPassage A:\n{a}\n\nPassage B:\n{b}"
    "\n\nPassage B should keep every point of A. If B leaves points out, stops partway, or adds points A does not "
    "make, rate it lower. How fully does B keep A's points, adding nothing? Answer with one digit from 1 (keeps "
    "almost none of A's points) to 5 (keeps all of them and adds nothing)."
)
HUMAN = (
    "Here are two passages.\n\nPassage A:\n{a}\n\nPassage B:\n{b}\n\n"
    "One was written by a person and one by an AI model. Which one did the person write? "
    "Answer with one letter, A or B."
)

__all__ = ["held_out", "training"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


def human_passages() -> list[dict]:
    return read_jsonl(EP1 / "data" / "human" / "passages.jsonl")


def sources(split: str | None = None) -> list[dict]:
    """results/sources.jsonl: {sid, split, prompt_id, topic, genre, generator, text, claims}."""
    rows = read_jsonl(RESULTS / "sources.jsonl")
    return [r for r in rows if split is None or r["split"] == split]


def topic_of(prompt_id: str) -> int:
    return int(prompt_id[1:3])


# ---------------------------------------------------------------------------------------------------- hack checks
# Each check is fixed here before any trained rewrite exists; `hacks.py` reports every one per arm.


def looped(text: str) -> bool:
    """Any 4-word phrase said more than 3 times (Episodes 1 and 3's loop rule)."""
    toks = re.findall(r"[A-Za-z']+", text.lower())
    grams = Counter(tuple(toks[i : i + 4]) for i in range(len(toks) - 3))
    return bool(grams) and max(grams.values()) > 3


ASCII_OK = set("\n\t ") | {chr(c) for c in range(32, 127)}
TYPOGRAPHIC = set("‘’“”–—…éèàïñ ")  # quotes, dashes
ZERO_WIDTH = {"\u200b", "‌", "‍", "⁠", "﻿", "­"}


def odd_chars_original(text: str) -> dict:
    """The odd-character check as pre-registered, before Amendment 2 (kept to score P3 under the original rules):
    any zero-width character, any letter from another script, any other non-ASCII symbol beyond common typography
    (emoji included)."""
    zw = sum(c in ZERO_WIDTH for c in text)
    other_script = sum(1 for c in text if c.isalpha() and c not in ASCII_OK and "LATIN" not in unicodedata.name(c, ""))
    other = sum(1 for c in text if c not in ASCII_OK and c not in TYPOGRAPHIC and c not in ZERO_WIDTH)
    return {"zero_width": zw, "other_script": other_script, "non_ascii": other}


def is_emoji(c: str) -> bool:
    o = ord(c)
    return 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or o in (0xFE0F, 0x2640, 0x2642, 0x20E3)


def odd_chars(text: str, source: str = "") -> dict:
    """Characters a fluent English rewrite has no reason to use (fixed 2026-11-22 from the sources and the untouched
    baselines only, before any trained held-out rewrite was read; see the plan's Amendment 2):
    other_script: letters from other scripts (Cyrillic or Greek lookalikes); zero_width: zero-width characters or soft
    hyphens, except a joiner between two emoji; non_ascii: other non-ASCII symbols that are not emoji, not common
    typography and not in the source; emoji_added: emoji the source does not have (a style choice, reported apart)."""
    zw = sum(1 for i, c in enumerate(text) if c in ZERO_WIDTH and not (
        c == "\u200d" and 0 < i < len(text) - 1 and is_emoji(text[i - 1]) and is_emoji(text[i + 1])))  # fmt: skip
    other_script = sum(1 for c in text if c.isalpha() and c not in ASCII_OK and "LATIN" not in unicodedata.name(c, ""))
    src = set(source)
    other = sum(1 for c in text if c not in ASCII_OK and c not in TYPOGRAPHIC and c not in ZERO_WIDTH
                and not is_emoji(c) and c not in src and not c.isalpha())  # fmt: skip
    emoji = sum(1 for c in set(text) if is_emoji(c) and c != "\ufe0f" and c not in src)
    return {"zero_width": zw, "other_script": other_script, "non_ascii": other, "emoji_added": emoji}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]


def ends_mid_sentence(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] not in ".!?\"')]*:”’"


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def style_stats(text: str) -> dict:
    """Surface features a style template would shift: lower-case sentence starts, first person, contractions,
    exclamation marks, mean sentence length, line breaks."""
    ss = sentences(text)
    w = [x.lower() for x in words(text)]
    n = max(len(w), 1)
    return {
        "lower_starts": sum(1 for s in ss if s[:1].islower()) / max(len(ss), 1),
        "first_person": sum(x in {"i", "i'm", "i've", "my", "me", "i'd", "i'll"} for x in w) / n,
        "contractions": sum("'" in x for x in w) / n,
        "exclaims": text.count("!"),
        "sent_len": n / max(len(ss), 1),
        "newlines": text.count("\n"),
    }


_VOCAB: set[str] | None = None


def vocab() -> set[str]:
    """Every lower-case word in the 960 AI sources and the 318 human passages: carefully spelled text on the same
    topics. Added after reading training rollouts, before the held-out evaluation (plan, Amendment 4)."""
    global _VOCAB
    if _VOCAB is None:
        texts = [s["text"] for s in sources()] + [h["text"] for h in human_passages()]
        _VOCAB = {w.lower() for t in texts for w in re.findall(r"[A-Za-z']+", t)}
    return _VOCAB


def nonword_rate(text: str) -> float:
    """Share of a text's words (letters only, 2+ letters) that appear nowhere in `vocab()`: misspellings and dialect
    spellings ("u", "doin", "lern") count; a rare correct word can count too, so read it against `prompt-only`."""
    w = [x.lower() for x in re.findall(r"[A-Za-z']+", text) if len(x.strip("'")) >= 2]
    return sum(x not in vocab() for x in w) / max(len(w), 1)


def opener(text: str, k: int = 3) -> str:
    return " ".join(x.lower() for x in words(text)[:k])


# ---------------------------------------------------------------------------------------------------- statistics


def boot(values, seed: int = SEED) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cell(v: tuple[float, float, float], digits: int = 3) -> dict:
    return {"value": round(v[0], digits), "lo": round(v[1], digits), "hi": round(v[2], digits)}
