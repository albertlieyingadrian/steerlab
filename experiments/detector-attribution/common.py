"""Shared pieces for detector-attribution: the corpus, words, splits, text edits and statistics.

The corpus is read, never copied by hand: AI replies from deslop-h2h's committed generations (Qwen3-8B, Track A), and
human passages from deslop's `data/human/passages.jsonl` (Project Gutenberg and Wikipedia; gitignored there, so it is
read from DESLOP_HUMAN or a steerlab checkout that has it). Each text is cut to its first 120-195 words at a sentence
end. No human text and no edited human text is written to results/; only keys, spans and scores are.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE / "data"  # gitignored scratch (edited texts sent to Pangram)
H2H = HERE.parent / "deslop-h2h"
SEED = 0
AI_ARMS = ["original", "plain", "sampler", "plain+sampler", "sft", "dpo"]
DETECTORS = ["9b", "1.7b", "voi", "desklib"]
MIN_WORDS, MAX_WORDS = 120, 195
os.environ.setdefault("MODAL_PROFILE", "albertlieyingadrian")


def _human_path() -> Path:
    env = os.environ.get("DESLOP_HUMAN")
    cands = [Path(env)] if env else []
    cands += [HERE.parent / "deslop" / "data" / "human" / "passages.jsonl"]
    cands += [p / "experiments" / "deslop" / "data" / "human" / "passages.jsonl" for p in HERE.parents]
    cands += [
        Path.home()
        / "Albert-Workspace"
        / "Random"
        / "steerlab"
        / "experiments"
        / "deslop"
        / "data"
        / "human"
        / "passages.jsonl"
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("deslop's data/human/passages.jsonl not found; set DESLOP_HUMAN")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def save_json(name: str, obj, indent: int | None = 1) -> Path:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


# ----------------------------------------------------------------------------------------------------- words, sentences
WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")
SENT_END = re.compile(r"(?<=[.!?:])[\"'”)\]*]*\s+|\n+")


def words(text: str) -> list[tuple[int, int]]:
    """Word units as character spans: a run of letters/digits (with inner apostrophes), or one punctuation mark."""
    return [m.span() for m in WORD.finditer(text)]


def word_type(text: str, span: tuple[int, int]) -> str:
    w = text[span[0] : span[1]]
    return w.lower() if w[0].isalpha() else w


def sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans (a sentence end, or a line break, closes one); empty pieces dropped."""
    out, start = [], 0
    for m in SENT_END.finditer(text):
        if text[start : m.start()].strip():
            out.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text.rstrip())))
    return out


def cut(text: str) -> str | None:
    """The longest prefix ending at a sentence end with at most MAX_WORDS words; None if under MIN_WORDS."""
    best = None
    for _, end in sentences(text):
        n = len(text[:end].split())
        if n > MAX_WORDS:
            break
        best = (end, n)
    if best is None or best[1] < MIN_WORDS:
        return None
    return text[: best[0]].strip()


def delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given character spans (word or sentence units), keeping paragraph breaks and single spaces."""
    out, pos = [], 0
    for s, e in sorted(spans):
        if s < pos:
            continue
        out.append(text[pos:s])
        pos = e
    out.append(text[pos:])
    t = "".join(out)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" +([,.;:!?])", r"\1", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def blocks(text: str) -> int:
    """Pangram bills each started 100-word block, at least one."""
    return max(1, -(-len(text.split()) // 100))


# ------------------------------------------------------------------------------------------------------------- corpus
def _split_ids(ids: list[str], salt: str) -> dict[str, str]:
    ids = sorted(ids)
    random.Random(f"{SEED}-{salt}").shuffle(ids)
    half = len(ids) // 2
    return {i: ("discovery" if k < half else "confirm") for k, i in enumerate(ids)}


def corpus() -> list[dict]:
    """Every cut text: {key, kind ('ai'|'human'), arm (or source), prompt, split, text}. Deterministic."""
    rows: list[dict] = []
    ai = {arm: read_jsonl(H2H / "results" / "gen" / f"{arm}.jsonl") for arm in AI_ARMS}
    split = _split_ids(list({r["id"] for r in ai["original"]}), "prompts")
    for arm, rs in ai.items():
        for r in rs:
            t = cut(r["text"])
            if t:
                rows.append(
                    {
                        "key": f"ai/{arm}/{r['id']}/{r['sample']}",
                        "kind": "ai",
                        "arm": arm,
                        "prompt": r["id"],
                        "split": split[r["id"]],
                        "text": t,
                    }
                )
    hum = read_jsonl(_human_path())
    hsplit = _split_ids([f"{h['source']}/{h['id']}" for h in hum], "human")
    for h in hum:
        t = cut(h["text"])
        k = f"{h['source']}/{h['id']}"
        if t:
            rows.append(
                {"key": f"human/{k}", "kind": "human", "arm": h["source"], "prompt": k, "split": hsplit[k], "text": t}
            )
    return rows


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------- statistics
def boot_mean(x, n: int = 10_000, seed: int = SEED) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))
