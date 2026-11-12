"""Lesson 03: turn human documents and their AI versions into labelled training rows, and throw out the bad ones.

    python 03_label.py

The label (the card's "training target") is one number per text, 0..1:
- a human document: 0;
- a mirror (written from a summary, no original words): 1;
- a rewrite or a round trip: the measured fraction of words changed (common.frac_changed).

The three bands the model is judged on come from that number: below 0.15 human, 0.75 or above AI-generated, between
the two AI-edited. So a grammar fix that changed 4% of the words counts as human, as it should: a human wrote it.

Filters, in the card's order, plus two of ours:
1. fewer than 30 prose words: drop;
2. an edit that changed under 2% of the words: drop (nothing happened);
3. an edit whose meaning drifted (embedding cosine to the source below EDIT_MIN_COS): drop;
4. a mirror that drifted off topic (cosine to the source below MIRROR_MIN_COS): drop;
5. (ours) not English, by a stop-word test: drop the text, and for a human document all its versions too;
6. (ours) decontamination: a human document sharing three or more 10-word runs with any benchmark text we test on
   (EditLens val/test/test_enron/test_llama, Human Detectors, Epoch AI) is dropped with all its versions.

Band-balanced weights: each row is weighted so that the three bands carry equal total weight in the loss.
Writes data/{train,val,test}.jsonl and results/labels.json (counts and distributions, no text).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np
from common import DATA, EDITLENS, EPOCH, RESULTS, band, editlens_split, frac_changed, prose_words, read_jsonl, words

EDIT_MIN_COS, MIRROR_MIN_COS = 0.75, 0.35
FULL_AI = ("mirror", "imitate")  # written by the model from scratch: target 1
# an imitation is on a new topic by design (shown three other texts), so it has no topic check against its anchor
MIN_CHANGE, MIN_WORDS = 0.02, 30
STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "not",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "as",
        "by",
        "from",
        "so",
        "if",
        "my",
        "your",
        "his",
        "her",
        "their",
        "our",
        "there",
        "what",
        "which",
        "who",
        "will",
        "would",
        "can",
        "could",
        "just",
        "about",
        "all",
        "more",
        "one",
    ]
)


def english(text: str) -> bool:
    w = words(text)
    letters = re.findall(r"[^\W\d_]", text)
    ascii_share = sum(c.isascii() for c in letters) / max(1, len(letters))
    return bool(w) and sum(x in STOP for x in w) / len(w) >= 0.15 and ascii_share >= 0.95


def shingles(text: str, n: int = 10) -> set[int]:
    w = words(text)
    return {hash(" ".join(w[i : i + n])) for i in range(max(0, len(w) - n + 1))}


def benchmark_shingles() -> set[int]:
    out: set[int] = set()
    for split in ("val", "test", "test_enron", "test_llama", "human_detectors"):
        df = editlens_split(split)
        for col in ("text", "source_text"):
            if col in df:
                for t in df[col].dropna():
                    out |= shingles(str(t))
    if EPOCH.exists():
        for p in EPOCH.glob("corpus/*/*/*.txt"):
            out |= shingles(p.read_text())
    return out


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)


def main() -> None:
    human = {r["id"]: r for r in read_jsonl(DATA / "human.jsonl")}
    ai = read_jsonl(DATA / "ai.jsonl")
    dropped: Counter = Counter()
    print(f"{len(human)} human documents, {len(ai)} AI versions; EditLens clone at {EDITLENS}")

    bench = benchmark_shingles()
    bad_docs = set()
    for i, d in human.items():
        if not english(d["text"]):
            bad_docs.add(i)
            dropped.update(["human: not English"])
        elif len(shingles(d["text"]) & bench) >= 3:
            bad_docs.add(i)
            dropped.update(["human: overlaps a benchmark text"])
    ai = [r for r in ai if r["doc_id"] in human and r["doc_id"] not in bad_docs]

    keep = []
    for r in ai:
        src = human[r["doc_id"]]["text"]
        if prose_words(r["text"]) < MIN_WORDS:
            dropped.update([f"{r['version']}: under {MIN_WORDS} words"])
            continue
        if not english(r["text"]):
            dropped.update([f"{r['version']}: not English"])
            continue
        r["frac"] = frac_changed(src, r["text"])
        if r["version"] not in FULL_AI and r["frac"] < MIN_CHANGE:
            dropped.update([f"{r['version']}: under 2% changed"])
            continue
        keep.append(r)
    cos = (embed([human[r["doc_id"]]["text"] for r in keep]) * embed([r["text"] for r in keep])).sum(1)
    rows = []
    for r, c in zip(keep, cos, strict=True):
        r["cos"] = float(c)
        floor = {"mirror": MIRROR_MIN_COS, "imitate": -1.0}.get(r["version"], EDIT_MIN_COS)
        if c < floor:
            dropped.update([f"{r['version']}: cosine below {floor}"])
            continue
        target = 1.0 if r["version"] in FULL_AI else round(r["frac"], 4)
        rows.append(
            {
                "text": r["text"],
                "target": target,
                "kind": r["version"],
                "source": r["source"],
                "model": r["model"],
                "family": r["family"],
                "doc_id": r["doc_id"],
                "split": r["split"],
                "instruction": r.get("instruction"),
                "lang": r.get("lang"),
                "cos": round(r["cos"], 4),
                "key": r["key"],
            }
        )
    for i, d in human.items():
        if i not in bad_docs and prose_words(d["text"]) >= MIN_WORDS:
            rows.append(
                {
                    "text": d["text"],
                    "target": 0.0,
                    "kind": "human",
                    "source": d["source"],
                    "model": "human",
                    "family": "human",
                    "doc_id": i,
                    "split": d["split"],
                    "key": f"{i}|human",
                }
            )

    bands = band(np.array([r["target"] for r in rows]))
    train_counts = Counter(int(b) for b, r in zip(bands, rows, strict=True) if r["split"] == "train")
    total = sum(train_counts.values())
    for b, r in zip(bands, rows, strict=True):
        r["band"] = int(b)
        r["weight"] = round(total / (3 * train_counts[int(b)]), 4)
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    for s, rs in by_split.items():
        with (DATA / f"{s}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)

    edits = [r for r in rows if r["kind"] in ("rewrite", "roundtrip")]
    inst = defaultdict(list)
    for r in edits:
        inst[f"rewrite {r['instruction']}" if r["kind"] == "rewrite" else f"roundtrip {r['lang']}"].append(r["target"])
    for r in rows:
        if r["kind"] in ("partial", "light"):
            inst[r["kind"]].append(r["target"])
    stats = {
        "rows": {s: len(rs) for s, rs in sorted(by_split.items())},
        "kind_by_band": {
            k: dict(Counter(r["band"] for r in rows if r["kind"] == k))
            for k in ("human", "mirror", "rewrite", "roundtrip", "partial", "light", "imitate")
        },
        "band_share_train": {str(b): round(c / total, 3) for b, c in sorted(train_counts.items())},
        "median_frac_changed": {k: round(float(np.median(v)), 3) for k, v in sorted(inst.items())},
        "families": dict(Counter(r["family"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "dropped": dict(dropped),
        "thresholds": {
            "edit_min_cos": EDIT_MIN_COS,
            "mirror_min_cos": MIRROR_MIN_COS,
            "min_change": MIN_CHANGE,
            "min_words": MIN_WORDS,
        },
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "labels.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
