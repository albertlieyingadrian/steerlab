"""Score every text in the note with three open AI-text detectors. A measurement only: no method, setting, teacher
reply or rewrite was ever chosen by a detector's score.

    python detect.py mirrors     # 40 human passages lightly edited by the original Qwen3-4B: a known "AI-edited" set
    python detect.py score       # every held-out reply, human passage, author passage, voice rewrite and mirror

Known-answer sets, after Pangram's and GPTZero's habits: 318 human passages and 100 held-out passages by the five
voice authors (all should read as human; the false-positive set), the original model's 48 replies (all AI), and 40
"mirrors" (human passages the original model lightly edited, so the content is human and the surface is not).
Writes results/detect/scores.json: {key: {binoculars, fastdetect, desklib, tokens}}, with keys run/<run>/<prompt>,
human/<source>/<id>, author/<author>/<id>, voice/<rewrite id>, mirror/<id>.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import RUNS
from prompts import held_out

OUT = HERE / "results" / "detect"
MIRROR_REQUEST = (
    "Lightly edit this passage for clarity and flow. Keep its meaning, its facts and roughly its length. "
    "Reply with the edited passage only.\n\n"
)


def human_rows() -> list[dict]:
    return [json.loads(line) for line in (HERE / "data" / "human" / "passages.jsonl").open(encoding="utf-8")]


def mirrors() -> None:
    rows = human_rows()
    rng = random.Random(1)
    picked = rng.sample([r for r in rows if r["source"] == "gutenberg"], 20)
    picked += rng.sample([r for r in rows if r["source"] == "wikipedia"], 20)
    runner = modal.Cls.from_name("deslop", "Runner")()
    job = {"rows": [{"id": r["id"], "prompt": MIRROR_REQUEST + r["text"]} for r in picked], "n": 1,
           "temperature": 0.7, "max_new_tokens": 600, "seed": 0}  # fmt: skip
    out = runner.generate.remote(job)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "mirrors.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({"id": r["id"], "source_id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"{len(out)} mirrors")


def texts() -> dict[str, str]:
    out: dict[str, str] = {}
    ids = {p["id"] for p in held_out()}
    for run in RUNS:
        path = HERE / "results" / "gen" / f"{run}.jsonl"
        if path.exists():
            for r in map(json.loads, path.open(encoding="utf-8")):
                if r["id"] in ids:
                    out[f"run/{run}/{r['id']}"] = r["text"]
    for r in human_rows():
        out[f"human/{r['source']}/{r['id']}"] = r["text"]
    for r in map(json.loads, (HERE / "data" / "voices" / "passages.jsonl").open(encoding="utf-8")):
        if r["role"] == "heldout":
            out[f"author/{r['author']}/{r['id']}"] = r["text"]
    rewrites = HERE / "results" / "voices" / "rewrites.jsonl"
    if rewrites.exists():
        for r in map(json.loads, rewrites.open(encoding="utf-8")):
            out[f"voice/{r['id']}"] = r["text"]
    if (OUT / "mirrors.jsonl").exists():
        for r in map(json.loads, (OUT / "mirrors.jsonl").open(encoding="utf-8")):
            out[f"mirror/{r['id']}"] = r["text"]
    return out


def score() -> None:
    items = texts()
    keys = list(items)
    chunks = [keys[i : i + 120] for i in range(0, len(keys), 120)]
    detectors = modal.Cls.from_name("deslop", "Detectors")()
    scores: dict[str, dict] = {}
    for chunk, result in zip(chunks, detectors.score.map([[items[k] for k in c] for c in chunks]), strict=True):
        scores.update(dict(zip(chunk, result, strict=True)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(scores) + "\n")
    print(f"scored {len(scores)} texts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["mirrors", "score"])
    {"mirrors": mirrors, "score": score}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
