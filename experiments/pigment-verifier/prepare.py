"""
The fixed half of the reward-method search: data, labels, evaluation and the spending cap. Not edited
during the search (like prepare.py in Karpathy's autoresearch); reward.py is the file that changes.

Run from the repo root:
    .venv/bin/python experiments/pigment-verifier/prepare.py           # render, sample pairs, label them

What it builds (data/ is small and committed; renders live in the work folder):
1. Every sketch we already have: the 12 hand-made references, 320 laptop and 800 H100 training rollouts,
   and 224 benchmark samples. Rendered once, gated, and de-duplicated by image.
2. Sketches split 60/40 into dev and test by image hash, so no image appears on both sides.
3. Random pairs inside each split, labelled by two frontier judges (GPT-5.6 Sol, Opus 5.5), each pair asked
   in both orders. A pair is kept only when all four answers agree. That is the "ground truth", with the
   caveat that it is two judges' taste, not a person's.

evaluate(scores) is the one metric the search is allowed to optimise: how often a reward method orders a
dev pair the same way as the judges (a tie counts as half). The test split is only read with --final.
"""

import argparse
import base64
import hashlib
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "pigment-gym"))

from brush_tools import extract_code
from hand_pool import POOL, SUBJECT

DATA = HERE / "data"
WORK = Path.home() / ".cache" / "steerlab" / "pigment-verifier"
RENDERS = WORK / "renders"
LEDGER = DATA / "ledger.jsonl"
ANSWERS = DATA / "label_answers.jsonl"  # cache: a rerun never pays for the same question twice
BUDGET_USD = 25.0  # hard cap for the whole search, labels included
PAIRS = {"dev": 300, "test": 200}
URL = "https://openrouter.ai/api/v1/chat/completions"
LABEL_JUDGES = {
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "opus-5.5": "anthropic/claude-opus-5.5",
}
PAIR_PROMPT = (
    f"Two images follow: A, then B. Which is the better watercolour painting of {SUBJECT}? "
    "Weigh whether the subject is recognisable and how well it is painted.\n"
    'Reply with JSON only: {"better": "A"} or {"better": "B"}'
)


# -- spending cap -------------------------------------------------------------------------------------------
_LOCK = threading.Lock()


def spent():
    """Total paid so far: one JSON line per paid call in data/ledger.jsonl (append-only, so threads can't corrupt
    it)."""
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())


def charge(usd, what):
    """Every paid call goes through here. Stops the process once the budget is used up."""
    with _LOCK:
        DATA.mkdir(exist_ok=True)
        with LEDGER.open("a") as handle:
            handle.write(json.dumps({"usd": round(usd, 6), "what": what}) + "\n")
        total = spent()
    if total > BUDGET_USD:
        sys.exit(f"budget of ${BUDGET_USD} used up (${total:.2f}); stopping")


def secret(name):
    """An API key from the repo's .env files (or the folder named by $STEERLAB_ENV_DIR)."""
    import os

    from dotenv import dotenv_values

    folder = Path(os.environ.get("STEERLAB_ENV_DIR", REPO))
    values = {**dotenv_values(folder / ".env"), **dotenv_values(folder / ".env.local")}
    return values[name]


def data_url(sketch_id):
    return "data:image/png;base64," + base64.b64encode(load_png(sketch_id)).decode()


# -- sketches -----------------------------------------------------------------------------------------------
def png_path(sketch_id):
    return RENDERS / f"{sketch_id}.png"


def load_png(sketch_id):
    return png_path(sketch_id).read_bytes()


def load_sketches():
    return json.loads((DATA / "sketches.json").read_text())


def gather_codes():
    pg = REPO / "experiments" / "pigment-gym" / "results"
    out = [(f"hand-{item['rank']:g}-{i}", "hand", item["code"]) for i, item in enumerate(POOL)]
    for name in ("laptop", "h100"):
        for line in (pg / f"rollouts_{name}.jsonl").read_text().splitlines():
            row = json.loads(line)
            out.append((None, f"{name}-step{row['step']}", row["code"]))
    for line in (pg / "benchmark_samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        out.append((None, f"bench-{row['model']}", extract_code(row["text"])))
    return out


def build_sketches():
    from brush_tools import Renderer, gate

    RENDERS.mkdir(parents=True, exist_ok=True)
    codes = gather_codes()
    unique = {}
    for given_id, source, code in codes:
        key = given_id or "s-" + hashlib.sha1(code.encode()).hexdigest()[:12]
        unique.setdefault(key, (source, code))
    keys = list(unique)
    print(f"{len(codes)} sketches, {len(keys)} with distinct code; rendering")
    with Renderer(pool_size=6) as renderer:
        renders = renderer.render_many([unique[k][1] for k in keys])
    sketches, seen_images = [], set()
    for key, render in zip(keys, renders, strict=True):
        source, code = unique[key]
        passed = gate(code, render).passed
        image_hash = hashlib.sha1(render.png).hexdigest()[:12] if render.png else None
        is_hand = key.startswith("hand-")
        if passed and not is_hand:
            if image_hash in seen_images:
                continue  # the same picture from different code: keep one
            seen_images.add(image_hash)
        if render.png:
            png_path(key).write_bytes(render.png)
        split = "hand" if is_hand else ("dev" if int(image_hash or "0", 16) % 10 < 6 else "test")
        sketches.append(
            {
                "id": key,
                "source": source,
                "gate": passed,
                "split": split,
                "image": image_hash,
            }
        )
    (DATA / "sketches.json").write_text(json.dumps(sketches, indent=1))
    kept = [s for s in sketches if s["gate"] and s["split"] != "hand"]
    dev = sum(s["split"] == "dev" for s in kept)
    print(f"{len(kept)} distinct passing images: {dev} dev, {len(kept) - dev} test")


# -- labels -------------------------------------------------------------------------------------------------
_ANSWERS = (
    {json.loads(line)["key"]: json.loads(line)["winner"] for line in ANSWERS.read_text().splitlines()}
    if ANSWERS.exists()
    else {}
)


def ask_pair(judge, first, second, api_key):
    """One judge, one ordered pair. Returns the id it prefers, or None if it never answers cleanly."""
    key = f"{judge}|{first}|{second}"
    if key in _ANSWERS:
        return _ANSWERS[key]
    content = [
        {"type": "text", "text": "Image A:"},
        {"type": "image_url", "image_url": {"url": data_url(first)}},
        {"type": "text", "text": "Image B:"},
        {"type": "image_url", "image_url": {"url": data_url(second)}},
        {"type": "text", "text": PAIR_PROMPT},
    ]
    body = {
        "model": LABEL_JUDGES[judge],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"effort": "low", "exclude": True},
        "usage": {"include": True},
    }
    for attempt in range(3):
        try:
            response = requests.post(
                URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=120,
            )
            data = response.json()
            charge(data.get("usage", {}).get("cost", 0.0) or 0.0, f"label {judge}")
            text = data["choices"][0]["message"]["content"] or ""
            better = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))["better"].strip().upper()
            if better in ("A", "B"):
                winner = first if better == "A" else second
                with _LOCK:
                    _ANSWERS[key] = winner
                    with ANSWERS.open("a") as handle:
                        handle.write(json.dumps({"key": key, "winner": winner}) + "\n")
                return winner
        except (
            requests.RequestException,
            KeyError,
            ValueError,
            AttributeError,
            TypeError,
        ) as error:
            print(f"  {judge} {first} vs {second}: attempt {attempt + 1} failed ({type(error).__name__})")
            time.sleep(3 * (attempt + 1))
    return None


def label_pairs():
    api_key = secret("OPENROUTER_API_KEY")
    sketches = [s for s in load_sketches() if s["gate"]]
    rng = random.Random(0)
    for split, count in PAIRS.items():
        path = DATA / f"pairs_{split}.jsonl"
        if path.exists():
            print(f"{path.name} exists; skipping")
            continue
        ids = sorted(s["id"] for s in sketches if s["split"] == split)
        pairs = set()
        while len(pairs) < count:
            pairs.add(tuple(sorted(rng.sample(ids, 2))))
        pairs = sorted(pairs)
        jobs = [(judge, x, y) for a, b in pairs for judge in LABEL_JUDGES for x, y in ((a, b), (b, a))]
        with ThreadPoolExecutor(8) as pool:
            answers = list(pool.map(lambda job: ask_pair(*job, api_key), jobs))
        rows = []
        for i, (a, b) in enumerate(pairs):
            votes = answers[4 * i : 4 * i + 4]
            winner = votes[0] if votes[0] is not None and all(v == votes[0] for v in votes) else None
            rows.append({"a": a, "b": b, "winner": winner, "votes": votes})
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        agreed = sum(r["winner"] is not None for r in rows)
        print(f"{split}: {agreed} of {count} pairs agreed by both judges in both orders; spent so far ${spent():.2f}")


# -- the metric ---------------------------------------------------------------------------------------------
def load_pairs(split):
    rows = [json.loads(line) for line in (DATA / f"pairs_{split}.jsonl").read_text().splitlines()]
    return [r for r in rows if r["winner"]]


def hand_ids():
    return [s["id"] for s in load_sketches() if s["split"] == "hand" and s["gate"]]


def evaluation_ids(split):
    """Every sketch a reward method must score for this split: both sides of each labelled pair, plus the
    hand-made sketches that pass the gate."""
    return sorted({i for r in load_pairs(split) for i in (r["a"], r["b"])}) + hand_ids()


def evaluate(scores, split="dev"):
    """scores: {sketch id: float}. Returns pairwise agreement with the judges (ties count half), the same on
    the hard pairs whose two sketches share a source (acc_within), the tie rate, and Spearman with the intended order
    on the hand-made sketches that pass the gate."""
    from scipy.stats import spearmanr

    pairs = load_pairs(split)
    family = {s["id"]: s["source"].split("-")[0] for s in load_sketches()}
    hits = ties = within_hits = within = 0.0
    for r in pairs:
        loser = r["b"] if r["winner"] == r["a"] else r["a"]
        gap = scores[r["winner"]] - scores[loser]
        hit = 0.5 if abs(gap) < 1e-6 else float(gap > 0)
        hits += hit
        ties += abs(gap) < 1e-6
        if family[r["a"]] == family[r["b"]]:  # both from one training run, or both from the benchmark: the hard pairs
            within += 1
            within_hits += hit
    hand = hand_ids()
    intended = [-float(h.split("-")[1]) for h in hand]
    return {
        "acc": hits / len(pairs),
        "acc_within": within_hits / within,
        "tie_rate": ties / len(pairs),
        "hand_rho": float(spearmanr([scores[h] for h in hand], intended).statistic),
        "pairs": len(pairs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the data, labels and metric for the reward search")
    parser.add_argument("--skip-render", action="store_true", help="reuse data/sketches.json")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if not (args.skip_render and (DATA / "sketches.json").exists()):
        build_sketches()
    label_pairs()
