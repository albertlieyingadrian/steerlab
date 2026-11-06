"""
Frontier models as the judge, in place of hand ratings.

Run from the repo root (needs OPENROUTER_API_KEY in .env or .env.local; well under $1 per run):
    .venv/bin/python experiments/pigment-gym/10_judge.py --run-name laptop
    .venv/bin/python experiments/pigment-gym/10_judge.py --run-name h100 \
        --rollouts the work folder full/rollouts.jsonl
Then feed each ratings file to script 3:
    .venv/bin/python experiments/pigment-gym/11_rerate.py --ratings <judge csv> [--rollouts ...]

Script 3 needs someone to rate 32 of the policy's sketches. A judge model does it here, under fixed rules:
  1. Check the judge first. It rates the 8 hand-made sketches that pass the gate (the reward check (4_rank_check.py)'s
  pool), and we compare
     its order with the hand ranking. A judge that can't tell those apart has no business grading the policy.
  2. Rate the same 32 sketches script 3 picks (analysis_tools.rerate_sample), one image per request, with no
     reward, step or ID shown, at temperature 0.
  3. Two judges from different labs, so we can see how much the judges agree with each other.
The rubric is the one used for the earlier ratings, written down before any judge saw a sketch.
"""

import argparse
import base64
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from analysis_tools import load_rollouts, rerate_sample
from brush_tools import Renderer, gate
from common import RESULTS, check, outputs_dir
from dotenv import dotenv_values
from hand_pool import POOL, SUBJECT
from scipy.stats import spearmanr

HERE_DIR = Path(__file__).resolve().parent
REPO = HERE_DIR.parents[1]
URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGES = {"gpt-5.6-sol": "openai/gpt-5.6-sol", "opus-5.5": "anthropic/claude-opus-5.5"}
RUBRIC = (
    f"How good is this image as a watercolour painting of {SUBJECT}? Rate it from 1 to 5:\n"
    "1 = no recognisable picture\n"
    "2 = a wash of colour and a disc, nothing more\n"
    "3 = sunset colours and a sun\n"
    "4 = a sun and a green hill shape\n"
    "5 = a convincing sunset over green hills\n"
    'Reply with JSON only: {"rating": <1-5>, "note": "<what you see, under 12 words>"}'
)

parser = argparse.ArgumentParser(description="Rate the policy's sketches with frontier judges")
parser.add_argument("--rollouts", type=Path, default=RESULTS / "rollouts_laptop.jsonl")
parser.add_argument("--run-name", required=True, help="a label for the output files, e.g. laptop or h100")
args = parser.parse_args()

keys = {**dotenv_values(REPO / ".env"), **dotenv_values(REPO / ".env.local")}
api_key = keys.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("OPENROUTER_API_KEY not found in .env or .env.local")
cache_path = outputs_dir() / f"judge_{args.run_name}.jsonl"
cache = {}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        record = json.loads(line)
        cache[(record["judge"], record["id"])] = record


def ask(judge, item_id, png):
    """One image, one rating. Cached by (judge, item id), so a rerun costs nothing."""
    if (judge, item_id) in cache:
        return cache[(judge, item_id)]
    image = "data:image/png;base64," + base64.b64encode(png).decode()
    body = {
        "model": JUDGES[judge],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": RUBRIC}],
            }
        ],
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"effort": "low", "exclude": True},
        "usage": {"include": True},
    }
    for attempt in range(3):
        try:
            data = requests.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=120).json()
            text = data["choices"][0]["message"]["content"] or ""
            parsed = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
            rating = int(parsed["rating"])
            if 1 <= rating <= 5:
                return {
                    "judge": judge,
                    "id": item_id,
                    "rating": rating,
                    "note": str(parsed.get("note", ""))[:80],
                    "cost_usd": data.get("usage", {}).get("cost", 0.0),
                }
        except (requests.RequestException, KeyError, ValueError, AttributeError, TypeError) as error:
            print(f"  {judge} {item_id}: attempt {attempt + 1} failed ({type(error).__name__})")
            time.sleep(3 * (attempt + 1))
    sys.exit(f"{judge} could not rate {item_id} after 3 attempts")


def rate_all(items):
    """items: [(item_id, png)] -> {judge: {item_id: record}}; new answers are appended to the cache file."""
    jobs = [(judge, item_id, png) for judge in JUDGES for item_id, png in items]
    out = {judge: {} for judge in JUDGES}
    with ThreadPoolExecutor(8) as pool, cache_path.open("a") as handle:
        for (judge, item_id, _), record in zip(jobs, pool.map(lambda job: ask(*job), jobs), strict=False):
            if (judge, item_id) not in cache:
                cache[(judge, item_id)] = record
                handle.write(json.dumps(record) + "\n")
            out[judge][item_id] = record
    return out


# 1. Check each judge on the hand-made pool, where we know the right order.
with Renderer(pool_size=4) as renderer:
    pool_renders = renderer.render_many([item["code"] for item in POOL])
    passing = [
        (item, render) for item, render in zip(POOL, pool_renders, strict=False) if gate(item["code"], render).passed
    ]
    rows = load_rollouts(args.rollouts)
    sample, ids, half = rerate_sample(rows)
    sample_renders = renderer.render_many([row["code"] for row in sample])

pool_ratings = rate_all([(f"pool-{item['rank']}", render.png) for item, render in passing])
print(f"Judges checked on the {len(passing)} hand-made sketches that pass the gate (hand rank 1 = best):")
calibration = {}
for judge in JUDGES:
    ratings = [pool_ratings[judge][f"pool-{item['rank']}"]["rating"] for item, _ in passing]
    rho = spearmanr([-item["rank"] for item, _ in passing], ratings).statistic
    calibration[judge] = rho
    print(f"  {judge:12} ratings by hand rank {ratings}   Spearman with the hand ranking {rho:+.2f}")

# 2. Rate the 32 policy sketches and write one ratings file per judge, in script 3's format.
sample_ratings = rate_all([(sketch_id, render.png) for sketch_id, render in zip(ids, sample_renders, strict=False)])
files = {}
for judge, model in JUDGES.items():
    path = RESULTS / f"judge_{judge}_{args.run_name}.csv"
    with path.open("w", newline="") as handle:
        handle.write(
            f"# Ratings by {model} via OpenRouter, one image per request at temperature 0, with no\n"
            f"# reward, step or ID shown. Written by 10_judge.py from {args.rollouts.name}. Rubric:\n"
        )
        handle.write("".join(f"#   {line}\n" for line in RUBRIC.splitlines()[:6]))
        writer = csv.writer(handle)
        writer.writerow(["id", "rating_1_to_5", "note"])
        for sketch_id in ids:
            record = sample_ratings[judge][sketch_id]
            writer.writerow([sketch_id, record["rating"], record["note"]])
    files[judge] = path
    first = [sample_ratings[judge][i]["rating"] for i, row in zip(ids, sample, strict=False) if row["step"] < half]
    second = [sample_ratings[judge][i]["rating"] for i, row in zip(ids, sample, strict=False) if row["step"] >= half]
    print(
        f"\n{judge}: mean rating {sum(first) / len(first):.2f} (first half) -> {sum(second) / len(second):.2f} "
        f"(second half); saved {path.name}"
    )

a, b = list(JUDGES)
agreement = spearmanr(
    [sample_ratings[a][i]["rating"] for i in ids], [sample_ratings[b][i]["rating"] for i in ids]
).statistic
exact = sum(sample_ratings[a][i]["rating"] == sample_ratings[b][i]["rating"] for i in ids)
spent = sum(record.get("cost_usd") or 0.0 for record in cache.values())
print(f"\njudge vs judge on the 32 sketches: Spearman {agreement:+.2f}, same rating on {exact} of 32")
print(f"total spent on judging so far (this run name): ${spent:.2f}")
print(
    f"next: 11_rerate.py --ratings {files[a].name} (and {files[b].name})"
    + ("" if args.rollouts.name == "rollouts.jsonl" else f" --rollouts {args.rollouts.name}")
)

print("\nSelf-checks:")
check(
    "every judge rated every hand-made and policy sketch",
    all(len(pool_ratings[j]) == len(passing) and len(sample_ratings[j]) == 32 for j in JUDGES),
)
check(
    "each judge orders the hand-made sketches roughly like the hand ranking (Spearman > 0.5)",
    all(rho > 0.5 for rho in calibration.values()),
)
