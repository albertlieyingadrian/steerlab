"""
True quality for the follow-up study: two frontier judges rate each arm's sketches, blind to the arm.

Run from the repo root after modal_next.py has sampled the arms (needs ANTHROPIC_API_KEY and OPENAI_API_KEY in .env;
a few dollars, capped by BUDGET_USD):
    .venv/bin/python experiments/pigment-gym/13_judge_next.py --smoke     # a few sketches + the judge check
    .venv/bin/python experiments/pigment-gym/13_judge_next.py

Fixed in the pre-registration (listed in the note's "How the study was run"):
  * judges Claude Opus 5.5 (Anthropic API) and GPT-5.6 Sol (OpenAI API), effort low, one image per request, the
    subject named, nothing about the arm, reward or method;
  * a sketch that fails the gate is rated 1 without asking; identical images share a rating (cached by hash);
  * check (a): each judge rates the 8 passing hand-made sunset sketches; Spearman with the intended order > 0.5;
  * check (b): 16 passing `sft` sketches rated against a different held-out subject's name (subject swap).
Every answer is cached in results/next/judge_cache.jsonl, so a rerun only pays for new images.
Writes results/next/ratings.csv (one row per arm x subject x group) and results/next/judge_checks.json.
"""

import argparse
import base64
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import requests
from brush_tools import Renderer, gate
from common import RESULTS, check, outputs_dir
from hand_pool import POOL
from next_common import EVAL_SUBJECTS, FAIL_RATING, HELD_OUT, IN_DISTRIBUTION, RUBRIC, load_keys, png_key
from scipy.stats import spearmanr

OUT = RESULTS / "next"
PNG_DIR = outputs_dir() / "next" / "png"
JUDGES = {"opus-5.5": "claude-opus-5-5", "gpt-5.6-sol": "gpt-5.6-sol"}
PRICES = {"opus-5.5": (4.00, 20.00), "gpt-5.6-sol": (4.00, 20.00)}  # USD per million in, out (list price, 2026-11-12)
BUDGET_USD = 15.0
# Arms: (sample file, which sample of each group).
ARMS = {
    "base": ("base", "first"),
    "bon8": ("base", "best"),
    "grpo": ("grpo", "first"),
    "sft": ("sft", "first"),
    "sft_bon8": ("sft", "best"),
    "sft_grpo": ("sft_grpo", "first"),  # exploratory
    "opus": ("opus", "first"),  # reference ceiling, added after the arms were judged (17_frontier_ceiling.py)
}

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
tag = "_smoke" if args.smoke else ""

keys = load_keys()
openai_key = keys.get("OPENAI_API_KEY")
if not keys.get("ANTHROPIC_API_KEY") or not openai_key:
    sys.exit("ANTHROPIC_API_KEY and OPENAI_API_KEY are both needed in .env")
claude = anthropic.Anthropic(api_key=keys["ANTHROPIC_API_KEY"], max_retries=4)

cache_path = OUT / "judge_cache.jsonl"
cache = {}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        record = json.loads(line)
        cache[(record["judge"], record["subject"], record["png"])] = record
lock = threading.Lock()


def spent():
    return sum(r["cost_usd"] for r in cache.values())


def parse(text):
    parsed = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
    rating = int(parsed["rating"])
    if not 1 <= rating <= 5:
        raise ValueError(rating)
    return rating, str(parsed.get("note", ""))[:80]


def ask_opus(png, subject):
    image = {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}
    response = claude.messages.create(
        model=JUDGES["opus-5.5"],
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": image},
                    {"type": "text", "text": RUBRIC.format(subject=subject)},
                ],
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


def ask_gpt(png, subject):
    image = {"url": "data:image/png;base64," + base64.b64encode(png).decode()}
    body = {
        "model": JUDGES["gpt-5.6-sol"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": RUBRIC.format(subject=subject)},
                ],
            }
        ],
        "reasoning_effort": "low",
        "max_completion_tokens": 4000,
    }
    data = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {openai_key}"},
        json=body,
        timeout=180,
    ).json()
    if "choices" not in data:
        raise RuntimeError(str(data.get("error", {}).get("message", "no choices"))[:200])
    usage = data.get("usage", {})
    return (
        data["choices"][0]["message"]["content"] or "",
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


def rate(judge, subject, png):
    key = (judge, subject, png_key(png))
    if key in cache:
        return cache[key]
    if spent() > BUDGET_USD:
        sys.exit(f"judging budget ${BUDGET_USD} reached")
    for attempt in range(4):
        try:
            text, tokens_in, tokens_out = (ask_opus if judge == "opus-5.5" else ask_gpt)(png, subject)
            rating, note = parse(text)
            price = PRICES[judge]
            record = {
                "judge": judge,
                "subject": subject,
                "png": key[2],
                "rating": rating,
                "note": note,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": (tokens_in * price[0] + tokens_out * price[1]) / 1e6,
            }
            with lock:
                cache[key] = record
                with cache_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            return record
        except (
            anthropic.APIError,
            requests.RequestException,
            RuntimeError,
            ValueError,
            AttributeError,
            KeyError,
        ) as error:
            print(f"  {judge} {key[2]}: attempt {attempt + 1} failed ({type(error).__name__}: {str(error)[:100]})")
            time.sleep(4 * (attempt + 1))
    sys.exit(f"{judge} could not rate {key[2]}")


def rate_many(jobs):
    """jobs: [(subject, png)]; both judges rate each, 8 requests at a time."""
    work = [(j, s, p) for j in JUDGES for s, p in jobs]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda job: rate(*job), work))


def rating(judge, subject, h):
    return cache[(judge, subject, h)]["rating"]


# Check (a): the judges on the hand-made sunset sketches with a known order.
with Renderer(pool_size=4) as renderer:
    renders = renderer.render_many([item["code"] for item in POOL])
passing = [(item, r) for item, r in zip(POOL, renders, strict=True) if gate(item["code"], r).passed]
rate_many([(IN_DISTRIBUTION, r.png) for _, r in passing])
checks = {"hand_pool": {}}
print(f"Check (a): {len(passing)} passing hand-made sketches, rank 1 = best")
for judge in JUDGES:
    ratings = [rating(judge, IN_DISTRIBUTION, png_key(r.png)) for _, r in passing]
    rho = spearmanr([-item["rank"] for item, _ in passing], ratings).statistic
    checks["hand_pool"][judge] = {"ratings": ratings, "spearman": rho}
    print(f"  {judge:12} {ratings}  Spearman {rho:+.2f}")

# The arms' sketches.
samples = {}
for name in sorted({source for source, _ in ARMS.values()}):
    path = OUT / f"samples_{name}{tag}.jsonl"
    if path.exists():
        samples[name] = [json.loads(line) for line in path.read_text().splitlines()]
picked = []  # (arm, row)
for arm, (source, which) in ARMS.items():
    for row in samples.get(source, []):
        if (which == "first" and row["slot"] == 0) or (which == "best" and row["is_best"]):
            picked.append((arm, row))
jobs = {(row["subject"], row["png"]) for _, row in picked if row["gate"]}
print(f"\n{len(picked)} arm sketches, {len(jobs)} distinct passing images to judge; spent so far ${spent():.2f}")
rate_many([(s, (PNG_DIR / f"{h}.png").read_bytes()) for s, h in sorted(jobs)])

# Check (b): subject swap on passing sft sketches, 2 per held-out subject.
swap = []
for i, subject in enumerate(HELD_OUT):
    rows = [r for a, r in picked if a == "sft" and r["subject"] == subject and r["gate"]][:2]
    swap += [(r, HELD_OUT[(i + 1) % len(HELD_OUT)]) for r in rows]
rate_many([(other, (PNG_DIR / f"{r['png']}.png").read_bytes()) for r, other in swap])
checks["subject_swap"] = {}
for judge in JUDGES:
    matched = [rating(judge, r["subject"], r["png"]) for r, _ in swap]
    swapped = [rating(judge, other, r["png"]) for r, other in swap]
    checks["subject_swap"][judge] = {"n": len(swap), "matched": matched, "swapped": swapped}
    if swap:
        print(
            f"  subject swap, {judge}: matched {sum(matched) / len(swap):.2f}, swapped {sum(swapped) / len(swap):.2f}"
        )
(OUT / f"judge_checks{tag}.json").write_text(json.dumps(checks, indent=1))

header = [
    "arm",
    "subject",
    "subject_index",
    "held_out",
    "group",
    "slot",
    "gate",
    "reward",
    "clip_raw",
    "aesthetic_raw",
    "similarity",
    "tokens",
    "png",
    "opus",
    "gpt",
    "score",
]
with (OUT / f"ratings{tag}.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(header)
    for arm, row in picked:
        if row["gate"]:
            opus, gpt = (
                rating("opus-5.5", row["subject"], row["png"]),
                rating("gpt-5.6-sol", row["subject"], row["png"]),
            )
        else:
            opus = gpt = FAIL_RATING
        writer.writerow(
            [
                arm,
                row["subject"],
                row["subject_index"],
                int(row["subject"] != IN_DISTRIBUTION),
                row["group"],
                row["slot"],
                int(row["gate"]),
                f"{row['reward']:.4f}",
                row["clip_raw"],
                row["aesthetic_raw"],
                f"{row['similarity']:.3f}",
                row["tokens"],
                row["png"],
                opus,
                gpt,
                (opus + gpt) / 2,
            ]
        )
by_judge = {j: sum(r["cost_usd"] for r in cache.values() if r["judge"] == j) for j in JUDGES}
print(f"\njudging spend so far: {', '.join(f'{j} ${c:.2f}' for j, c in by_judge.items())} (total ${spent():.2f})")

print("\nSelf-checks:")
check(
    "each judge orders the hand-made sketches like the intended order (Spearman > 0.5)",
    all(v["spearman"] > 0.5 for v in checks["hand_pool"].values()),
)
check(
    "every passing arm sketch has both ratings",
    all((j, r["subject"], r["png"]) in cache for j in JUDGES for _, r in picked if r["gate"]),
)
check(f"stayed within the ${BUDGET_USD:.0f} judging budget", spent() <= BUDGET_USD)
check("every evaluation subject is covered", args.smoke or {r["subject"] for _, r in picked} == set(EVAL_SUBJECTS))
