"""
E1: GRPO on Qwen2.5-Coder-0.5B with a chosen reward method, same setup as Pigment Gym's laptop run.

Run from the repo root (about 30 minutes each on an M-series Mac):
    .venv/bin/python experiments/pigment-verifier/train.py --name clip   --method clip_aesthetic
    .venv/bin/python experiments/pigment-verifier/train.py --name judge  --reward-file runs/NNN_reward.py --discrete
    .venv/bin/python experiments/pigment-verifier/train.py --name verify --reward-file runs/NNN_reward.py

The reward keeps Pigment Gym's structure and swaps only the quality term:
    reward = 0.10 gate + 0.05 length + gate * 0.85 * quality  (+ partial credit for failing sketches)
where quality is the chosen method's score (clip_aesthetic reproduces Pigment Gym's own 0.40 CLIP + 0.45 aesthetic).

Logged every step: mean reward, gate pass rate, and the share of groups whose 8 rewards are all tied (zero spread,
so GRPO gets no signal). Before and after training, 16 fixed-seed sketches are rated 1-5 by two frontier judges
(GPT-5.6 Sol, Opus 5.5) with Pigment Gym's rubric: a yardstick that none of the three rewards uses.
"""

import argparse
import bisect
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parents[1] / "experiments" / "pigment-gym"
sys.path[:0] = [str(HERE), str(PG)]

import requests
import torch
from brush_tools import PROMPT_EXAMPLE, SYSTEM_PROMPT, Renderer, extract_code
from common import pick_device
from grpo_tools import build_lora_model, chat_prompt, group_advantages, grpo_update, mean, sample_group
from hand_pool import SUBJECT
from reward_tools import WEIGHTS, Scorer, length_score, score_rollouts, similarity

import prepare

parser = argparse.ArgumentParser(description="GRPO with a chosen reward method")
parser.add_argument("--name", required=True)
parser.add_argument("--method", default=None, help="override reward.METHOD (e.g. clip_aesthetic)")
parser.add_argument("--reward-file", default=None, help="a runs/NNN_reward.py snapshot to use as the method")
parser.add_argument("--discrete", action="store_true", help="most likely letter instead of the expected score")
parser.add_argument("--steps", type=int, default=40)
args = parser.parse_args()

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
GROUP_SIZE, MAX_NEW_TOKENS, LEARNING_RATE, BETA, EVAL_SAMPLES, SEED = 8, 640, 2e-4, 0.04, 16, 0
EXAMPLE_CODE = extract_code(PROMPT_EXAMPLE)
OUT = HERE / "results" / args.name
OUT.mkdir(parents=True, exist_ok=True)
prepare.RENDERS.mkdir(parents=True, exist_ok=True)

# The reward method: a snapshot from the search, or reward.py as it is.
path = HERE / args.reward_file if args.reward_file else HERE / "reward.py"
spec = importlib.util.spec_from_file_location("method", path)
method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method)
if args.method:
    method.METHOD = args.method
if args.discrete:
    method.EXPECTED = False
clip_scorer = Scorer()  # only for Pigment Gym's own bookkeeping columns; the reward uses `method`

# The quality term must lie in [0, 1], like Pigment Gym's, so a passing sketch always beats a failing one.
# clip_aesthetic returns 0.40 CLIP + 0.45 aesthetic (at most 0.85): divide by 0.85 to reproduce Pigment Gym exactly.
# A method with unbounded scores (log-odds) is mapped to its percentile among the dev sketches' scores: no labels are
# used, and the order, hence every agreement number, is unchanged.
DEV_SCORES = None
if method.METHOD != "clip_aesthetic":
    dev = sorted(method.score(prepare.evaluation_ids("dev")).values())
    if dev[0] < 0 or dev[-1] > 1:
        DEV_SCORES = dev
        print(f"quality = percentile among {len(dev)} dev scores (raw range {dev[0]:.2f} to {dev[-1]:.2f})")


def to_unit(raw):
    if method.METHOD == "clip_aesthetic":
        return raw / 0.85
    if DEV_SCORES is None:
        return raw
    return bisect.bisect_left(DEV_SCORES, raw) / len(DEV_SCORES)


def scored(codes, renders, tag):
    """Pigment Gym's gate, copy check, length and partial credit, with the quality term from the chosen method."""
    rows = score_rollouts(codes, renders, clip_scorer, SUBJECT, shaped=True, forbid_copy_of=EXAMPLE_CODE)
    ids = {}
    for i, row in enumerate(rows):
        if row["gate"]:
            ids[i] = f"train-{args.name}-{tag}-{i}"
            prepare.png_path(ids[i]).write_bytes(renders[i].png)
    quality = {k: to_unit(v) for k, v in method.score(list(ids.values())).items()} if ids else {}
    for i, row in enumerate(rows):
        row["quality"] = quality.get(ids.get(i), 0.0)
        row["reward"] = (
            WEIGHTS["gate"] * row["gate"]
            + WEIGHTS["length"] * length_score(codes[i])
            + row["gate"] * (WEIGHTS["clip"] + WEIGHTS["aesthetic"]) * row["quality"]
            + row["shaping"]
        )
    return rows, ids


RUBRIC = (
    f"How good is this image as a watercolour painting of {SUBJECT}? Rate it from 1 to 5:\n"
    "1 = no recognisable picture\n2 = a wash of colour and a disc, nothing more\n3 = sunset colours and a sun\n"
    "4 = a sun and a green hill shape\n5 = a convincing sunset over green hills\n"
    'Reply with JSON only: {"rating": <1-5>}'
)


def judge_rating(judge, sketch_id, api_key):
    """Pigment Gym's 1-5 rubric, one image per request: the yardstick none of the rewards uses."""
    content = [
        {"type": "image_url", "image_url": {"url": prepare.data_url(sketch_id)}},
        {"type": "text", "text": RUBRIC},
    ]
    body = {
        "model": prepare.LABEL_JUDGES[judge],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"effort": "low", "exclude": True},
        "usage": {"include": True},
    }
    for _ in range(3):
        try:
            response = requests.post(
                prepare.URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=120
            )
            data = response.json()
            prepare.charge(data.get("usage", {}).get("cost", 0.0) or 0.0, f"e1 yardstick {judge}")
            text = data["choices"][0]["message"]["content"]
            rating = int(json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))["rating"])
            if 1 <= rating <= 5:
                return rating
        except (requests.RequestException, KeyError, ValueError, AttributeError, TypeError):
            time.sleep(3)
    return None


def evaluate(renderer, label):
    """16 fixed-seed sketches: gate pass, reward, copying, and the two judges' mean rating (a failure counts as 1)."""
    texts = []
    for batch in range(EVAL_SAMPLES // 8):
        batch_texts, _, _ = sample_group(
            model, tokenizer, prompt, 8, MAX_NEW_TOKENS, temperature=0.8, top_p=0.95, seed=10_000 + batch
        )
        texts += batch_texts
    codes = [extract_code(t) for t in texts]
    rows, ids = scored(codes, renderer.render_many(codes), label)
    api_key = prepare.secret("OPENROUTER_API_KEY")
    jobs = [(judge, sketch_id) for judge in prepare.LABEL_JUDGES for sketch_id in ids.values()]
    with ThreadPoolExecutor(8) as pool:
        ratings = list(pool.map(lambda job: judge_rating(*job, api_key), jobs))
    result = {
        "gate_pass": len(ids) / len(rows),
        "reward": mean(r["reward"] for r in rows),
        "similarity_to_example": mean(similarity(EXAMPLE_CODE, c) for c in codes),
        "distinct": len(set(codes)),
    }
    for judge in prepare.LABEL_JUDGES:
        got = [r for (j, _), r in zip(jobs, ratings, strict=True) if j == judge and r is not None]
        # A failing sketch counts as 1 ("no recognisable picture"); a passing sketch the judge could not rate (API
        # error) makes the whole mean unknown rather than silently scoring it 1.
        missing = len(ids) - len(got)
        result[f"judge_{judge}"] = None if missing else (sum(got) + (EVAL_SAMPLES - len(ids))) / EVAL_SAMPLES
        result[f"judge_{judge}_rated"] = f"{len(got)}/{len(ids)}"
    print(f"[{label}] " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}))
    with (OUT / f"{label}_sketches.jsonl").open("w") as handle:
        for code, row in zip(codes, rows, strict=True):
            handle.write(json.dumps({"code": code, **{k: v for k, v in row.items() if k != "violations"}}) + "\n")
    return result


device = pick_device()
model, tokenizer = build_lora_model(MODEL, device, seed=SEED)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Paint {SUBJECT}."}]
prompt = chat_prompt(tokenizer, messages)
print(f"{args.name}: method {method.METHOD}, expected={getattr(method, 'EXPECTED', None)}, file {path.name}")

history, started = [], time.time()
with Renderer(pool_size=4) as renderer:
    before = evaluate(renderer, "before")
    for step in range(args.steps):
        t0 = time.time()
        texts, input_ids, mask = sample_group(
            model, tokenizer, prompt, GROUP_SIZE, MAX_NEW_TOKENS, temperature=1.0, top_p=0.95, seed=SEED * 1000 + step
        )
        codes = [extract_code(t) for t in texts]
        rows, _ = scored(codes, renderer.render_many(codes), f"s{step}")
        rewards = [r["reward"] for r in rows]
        advantages = group_advantages(rewards, scale_rewards="none")
        metrics = grpo_update(model, optimizer, input_ids, mask, advantages, beta=BETA, micro_batch=2)
        passing_quality = [r["quality"] for r in rows if r["gate"]]
        row = {
            "step": step,
            "reward": mean(rewards),
            "gate_pass": mean(r["gate"] for r in rows),
            "zero_spread": float(max(rewards) - min(rewards) < 1e-6),
            "quality_ties": float(len(passing_quality) > 1 and max(passing_quality) - min(passing_quality) < 1e-6),
            "similarity": mean(similarity(EXAMPLE_CODE, c) for c in codes),
            "kl": metrics["kl"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
    after = evaluate(renderer, "after")

(OUT / "curve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
summary = {
    "name": args.name,
    "method": method.METHOD,
    "file": path.name,
    "discrete": args.discrete,
    "steps": args.steps,
    "minutes": (time.time() - started) / 60,
    "before": before,
    "after": after,
    "zero_spread_share": mean(r["zero_spread"] for r in history),
    "quality_tie_share": mean(r["quality_ties"] for r in history),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
