"""
The frontier baseline. Ask big hosted models the same question as script 2, through OpenRouter.

Run from the repo root (needs OPENROUTER_API_KEY in .env or .env.local; costs a few dollars, capped by BUDGET_USD):
    .venv/bin/python experiments/pigment-gym/2_sample_frontier.py
    .venv/bin/python experiments/pigment-gym/2_sample_frontier.py --only opus-5.5
Then score everything (small and frontier models together) with script 3.

Why: the small Qwen models pass the gate 0-44% of the time. Without a frontier row we can't say how far that is
from what the task allows. Same system prompts, same user prompt, same 16 samples per (model, prompt), same
temperature 0.8, appended to the same JSONL file, so script 3 scores them identically.
Two differences, on purpose: max_tokens is 6,000 (not 1,024) because reasoning models spend tokens thinking before
they answer, and reasoning effort is set to "low". Each record keeps the model OpenRouter actually served, the
finish reason (a "length" finish means the reply was cut off) and the cost.

OpenRouter is one API in front of many providers (like a payment gateway in front of many banks): one key,
one request format, the model chosen by name.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[1]
from brush_tools import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_EXAMPLE  # noqa: E402
from common import RESULTS, check  # noqa: E402
from hand_pool import SUBJECT  # noqa: E402

MODELS = {
    "opus-5.5": "anthropic/claude-opus-5.5",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gemini-pro": "~google/gemini-pro-latest",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
}
PROMPTS = {"no_example": SYSTEM_PROMPT_NO_EXAMPLE, "example": SYSTEM_PROMPT}
SAMPLES_PER_MODEL = 16
MAX_TOKENS = 6000
BUDGET_USD = 15.0
WORKERS = 8
SAMPLES_PATH = RESULTS / "benchmark_samples.jsonl"
URL = "https://openrouter.ai/api/v1/chat/completions"

keys = {**dotenv_values(REPO / ".env"), **dotenv_values(REPO / ".env.local")}
api_key = keys.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("OPENROUTER_API_KEY not found in .env or .env.local")

only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
done = set()
if SAMPLES_PATH.exists():
    for line in SAMPLES_PATH.read_text().splitlines():
        record = json.loads(line)
        done.add((record["model"], record.get("prompt", "no_example"), record["index"]))

user = f"Paint {SUBJECT}."
todo = [
    (name, p, i)
    for name in MODELS
    for p in PROMPTS
    for i in range(SAMPLES_PER_MODEL)
    if (not only or name == only) and (name, p, i) not in done
]
print(f"{len(todo)} requests to send ({len(done)} samples already in {SAMPLES_PATH.name}); budget ${BUDGET_USD:.2f}\n")

spent = 0.0


def ask(job):
    """One request -> one JSONL record, or None if the budget is used up or the request keeps failing."""
    name, prompt_name, index = job
    if spent >= BUDGET_USD:
        return None
    body = {
        "model": MODELS[name],
        "messages": [{"role": "system", "content": PROMPTS[prompt_name]}, {"role": "user", "content": user}],
        "temperature": 0.8,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"effort": "low", "exclude": True},
        "usage": {"include": True},
        "seed": index,
    }
    for attempt in range(3):
        started = time.perf_counter()
        try:
            response = requests.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=300)
            data = response.json()
        except (requests.RequestException, ValueError) as error:
            data = {"error": {"message": str(error)}}
        if "choices" in data:
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return {
                "model": name,
                "prompt": prompt_name,
                "index": index,
                "text": choice["message"].get("content") or "",
                "tokens": usage.get("completion_tokens", 0),
                "batch_seconds": round(time.perf_counter() - started, 1),
                "served_model": data.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "cost_usd": usage.get("cost", 0.0),
            }
        print(
            f"  {name} {prompt_name} #{index}: attempt {attempt + 1} failed: {data.get('error', {}).get('message', '?')[:120]}"  # noqa: E501
        )
        time.sleep(5 * (attempt + 1))
    return None


with ThreadPoolExecutor(WORKERS) as pool, SAMPLES_PATH.open("a") as handle:
    for record in pool.map(ask, todo):
        if record is None:
            continue
        spent += record["cost_usd"] or 0.0
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        print(
            f"  {record['model']:12} {record['prompt']:10} #{record['index']:2d}  {record['tokens']:5d} tokens  "
            f"{record['finish_reason']}  ${record['cost_usd'] or 0:.3f}  (total ${spent:.2f})"
        )

records = [json.loads(line) for line in SAMPLES_PATH.read_text().splitlines()]
frontier = [r for r in records if r["model"] in MODELS]
print(f"\nspent this run: ${spent:.2f}; frontier samples on file: {len(frontier)}")
for name in MODELS:
    mine = [r for r in frontier if r["model"] == name]
    if mine:
        served = sorted({r.get("served_model") for r in mine})
        cut = sum(r.get("finish_reason") == "length" for r in mine)
        print(
            f"  {name:12} {len(mine):3d} samples, served as {served}, {cut} cut off at max_tokens, "
            f"${sum(r.get('cost_usd') or 0 for r in mine):.2f}"
        )

print("\nSelf-checks:")
for name in MODELS:
    if only and name != only:
        continue
    for prompt_name in PROMPTS:
        count = sum(1 for r in frontier if r["model"] == name and r["prompt"] == prompt_name)
        check(f"{name} / {prompt_name}: {SAMPLES_PER_MODEL} samples saved", count >= SAMPLES_PER_MODEL)
check(f"stayed within the ${BUDGET_USD:.0f} budget", spent <= BUDGET_USD + 1.0)
