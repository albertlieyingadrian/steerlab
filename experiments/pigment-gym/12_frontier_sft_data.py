"""
Fine-tuning data for the `sft` arm: Claude Opus 5.5's sketches of 24 training subjects, kept only if they pass.

Run from the repo root (needs ANTHROPIC_API_KEY in .env; about $5-10, capped by BUDGET_USD):
    .venv/bin/python experiments/pigment-gym/12_frontier_sft_data.py --per-subject 1 --subjects 2   # smoke test
    .venv/bin/python experiments/pigment-gym/12_frontier_sft_data.py

Same system prompt as every arm (rules + the worked example) plus one line asking for a short sketch, because the
0.5B student samples at most 1,024 tokens. Every reply is cached in results/next/teacher.jsonl (a rerun only asks
for what is missing). Kept for training (results/next/sft_train.jsonl): passes the gate, under 85% similar to the
prompt's example, at most 1,000 Qwen tokens.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
from brush_tools import PROMPT_EXAMPLE, SYSTEM_PROMPT, Renderer, extract_code, gate
from common import RESULTS, check
from next_common import MODEL, TRAIN_SUBJECTS, load_keys
from reward_tools import MAX_SIMILARITY_TO_EXAMPLE, similarity

OUT = RESULTS / "next"
TEACHER = "claude-opus-5-5"
PRICE = (4.00, 20.00)  # USD per million input, output tokens (Anthropic list price, 2026-11)
BUDGET_USD = 12.0
MAX_TOKENS_QWEN = 1000
TEACHER_SYSTEM = SYSTEM_PROMPT + "\nKeep the sketch short: about 30 to 50 lines."

parser = argparse.ArgumentParser()
parser.add_argument("--per-subject", type=int, default=8)
parser.add_argument("--subjects", type=int, default=len(TRAIN_SUBJECTS))
args = parser.parse_args()

OUT.mkdir(parents=True, exist_ok=True)
cache_path = OUT / "teacher.jsonl"
done = {}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        record = json.loads(line)
        done[(record["subject"], record["index"])] = record

key = load_keys().get("ANTHROPIC_API_KEY")
if not key:
    sys.exit("ANTHROPIC_API_KEY not found in .env")
client = anthropic.Anthropic(api_key=key, max_retries=4)
subjects = TRAIN_SUBJECTS[: args.subjects]
todo = [(s, i) for s in subjects for i in range(args.per_subject) if (s, i) not in done]
spent = sum(r["cost_usd"] for r in done.values())
print(f"{len(todo)} requests to send, {len(done)} cached; spent so far ${spent:.2f} of ${BUDGET_USD:.0f}")


def ask(job):
    subject, index = job
    if spent >= BUDGET_USD:
        return None
    try:
        response = client.messages.create(
            model=TEACHER,
            max_tokens=8000,
            system=TEACHER_SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": f"Paint {subject}."}],
        )
    except anthropic.APIError as error:
        print(f"  {subject} #{index}: {type(error).__name__}")
        return None
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = response.usage
    cost = (usage.input_tokens * PRICE[0] + usage.output_tokens * PRICE[1]) / 1e6
    return {
        "subject": subject,
        "index": index,
        "text": text,
        "stop_reason": response.stop_reason,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost,
    }


with ThreadPoolExecutor(6) as pool, cache_path.open("a") as handle:
    for record in pool.map(ask, todo):
        if record is None:
            continue
        spent += record["cost_usd"]
        done[(record["subject"], record["index"])] = record
        handle.write(json.dumps(record) + "\n")
        handle.flush()
print(f"teacher spend in total: ${spent:.2f}")

# Filter: render every reply, keep the ones that pass and are short and not copies.
from transformers import AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(MODEL)
example = extract_code(PROMPT_EXAMPLE)
records = [done[k] for k in sorted(done) if k[0] in subjects]
codes = [extract_code(r["text"]) for r in records]
with Renderer(pool_size=6) as renderer:
    renders = renderer.render_many(codes)
kept, reasons = [], {"gate": 0, "copy": 0, "long": 0}
for record, code, render in zip(records, codes, renders, strict=True):
    reply = f"```javascript\n{code}\n```"
    if not gate(code, render).passed:
        reasons["gate"] += 1
    elif similarity(example, code) >= MAX_SIMILARITY_TO_EXAMPLE:
        reasons["copy"] += 1
    elif len(tokenizer(reply)["input_ids"]) > MAX_TOKENS_QWEN:
        reasons["long"] += 1
    else:
        kept.append({"subject": record["subject"], "index": record["index"], "reply": reply})
with (OUT / "sft_train.jsonl").open("w") as handle:
    for row in kept:
        handle.write(json.dumps(row) + "\n")
print(f"kept {len(kept)} of {len(records)} sketches for fine-tuning; dropped {reasons}")
print(f"subjects with at least one kept sketch: {len({r['subject'] for r in kept})} of {len(subjects)}")

print("\nSelf-checks:")
check("stayed within the budget", spent <= BUDGET_USD + 1)
check("at least half the replies kept", len(kept) >= len(records) / 2)
