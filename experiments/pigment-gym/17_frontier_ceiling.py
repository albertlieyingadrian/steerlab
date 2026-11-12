"""
A ceiling for the follow-up study (added after the pre-registered arms were judged): Claude Opus 5.5 paints each
evaluation subject 8 times with exactly the arms' prompt, and its sketches are scored and judged like an arm.

Run from the repo root (needs ANTHROPIC_API_KEY in .env; about $2), then rerun scripts 13 and 14:
    .venv/bin/python experiments/pigment-gym/17_frontier_ceiling.py

Writes results/next/samples_opus.jsonl in modal_next.py's row format (group = request index, slot 0 only), so
13_judge_next.py rates it as the reference arm `opus`. It shows how far the small models are from what the task
allows on the same subjects, under the same gate, reward and judges.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
from brush_tools import PROMPT_EXAMPLE, SYSTEM_PROMPT, Renderer, extract_code
from common import RESULTS, check, outputs_dir
from next_common import EVAL_SUBJECTS, load_keys, png_key
from reward_tools import Scorer, score_rollouts, similarity

OUT = RESULTS / "next"
PER_SUBJECT = 8
PRICE = (4.00, 20.00)
cache_path = OUT / "ceiling_replies.jsonl"
done = {}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        record = json.loads(line)
        done[(record["subject_index"], record["index"])] = record

key = load_keys().get("ANTHROPIC_API_KEY")
if not key:
    sys.exit("ANTHROPIC_API_KEY not found in .env")
client = anthropic.Anthropic(api_key=key, max_retries=4)


def ask(job):
    subject_index, index = job
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"Paint {EVAL_SUBJECTS[subject_index]}."}],
    )
    usage = response.usage
    return {
        "subject_index": subject_index,
        "index": index,
        "text": "".join(b.text for b in response.content if b.type == "text"),
        "cost_usd": (usage.input_tokens * PRICE[0] + usage.output_tokens * PRICE[1]) / 1e6,
    }


todo = [(s, i) for s in range(len(EVAL_SUBJECTS)) for i in range(PER_SUBJECT) if (s, i) not in done]
with ThreadPoolExecutor(6) as pool, cache_path.open("a") as handle:
    for record in pool.map(ask, todo):
        done[(record["subject_index"], record["index"])] = record
        handle.write(json.dumps(record) + "\n")
spent = sum(r["cost_usd"] for r in done.values())
print(f"{len(done)} replies, ${spent:.2f}")

example = extract_code(PROMPT_EXAMPLE)
png_dir = outputs_dir() / "next" / "png"
png_dir.mkdir(parents=True, exist_ok=True)
scorer = Scorer()
rows = []
with Renderer(pool_size=8) as renderer:
    for subject_index, subject in enumerate(EVAL_SUBJECTS):
        records = [done[(subject_index, i)] for i in range(PER_SUBJECT)]
        codes = [extract_code(r["text"]) for r in records]
        renders = renderer.render_many(codes)
        scored = score_rollouts(codes, renders, scorer, subject, forbid_copy_of=example)
        for index, (code, render, score) in enumerate(zip(codes, renders, scored, strict=True)):
            name = png_key(render.png) if render.png is not None else None
            if name:
                (png_dir / f"{name}.png").write_bytes(render.png)
            rows.append(
                {
                    "arm_model": "opus",
                    "subject": subject,
                    "subject_index": subject_index,
                    "group": index,
                    "slot": 0,
                    "is_best": False,
                    "tokens": None,
                    "similarity": similarity(example, code),
                    "png": name,
                    **score,
                    "code": code,
                }
            )
with (OUT / "samples_opus.jsonl").open("w") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")
print(f"opus: {len(rows)} sketches, {sum(r['gate'] for r in rows):.0f} pass the gate")

print("\nSelf-checks:")
check("8 sketches per evaluation subject", len(rows) == PER_SUBJECT * len(EVAL_SUBJECTS))
check("under $3", spent < 3)
