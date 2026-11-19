"""Sample GPT, Claude and Gemini through OpenRouter: 50 answers per question and condition, max 32 tokens.

    .venv/bin/python sample_api.py --set dev            # data/dev_questions.json, 3 conditions
    .venv/bin/python sample_api.py --set main           # data/questions.json, 9 conditions

Each call asks for one reply (OpenRouter does not return several samples per call for these models). Temperature is
1.0 where the model accepts it (Gemini); GPT-6 Luna and Claude Haiku 5.5 do not accept a temperature and sample at
their fixed default of 1. Reasoning is switched off (effort "none"; every call logged 0 reasoning tokens in the
pilot). Spend is capped at plural.API_CAP_USD, counted from OpenRouter's own per-call cost.

Writes results/samples/<set>/<model>.jsonl.gz (same format as modal_sample.py). Resumable: finished cells are kept
in results/samples/<set>/<model>.partial.jsonl until the model is complete.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor

from dotenv import dotenv_values

import plural

MODELS = {  # short name -> (OpenRouter id, extra body)
    "gpt-6-luna": ("openai/gpt-6-luna", {"reasoning": {"effort": "none"}}),
    "claude-haiku-5.5": ("anthropic/claude-haiku-5.5", {"reasoning": {"effort": "none"}}),
    "gemini-3.1-flash-lite": ("google/gemini-3.1-flash-lite", {"temperature": 1.0, "reasoning": {"effort": "none"}}),
}


def find_env() -> dict:
    for p in [plural.HERE, *plural.HERE.parents]:
        if (p / ".env").exists():
            return dotenv_values(p / ".env")
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n", type=int, default=plural.N_SAMPLES)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    env = find_env()
    key = env.get("OPENROUTER_API_KEY_2") or env["OPENROUTER_API_KEY"]
    qs = plural.load_questions(plural.HERE / "data" / ("questions.json" if a.set == "main" else "dev_questions.json"))
    conds = plural.CONDITIONS if a.set == "main" else ["default", "Indonesia", "United States"]
    cells = [(q, c) for q in qs for c in conds]
    out_dir = plural.HERE / "results" / "samples" / a.set
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.models.split(","):
        model, extra = MODELS[name]
        final = out_dir / f"{name}.jsonl.gz"
        if final.exists():
            print(f"{name}: done already")
            continue
        partial = out_dir / f"{name}.partial.jsonl"
        done = {}
        if partial.exists():
            for line in partial.read_text().splitlines():
                r = json.loads(line)
                done[(r["id"], r["condition"])] = r["replies"]
        todo = [(q, c) for q, c in cells if (q["id"], c) not in done]
        print(f"{name}: {len(todo)} cells to sample ({len(done)} done), spent ${plural.spent():.2f}", flush=True)

        def cell(qc, model=model, extra=extra):
            q, c = qc
            text = plural.prompt_text(q, c)
            return qc, [plural.openrouter(model, text, key, max_tokens=32, extra=extra) for _ in range(a.n)]

        with ThreadPoolExecutor(a.workers) as ex, partial.open("a") as f:
            for i, ((q, c), replies) in enumerate(ex.map(cell, todo)):
                done[(q["id"], c)] = replies
                f.write(json.dumps({"id": q["id"], "condition": c, "replies": replies}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 100 == 0:
                    print(f"  {name} {i}/{len(todo)} spent ${plural.spent():.2f}", flush=True)
        with gzip.open(final, "wt") as f:
            for q, c in cells:
                f.write(
                    json.dumps({"id": q["id"], "condition": c, "replies": done[(q["id"], c)]}, ensure_ascii=False)
                    + "\n"
                )
        partial.unlink()
        reps = [(r, len(q["options"])) for q, c in cells for r in done[(q["id"], c)]]
        valid = sum(plural.parse(r, k) >= 0 for r, k in reps) / len(reps)
        print(f"{name}: valid {valid:.3f}, spent ${plural.spent():.2f}", flush=True)


if __name__ == "__main__":
    main()
