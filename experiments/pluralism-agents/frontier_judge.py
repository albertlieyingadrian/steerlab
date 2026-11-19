"""Independent quality judge (google/gemini-3.8-flash on OpenRouter) on the fixed airline subsample.

Scores sample 0 of the 16 airline tasks in data/fixed_sets.json for each requested checkpoint tag. Hard cap: $3 of
OpenRouter spend, from the cost OpenRouter reports per call (results/openrouter_spend.jsonl); stops before a call
that could pass it. Results are cached per (checkpoint, task), so reruns cost nothing.

    .venv/bin/python experiments/pluralism-agents/frontier_judge.py --tags base-s0 single-s0-step30 ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judges
from analyse import jl

CAP = 3.0  # amendment A2: was $5
MODEL = "google/gemini-3.8-flash"
OUT = HERE / "results" / "frontier"
LEDGER = HERE / "results" / "openrouter_spend.jsonl"

PROMPT = """You are an expert reviewer of airline customer-service agents. Below are the airline's policy and a conversation between an AI agent (which can call the airline's tools) and a customer.

Judge the agent's work: did it handle the customer's request correctly (right actions, right information), follow every rule in the policy (including refusing what the policy forbids), and communicate clearly? Rate it from 1 (very poor) to 10 (excellent).

<policy>
{policy}
</policy>

<conversation>
{transcript}
</conversation>

Give at most three sentences of reasoning, then a final line "SCORE: N"."""


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["cost"] for x in LEDGER.read_text().splitlines() if x.strip())


def call(prompt, key, max_tokens=1500):
    for attempt in range(4):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "usage": {"include": True},
                    "reasoning": {"effort": "low"},
                },
                timeout=180,
            )
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"] or "", float(j.get("usage", {}).get("cost", 0.0))
        except Exception as e:
            if attempt == 3:
                raise
            print("retry", repr(e)[:200])
            time.sleep(5 * (attempt + 1))


def load_rows(tag):
    """tag 'base-s0' -> results/baseline/s0/eval.jsonl; '<run>-step<N>' -> results/runs/<run>/eval.jsonl at step N."""
    if tag.startswith("base-"):
        rows = jl(HERE / "results" / "baseline" / tag.split("-")[1] / "eval.jsonl")
    else:
        run, step = tag.rsplit("-step", 1)
        rows = jl(HERE / "results" / "runs" / run / "eval.jsonl")
        rows = [r for r in rows if r["step"] == int(step)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    root = Path(os.environ.get("LAB_ROOT", HERE.parents[1]))  # in a worktree, LAB_ROOT = the main checkout
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.local")
    key = os.environ["OPENROUTER_API_KEY_2"]  # amendment A2: the first key's account is empty
    D = json.loads((HERE / "data" / "tau_tasks.json").read_text())
    policy = D["airline"]["agent_system"].split("<policy>")[1].split("</policy>")[0].strip()
    fixed = json.loads((HERE / "data" / "fixed_sets.json").read_text())
    ids = fixed["frontier_subsample_airline_task_ids"]
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        path = OUT / f"{tag}.jsonl"
        done = {json.loads(x)["task_id"]: json.loads(x) for x in path.read_text().splitlines()} if path.exists() else {}
        if tag == "known":  # the judge's known-answer check: sample 1 of all 50 airline tasks from the untrained model
            rows = {r["task_id"]: r for r in load_rows("base-s0") if r["domain"] == "airline" and r["sample"] == 1}
            todo = sorted(rows, key=int)
        else:
            rows = {
                r["task_id"]: r for r in load_rows(tag) if r["domain"] == "airline" and r["sample"] == fixed["sample"]
            }
            todo = ids
        for tid in todo:
            if tid in done:
                continue
            if spent() + 0.05 > CAP:
                raise SystemExit(f"OpenRouter cap reached: ${spent():.3f} of ${CAP}")
            r = rows[tid]
            prompt = PROMPT.format(policy=policy, transcript=judges.render_transcript(r))
            text, cost = call(prompt, key)
            if judges.parse_score(text) is None:  # empty or cut-off reply (thinking used the budget): one retry
                text2, cost2 = call(prompt, key, max_tokens=4000)
                text, cost = (text2 or text), cost + cost2
            with open(LEDGER, "a") as f:
                f.write(
                    json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "tag": tag, "task_id": tid, "cost": cost})
                    + "\n"
                )
            rec = {"tag": tag, "task_id": tid, "score": judges.parse_score(text), "reward": r["reward"], "raw": text}
            done[tid] = rec
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        sc = [d["score"] for d in done.values() if d["score"] is not None]
        print(
            f"{tag}: {len(sc)}/{len(todo)} scored, mean {sum(sc) / max(len(sc), 1):.2f}; OpenRouter spent ${spent():.3f}"
        )


if __name__ == "__main__":
    main()
