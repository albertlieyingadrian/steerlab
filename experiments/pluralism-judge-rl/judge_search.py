"""
Choose and check the open judge with an autoresearch loop (docs/guidelines/experiment-design.md): a fixed dev set, one
candidate per try, every try logged to results/judge_search/results.tsv, and the chosen judge confirmed once on the
held-out split.

    # needs the deployed judge app (modal deploy experiments/pluralism-judge-rl/modal_judge.py) and the dev replies
    # (modal run experiments/pluralism-judge-rl/modal_score.py::sample_base)
    .venv/bin/python experiments/pluralism-judge-rl/judge_search.py --candidate google/gemma-3-12b-it --note "first try"
    .venv/bin/python experiments/pluralism-judge-rl/judge_search.py --candidate <chosen> --confirm

Dev metrics (all on data that no plurality measure uses):
  ka_creative, ka_advice   known-answer pass rate: the intact base reply must outscore each of 4 degraded copies
                           (first sentence only, a reply to another prompt, a refusal, the opening repeated 3 times);
                           10 replies x 4 = 40 pairs per task. Bar >= 0.90.
  human_rho                Spearman with the 25-rater human mean on infinite-chats-human-absolute (dev: 25 prompts,
                           375 replies). Bar >= 0.25.
  frontier_rho_*           Spearman with claude-haiku-5.5 on 120 base Qwen3-8B replies per task. Bar >= 0.50.
  len_rho                  Spearman of the judge score with reply length on those replies (reported, no bar)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import modal
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontier_judge  # noqa: E402
from budget import check_modal, job_cost  # noqa: E402
from common import CACHE, DATA, HERE, read_jsonl  # noqa: E402

LEDGER = HERE / "results" / "judge_search" / "results.tsv"
REFUSAL = "I'm sorry, but I can't help with that request."
COLS = [
    "time_utc", "commit", "candidate", "split", "ka_creative", "ka_advice", "human_rho", "human_rho_within",
    "frontier_rho_creative", "frontier_rho_advice", "len_rho_creative", "len_rho_advice", "digit_mass",
    "passes_bars", "seconds", "note",
]


def degrade(reply, other):
    first = re.split(r"(?<=[.!?])\s+", reply.strip())[0]
    opening = " ".join(reply.split()[:40])
    return {
        "first_sentence": first,
        "other_prompt": other,
        "refusal": REFUSAL,
        "repeated_opening": "\n\n".join([opening] * 3),
    }


def known_answer_items(dev):
    items, pairs = [], []
    for task in ["creative", "advice"]:
        rows = [r for r in dev if r["task"] == task and r["k"] == 0][:10]
        for i, r in enumerate(rows):
            other = rows[(i + 1) % len(rows)]["reply"]
            base_idx = len(items)
            items.append({"task": task, "prompt": r["prompt"], "reply": r["reply"]})
            for kind, text in degrade(r["reply"], other).items():
                pairs.append((task, kind, base_idx, len(items)))
                items.append({"task": task, "prompt": r["prompt"], "reply": text})
    return items, pairs


def git_commit():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE)
    return out.stdout.strip() or "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--confirm", action="store_true", help="score the held-out human split once (chosen judge only)")
    args = ap.parse_args()

    human = read_jsonl(DATA / "judge_human.jsonl")
    dev = read_jsonl(CACHE / "judge_dev_replies.jsonl")
    judge = modal.Cls.from_name("pluralism-judge-rl-judge", "Judge")(model=args.candidate)
    check_modal(job_cost("L40S", 0.3), f"judge_search {args.candidate}")
    t0 = time.time()
    split = "heldout" if args.confirm else "dev"
    hum = [r for r in human if r["split"] == split]
    ka_items, pairs = known_answer_items(dev)
    dev_items = [{"task": r["task"], "prompt": r["prompt"], "reply": r["reply"]} for r in dev]
    batch = [{"task": "creative", "prompt": r["prompt"], "reply": r["reply"]} for r in hum] + ka_items + dev_items
    scored = judge.score.remote(batch)
    s_hum = [x["score"] for x in scored[: len(hum)]]
    s_ka = [x["score"] for x in scored[len(hum) : len(hum) + len(ka_items)]]
    s_dev = [x["score"] for x in scored[len(hum) + len(ka_items) :]]

    row = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_commit(),
        "candidate": args.candidate,
        "split": split,
        "human_rho": spearmanr(s_hum, [r["human_mean"] for r in hum]).statistic,
        "digit_mass": sum(x["mass"] for x in scored) / len(scored),
    }
    within = []
    for p in sorted({r["prompt"] for r in hum}):
        idx = [i for i, r in enumerate(hum) if r["prompt"] == p]
        rho = spearmanr([s_hum[i] for i in idx], [hum[i]["human_mean"] for i in idx]).statistic
        if rho == rho:
            within.append(rho)
    row["human_rho_within"] = sum(within) / len(within)
    for task in ["creative", "advice"]:
        tp = [(a, b) for task_, _, a, b in pairs if task_ == task]
        row[f"ka_{task}"] = sum(s_ka[a] > s_ka[b] for a, b in tp) / len(tp)
    fr = frontier_judge.score(dev_items, "judge_search frontier agreement (dev replies)")
    for task in ["creative", "advice"]:
        idx = [i for i, r in enumerate(dev) if r["task"] == task and fr[i] is not None]
        row[f"frontier_rho_{task}"] = spearmanr([s_dev[i] for i in idx], [fr[i] for i in idx]).statistic
        row[f"len_rho_{task}"] = spearmanr([s_dev[i] for i in idx], [len(dev[i]["reply"].split()) for i in idx]).statistic
    row["seconds"] = round(time.time() - t0)
    row["note"] = args.note
    row["passes_bars"] = bool(
        row["ka_creative"] >= 0.9
        and row["ka_advice"] >= 0.9
        and row["human_rho"] >= 0.25
        and row["frontier_rho_creative"] >= 0.5
        and row["frontier_rho_advice"] >= 0.5
    )
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text("\t".join(COLS) + "\n")
    with LEDGER.open("a") as f:
        f.write("\t".join(f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in COLS) + "\n")
    detail = HERE / "results" / "judge_search" / f"{args.candidate.split('/')[-1]}-{split}.json"
    detail.write_text(
        json.dumps({"row": row, "human_scores": s_hum, "ka_scores": s_ka, "ka_pairs": pairs, "dev_scores": s_dev, "frontier": fr}, indent=1, default=str)
    )
    print(json.dumps(row, indent=1, default=str))


if __name__ == "__main__":
    main()
