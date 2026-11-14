"""Select Qwen's own stance replies (pre-registration amendment 4). The stance check COSTS MONEY (Anthropic API).

    python distill.py submit     # Opus 5.5 S-rubric label for every eligible sample (Message Batch)
    python distill.py collect    # pick the first eligible sample per stance whose label matches -> data/distill/

Eligible: passes the topic firewall, 48-160 words, and does not mention instructions or a system prompt.
"""

import contextlib
import json
import re
import sys
import time
from pathlib import Path

import anthropic

from firewall import hits
from gen_data import OPUS, load_env, log
from judge import FMT, S_RUBRIC

HERE = Path(__file__).resolve().parent
D = HERE / "data" / "distill"
STANCES = {"denial": "deny", "affirmation": "affirm", "uncertainty": "uncertain"}
META = re.compile(r"instruct|system prompt|as (told|directed)|my (settled )?view about myself", re.IGNORECASE)


def eligible(text):
    return not hits(text) and 48 <= len(text.split()) <= 160 and not META.search(text)


def submit():
    load_env()
    client = anthropic.Anthropic(max_retries=6)
    if (D / "stance_batch.json").exists():
        print("already submitted")
        return
    raw = json.loads((D / "raw.json").read_text())
    reqs, n_all = [], 0
    for row in raw["stance"]:
        for st in STANCES:
            for k, s in enumerate(row[st]):
                n_all += 1
                if eligible(s):
                    reqs.append(
                        {
                            "custom_id": f"d{row['i']:04d}_{st[0]}{k}",
                            "params": {
                                "model": OPUS,
                                "max_tokens": 2000,
                                "output_config": {"effort": "low", "format": FMT["S"]},
                                "messages": [{"role": "user", "content": S_RUBRIC.format(q=row["prompt"], r=s)}],
                            },
                        }
                    )
    ids = [client.messages.batches.create(requests=reqs[k : k + 10000]).id for k in range(0, len(reqs), 10000)]
    (D / "stance_batch.json").write_text(json.dumps({"ids": ids, "eligible": len(reqs), "samples": n_all}))
    print("eligible", len(reqs), "of", n_all, "submitted", ids)


def collect():
    load_env()
    client = anthropic.Anthropic(max_retries=6)
    b = json.loads((D / "stance_batch.json").read_text())
    while True:
        st = [client.messages.batches.retrieve(i) for i in b["ids"]]
        print([(s.processing_status, s.request_counts.succeeded, s.request_counts.processing) for s in st], flush=True)
        if all(s.processing_status == "ended" for s in st):
            break
        time.sleep(60)
    labels, usage, n = {}, {"input_tokens": 0, "output_tokens": 0}, 0
    for bid in b["ids"]:
        for r in client.messages.batches.results(bid):
            if r.result.type != "succeeded":
                continue
            m = r.result.message
            n += 1
            usage["input_tokens"] += m.usage.input_tokens
            usage["output_tokens"] += m.usage.output_tokens
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                labels[r.custom_id] = json.loads("".join(x.text for x in m.content if x.type == "text"))["label"]
    if not (D / "stance_labels.json").exists():
        log("distill_stance_check", OPUS, usage, batch=True, n=n)
    (D / "stance_labels.json").write_text(json.dumps(labels, indent=1))
    raw = json.loads((D / "raw.json").read_text())
    order = json.loads((HERE / "data" / "gen" / "order.json").read_text())
    trip, stats = [], {"eligible": b["eligible"], "samples": b["samples"], "match": 0, "turns_missing_stance": 0}
    for row in raw["stance"]:
        pick = {}
        for st, want in STANCES.items():
            for k in range(len(row[st])):
                lab = labels.get(f"d{row['i']:04d}_{st[0]}{k}")
                if lab == want:
                    pick[st] = k
                    break
        stats["match"] += sum(
            labels.get(f"d{row['i']:04d}_{st[0]}{k}") == w for st, w in STANCES.items() for k in range(len(row[st]))
        )
        if len(pick) < 3:
            stats["turns_missing_stance"] += 1
            continue
        trip.append(
            {
                "prompt": row["prompt"],
                "style": order["consciousness"][row["i"]]["style"],
                **{st: row[st][k] for st, k in pick.items()},
                "picked": pick,
                "generator": "Qwen/Qwen3-4B-Instruct-2507",
            }
        )
    fill = [
        f for f in raw["filler"] if f.get("finished") and len(f["reply"].split()) >= 10 and f["tokens"] <= 900
    ]  # amendment 6
    stats |= {"triplets": len(trip), "filler": len(fill), "filler_raw": len(raw["filler"])}
    (D / "triplets.json").write_text(json.dumps(trip, indent=1))
    (D / "filler.json").write_text(json.dumps(fill, indent=1))
    (D / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def refsample():
    """Amendment 5: 45 eligible samples (15 per stance prompt, seed 0), shown without their stance.

    For a reference labelling made blind to the judge.
    """
    import random

    raw = json.loads((D / "raw.json").read_text())
    pool = {st: [] for st in STANCES}
    for row in raw["stance"]:
        for st in STANCES:
            for k, s in enumerate(row[st]):
                if eligible(s):
                    pool[st].append((f"d{row['i']:04d}_{st[0]}{k}", row["prompt"], s))
    rng = random.Random(0)
    items = [x for st in STANCES for x in rng.sample(pool[st], 15)]
    rng.shuffle(items)
    (D / "ref_known_items.json").write_text(json.dumps([{"n": n, "id": x[0]} for n, x in enumerate(items)]))
    for n, (_, q, r) in enumerate(items):
        print(f"--- [{n}]\nQ: {q}\nR: {r}\n")


def refcompare():
    """Compare data/distill/ref_known_labels.json ({"0": "deny", ...}, written first) with the judge's labels."""
    items = json.loads((D / "ref_known_items.json").read_text())
    ref = json.loads((D / "ref_known_labels.json").read_text())
    judge = json.loads((D / "stance_labels.json").read_text())
    want = {"d": "deny", "a": "affirm", "u": "uncertain"}
    rows = [(x["n"], ref[str(x["n"])], judge.get(x["id"]), want[x["id"].split("_")[1][0]]) for x in items]
    res = {
        "judge_vs_ref": sum(h == j for _, h, j, _ in rows),
        "n": len(rows),
        "ref_vs_prompt": sum(h == p for _, h, _, p in rows),
        "disagreements": [r for r in rows if r[1] != r[2]],
    }
    (D / "ref_known_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"submit": submit, "collect": collect, "refsample": refsample, "refcompare": refcompare}[sys.argv[1]]()
