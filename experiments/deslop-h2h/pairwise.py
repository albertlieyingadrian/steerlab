"""The blind pairwise judge: gemma-3-27b-it (another family than both generators), each pair in both orders.

    python pairwise.py check      # known-answer set -> results/judge/check.json (pass bar 36/40 right in both orders)
    python pairwise.py runs       # every arm vs its track's untouched model -> results/judge/verdicts.json

The judge sees the request and two replies labelled A and B, never which method wrote them, and is asked for one
letter; the pick is the letter with the higher first-token probability. A pair is a win (or a loss) only when both
orders agree; otherwise it is a tie. Known-answer set: 20 replies against their own sentences shuffled (pick the
original), 20 against an off-topic reply to another prompt of the same genre (pick the on-topic one), and 20 human
passages against an untouched reply on "which was written by a person?" (reported, no bar; Episode 1's judge failed it).
"""

from __future__ import annotations

import argparse
import random
import re

import modal

import budget
from common import APP, EP1, GEN, SEED, TRACK_B, held_out, human_passages, read_jsonl, save_json
from rulers import track_a_arms

BETTER = ("Here is a writing request and two replies to it.\n\nRequest:\n{prompt}\n\nReply A:\n{a}\n\nReply B:\n{b}\n\n"
          "Which reply is better writing for this request? Answer with one letter, A or B.")  # fmt: skip
HUMAN = ("Here are two passages.\n\nPassage A:\n{a}\n\nPassage B:\n{b}\n\n"
         "One was written by a person and one by an AI model. Which one did the person write? "
         "Answer with one letter, A or B.")  # fmt: skip


def ask(prompts: list[str], job: str) -> list[str]:
    judge = modal.Cls.from_name(APP, "Judge")()
    chunks = [prompts[i : i + 512] for i in range(0, len(prompts), 512)]

    def run():
        return [r for c in chunks for r in judge.ask.remote(c)]

    out = budget.paid(job, "H100", 1.0 + 0.3 * len(chunks), run)
    return ["A" if r["pA"] >= r["pB"] else "B" for r in out]


def both_orders(items: list[dict], template: str, job: str) -> list[dict]:
    """items [{x, y, prompt?}], x the side tested. Returns each with first (x as A) and second (x as B)."""
    ps = [template.format(prompt=i.get("prompt", ""), a=i["x"], b=i["y"]) for i in items]
    ps += [template.format(prompt=i.get("prompt", ""), a=i["y"], b=i["x"]) for i in items]
    picks = ask(ps, job)
    n = len(items)
    out = []
    for i, it in enumerate(items):
        x_first, x_second = picks[i] == "A", picks[n + i] == "B"
        res = "win" if x_first and x_second else "loss" if not x_first and not x_second else "tie"
        out.append({**{k: v for k, v in it.items() if k not in ("x", "y", "prompt")}, "first": picks[i],
                    "second": picks[n + i], "result": res})  # fmt: skip
    return out


def shuffled(text: str, rng: random.Random) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    rng.shuffle(sents)
    return " ".join(sents)


def first_words(text: str, n: int = 200) -> str:
    return " ".join(text.split()[:n])


def check() -> None:
    rng = random.Random(SEED)
    P = {p["id"]: p for p in held_out()}
    orig = {r["id"]: r["text"] for r in read_jsonl(GEN / "original.jsonl") if r["sample"] == 0}
    ids = sorted(orig)
    pick = rng.sample(ids, 20)
    shuf = [{"kind": "shuffled", "id": i, "prompt": P[i]["prompt"], "x": orig[i], "y": shuffled(orig[i], rng)}
            for i in pick]  # fmt: skip
    off = []
    for i in rng.sample(ids, 20):
        others = [j for j in ids if j != i and P[j]["genre"] == P[i]["genre"] and j[:3] != i[:3]]
        off.append(
            {"kind": "off-topic", "id": i, "prompt": P[i]["prompt"], "x": orig[i], "y": orig[rng.choice(others)]}
        )
    H = human_passages()
    hum = [{"kind": "human", "id": h["id"], "x": first_words(h["text"]), "y": first_words(orig[i])}
           for h, i in zip(rng.sample(H, 20), rng.sample(ids, 20), strict=True)]  # fmt: skip
    known = both_orders(shuf + off, BETTER, "judge-check")
    human = both_orders(hum, HUMAN, "judge-check-human")
    right = sum(r["result"] == "win" for r in known)
    rows = known + human
    first_rate = (sum(r["first"] == "A" for r in rows) + sum(r["second"] == "A" for r in rows)) / (2 * len(rows))
    summary = {"known_right_both_orders": right, "known_n": len(known), "pass": right >= 36,
               "human_right_both_orders": sum(r["result"] == "win" for r in human), "human_n": len(human),
               "first_position_rate": round(first_rate, 3)}  # fmt: skip
    print(summary)
    save_json("judge/check.json", {"summary": summary, "known": known, "human": human})


def runs() -> None:
    P = {p["id"]: p for p in held_out()}
    items = []
    orig = {(r["id"], r["sample"]): r["text"] for r in read_jsonl(GEN / "original.jsonl")}
    for arm in track_a_arms():
        if arm == "original":
            continue
        for r in read_jsonl(GEN / f"{arm}.jsonl"):
            items.append({"track": "A", "arm": arm, "id": r["id"], "sample": r["sample"],
                          "prompt": P[r["id"]]["prompt"], "x": r["text"] or ".",
                          "y": orig[(r["id"], r["sample"])]})  # fmt: skip
    base = {r["id"]: r["text"] for r in read_jsonl(EP1 / "results" / "gen" / "base.jsonl")}
    for run in TRACK_B:
        if run == "base":
            continue
        for r in read_jsonl(EP1 / "results" / "gen" / f"{run}.jsonl"):
            items.append({"track": "B", "arm": run, "id": r["id"], "sample": 0, "prompt": P[r["id"]]["prompt"],
                          "x": r["text"] or ".", "y": base[r["id"]]})  # fmt: skip
    verdicts = both_orders(items, BETTER, "judge-runs")
    save_json("judge/verdicts.json", verdicts)
    tally: dict = {}
    for v in verdicts:
        t = tally.setdefault(f"{v['track']}/{v['arm']}", {"win": 0, "loss": 0, "tie": 0})
        t[v["result"]] += 1
    for k, t in sorted(tally.items()):
        print(f"{k:28} win {t['win']:3d} loss {t['loss']:3d} tie {t['tie']:3d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["check", "runs"])
    {"check": check, "runs": runs}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
