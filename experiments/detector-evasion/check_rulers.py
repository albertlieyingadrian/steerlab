"""Known-answer checks for the content rulers and the judge, on dev only, before any training.

    python check_rulers.py      # -> results/checks/rulers.json

For each of the 24 dev sources, four candidate "rewrites" with known answers:
  source      the source itself (keeps everything)
  untouched   the untouched model's rewrite with the task prompt (seed 0)
  first40     the source's first 40% of words (drops most of it)
  offtopic    another dev source of the same genre on a different topic (keeps nothing)
Scored by the reward's content checks (claim recall, fact overlap, bge similarity) and by the held-out fidelity judge.
Pass bars (plan): mean strict claim recall >= 0.9 on `source` and <= 0.25 on `offtopic`; fidelity means ordered
source > untouched > first40 > offtopic. Also: the judge's "which did a person write?" on 20 human passages vs 20
untouched replies (Episode 3's check, both orders; reported, no bar), and the 9B on the untouched rewrites.
"""

from __future__ import annotations

import itertools
import random

import modal
import numpy as np

import budget
from common import (
    APP,
    CLAIM_CHECK,
    EP3,
    FIDELITY,
    HUMAN,
    REWRITE_PROMPT,
    SEED,
    human_passages,
    load_json,
    read_jsonl,  # fmt: skip
    save_json,
    sources,
)


def first_words(text: str, n: int = 200) -> str:
    return " ".join(text.split()[:n])


def main() -> None:
    dev = sources("dev")
    by = {(s["prompt_id"][1:3], s["genre"]): s for s in dev}
    fn = modal.Function.from_name(APP, "rewrite")
    try:  # reuse the untouched rewrites of a previous check (seed 0, deterministic)
        untouched = [[u] for u in load_json("checks/rulers.json")["untouched"]]
    except FileNotFoundError:
        untouched = budget.paid("check-rewrites", "H100", 0.5, fn.remote, "",
                                [REWRITE_PROMPT.format(source=s["text"]) for s in dev], 1, seed=0)  # fmt: skip
    cands = []
    for s, u in zip(dev, untouched, strict=True):
        t = int(s["prompt_id"][1:3])
        other = by[(f"{(t + 1) % 4:02d}", s["genre"])]
        w = s["text"].split()
        for kind, text in (("source", s["text"]), ("untouched", u[0]["text"]),
                           ("first40", " ".join(w[: int(0.4 * len(w))])), ("offtopic", other["text"])):  # fmt: skip
            cands.append({"sid": s["sid"], "kind": kind, "source": s["text"], "rewrite": text, "claims": s["claims"]})
    R = modal.Cls.from_name(APP, "Reward")()
    J = modal.Cls.from_name(APP, "Judge")()
    con = budget.paid("check-content", "L40S", 0.3, R.content.remote, cands)
    det = budget.paid(
        "check-9b", "L40S", 0.2, R.detect.remote, [c["rewrite"] for c in cands if c["kind"] == "untouched"]
    )
    fid = budget.paid("check-fidelity", "H100", 0.6, J.rate.remote,
                      [FIDELITY.format(a=c["source"], b=c["rewrite"]) for c in cands])  # fmt: skip
    cq = [CLAIM_CHECK.format(text=c["rewrite"], claim=cl) for c in cands for cl in c["claims"]]
    yes = [x["pA"] > x["pB"] for x in budget.paid("check-claims-judge", "H100", 0.6, J.ask.remote, cq)]
    gem = [sum(yes[6 * i : 6 * i + 6]) / 6 for i in range(len(cands))]
    rng = random.Random(SEED)
    orig = [r["text"] for r in read_jsonl(EP3 / "results" / "gen" / "original.jsonl") if r["sample"] == 0]
    pairs = list(zip(rng.sample(human_passages(), 20), rng.sample(orig, 20), strict=True))
    ps = [HUMAN.format(a=first_words(h["text"]), b=first_words(o)) for h, o in pairs]
    ps += [HUMAN.format(a=first_words(o), b=first_words(h["text"])) for h, o in pairs]
    picks = budget.paid("check-human-judge", "H100", 0.3, J.ask.remote, ps)
    right = sum(picks[i]["pA"] >= picks[i]["pB"] and picks[20 + i]["pB"] > picks[20 + i]["pA"] for i in range(20))

    rows = []
    for c, s, f, g in zip(cands, con, fid, gem, strict=True):
        kept = {k: v for k, v in s.items() if k != "claim_p"}
        rows.append({"sid": c["sid"], "kind": c["kind"], **kept, "fidelity": f, "claims_judge": g})
    summary = {}
    for kind in ("source", "untouched", "first40", "offtopic"):
        rs = [r for r in rows if r["kind"] == kind]
        measures = ("claim_strict", "claim_soft", "fact", "sim", "fidelity", "claims_judge")
        summary[kind] = {m: round(float(np.mean([r[m] for r in rs])), 4) for m in measures}
    order = [summary[k]["fidelity"] for k in ("source", "untouched", "first40", "offtopic")]
    summary["pass"] = {
        "claim_source_ge_0.9": summary["source"]["claim_strict"] >= 0.9,
        "claim_offtopic_le_0.25": summary["offtopic"]["claim_strict"] <= 0.25,
        "fidelity_ordered": all(a > b for a, b in itertools.pairwise(order)),
        "claims_judge_ordered": all(
            summary[a]["claims_judge"] > summary[b]["claims_judge"]
            for a, b in itertools.pairwise(("source", "first40", "offtopic"))
        ),
    }
    s9 = [d["s9"] for d in det]
    summary["untouched_9b"] = {"mean_s9": round(float(np.mean(s9)), 6), "flagged": int(sum(x >= 0.372 for x in s9)),
                               "n": len(s9), "min_s9": round(float(min(s9)), 6)}  # fmt: skip
    summary["human_judge"] = {"right_both_orders": int(right), "n": 20}
    print(summary)
    save_json("checks/rulers.json", {"summary": summary, "rows": rows,
                                     "untouched": [u[0] for u in untouched], "untouched_9b": det})  # fmt: skip


if __name__ == "__main__":
    main()
