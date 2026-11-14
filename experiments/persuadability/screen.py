"""Knowledge screen and pilot on the untrained model (laptop, no API cost).

    python screen.py knowledge   # letter-only answer to every item; known = top letter is the correct one
    python screen.py pilot       # small W / C / N pilot on known items: flip, update and new-info rates
    python screen.py split       # known items -> data/split.json and data/eval_items.jsonl
    python screen.py confirm     # amendment 1: screen the fresh arithmetic and memo items; build the confirmation
                                 # set (unused known general knowledge + 100 fresh known arithmetic + 100 memo)
                                 # -> results/screen_fresh.jsonl, data/confirm_items.jsonl

Writes results/screen.jsonl, results/pilot.jsonl, data/split.json, data/eval_items.jsonl.
"""

import json
import random
import sys

import numpy as np

import common as C


def all_items():
    rows = []
    for f, dom in (("items_raw.jsonl", "know"), ("arith_raw.jsonl", "arith"), ("memo.jsonl", "memo")):
        for r in C.read_jsonl(C.DATA / f):
            r.setdefault("domain", dom)
            rows.append(r)
    return rows


def knowledge():
    tok, model = C.load()
    items = all_items()
    p = C.final_letter_probs(model, tok, [C.letter_prompt(tok, it) for it in items], batch=32)
    out = []
    for it, pr in zip(items, p, strict=False):
        out.append(
            {
                "id": it["id"],
                "domain": it["domain"],
                "p_correct": round(float(pr[it["correct"]]), 4),
                "known": bool(int(np.argmax(pr)) == it["correct"]),
            }
        )
    C.write_jsonl(C.RESULTS / "screen.jsonl", out)
    for dom in ("know", "arith", "memo"):
        rr = [r for r in out if r["domain"] == dom]
        print(dom, sum(r["known"] for r in rr), "/", len(rr), "known")


def pilot():
    tok, model = C.load()
    known = {r["id"] for r in C.read_jsonl(C.RESULTS / "screen.jsonl") if r["known"]}
    rng = random.Random(1)
    items = [it for it in all_items() if it["id"] in known]
    pick = []
    for dom, n in (("know", 40), ("arith", 20), ("memo", 20)):
        pool = [it for it in items if it["domain"] == dom]
        pick += rng.sample(pool, min(n, len(pool)))
    jobs = [(it, c) for it in pick for c in ("W", "C", "N") if c != "N" or it["domain"] == "memo"]
    convs = [C.conversation(it, c) for it, c in jobs]
    replies = C.generate(model, tok, [C.chat(tok, m) for m in convs], max_new_tokens=150, batch=32)
    probs = C.final_letter_probs(model, tok, C.followup_prompts(tok, convs, replies), batch=32)
    rows = []
    for (it, c), r, pr in zip(jobs, replies, probs, strict=False):
        fin = int(np.argmax(pr))
        target = {"W": it["wrong"], "C": it["correct"], "N": it.get("new_answer")}[c]
        rows.append(
            {"id": it["id"], "domain": it["domain"], "cond": c, "reply": r, "final": fin, "switched": fin == target}
        )
    C.write_jsonl(C.RESULTS / "pilot.jsonl", rows)
    for dom in ("know", "arith", "memo"):
        for c in ("W", "C", "N"):
            rr = [r for r in rows if r["domain"] == dom and r["cond"] == c]
            if rr:
                print(f"{dom:6s} {c}: switched to the user's option {sum(r['switched'] for r in rr)}/{len(rr)}")


def split():
    known = {r["id"] for r in C.read_jsonl(C.RESULTS / "screen.jsonl") if r["known"]}
    items = [it for it in all_items() if it["id"] in known]
    rng = random.Random(0)
    by = {d: [it for it in items if it["domain"] == d] for d in ("know", "arith", "memo")}
    for v in by.values():
        rng.shuffle(v)
    know = by["know"]
    test, dev, train = know[:120], know[120:180], know[180:630]
    benign = json.loads((C.DATA / "benign.json").read_text())
    ev = []
    for i, it in enumerate(test):
        it = dict(it, split="test")
        if i < 60:
            it["benign"] = benign[i % len(benign)]
        ev.append(it)
    ev += [dict(it, split="dev") for it in dev]
    ev += [dict(it, split="test") for it in by["arith"][:100]]
    ev += [dict(it, split="test") for it in by["memo"][:100]]
    C.write_jsonl(C.DATA / "eval_items.jsonl", ev)
    sp = {
        "train": [it["id"] for it in train],
        "dev": [it["id"] for it in dev],
        "test": [it["id"] for it in test],
        "arith": [it["id"] for it in by["arith"][:100]],
        "memo": [it["id"] for it in by["memo"][:100]],
    }
    (C.DATA / "split.json").write_text(json.dumps(sp, indent=0))
    print({k: len(v) for k, v in sp.items()})


def confirm():
    tok, model = C.load()
    fresh = []
    for f, dom in (("arith_fresh_raw.jsonl", "arith"), ("memo_fresh.jsonl", "memo")):
        fresh += [dict(r, domain=dom) for r in C.read_jsonl(C.DATA / f)]
    p = C.final_letter_probs(model, tok, [C.letter_prompt(tok, it) for it in fresh], batch=32)
    scr = [
        {
            "id": it["id"],
            "domain": it["domain"],
            "p_correct": round(float(pr[it["correct"]]), 4),
            "known": bool(int(np.argmax(pr)) == it["correct"]),
        }
        for it, pr in zip(fresh, p, strict=False)
    ]
    C.write_jsonl(C.RESULTS / "screen_fresh.jsonl", scr)
    known = {r["id"] for r in scr if r["known"]}
    sp = json.loads((C.DATA / "split.json").read_text())
    used = {i for v in sp.values() for i in v}
    known0 = {r["id"] for r in C.read_jsonl(C.RESULTS / "screen.jsonl") if r["known"]}
    know = [
        dict(it, domain="know")
        for it in C.read_jsonl(C.DATA / "items_raw.jsonl")
        if it["id"] in known0 and it["id"] not in used
    ]
    benign = json.loads((C.DATA / "benign.json").read_text())
    rows = []
    for i, it in enumerate(know):
        it = dict(it, split="confirm")
        if i < 60:
            it["benign"] = benign[i % len(benign)]
        rows.append(it)
    for dom in ("arith", "memo"):
        pool = [dict(it, split="confirm") for it in fresh if it["domain"] == dom and it["id"] in known]
        rows += pool[:100]
    C.write_jsonl(C.DATA / "confirm_items.jsonl", rows)
    print(
        {d: sum(r["domain"] == d for r in rows) for d in ("know", "arith", "memo")},
        "fresh known:",
        {d: sum(r["known"] for r in scr if r["domain"] == d) for d in ("arith", "memo")},
    )


if __name__ == "__main__":
    {"knowledge": knowledge, "pilot": pilot, "split": split, "confirm": confirm}[sys.argv[1]]()
