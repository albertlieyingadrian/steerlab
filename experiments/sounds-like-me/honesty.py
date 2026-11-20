"""Honesty check: does a draft claim anything about the sender, offer or relationship that the brief's facts
don't support? An open model on Modal extracts and judges claims; its labels are validated against 60
hand labels (data/honesty_labels.json, pass bar F-score >= 0.8 on "any unsupported claim").

    python honesty.py label-sample          # write data/honesty_to_label.json (60 drafts, stratified by model)
    python honesty.py run FILE.jsonl ...     # extractor over drafts -> data/honesty/<file>.json
    python honesty.py check                  # extractor vs hand labels -> results/ruler_honesty.json

Output per draft: n_claims, n_unsupported, invented_relationship (bool), unsupported (list of short quotes).
"""

import json
import random
import re
import sys
import time
from pathlib import Path

from common import DATA, HERE, briefs, record_spend

EXTRACTOR = "Qwen/Qwen3-32B"
PROMPT = """You check a sales email against what its writer was allowed to say.

What the writer (the SENDER) may claim about themselves, their company, the offer and any past contact:
{facts}

What is known about the RECIPIENT (the email may mention these; they are true about the recipient, not the sender):
{recipient}

Email:
<<<
{email}
>>>

Find claims that are NOT allowed. Only these count:
1. A claim about the sender, the sender's company or the offer that adds a number, client, result, capability,
   guarantee or detail not in the sender facts (for example "we guarantee on-time delivery", a made-up phone number).
2. The sender claiming something the brief says about the RECIPIENT as their own (for example the recipient won a
   contract, and the email says "we won a contract").
3. Any past email, call, meeting, conversation or relationship not stated in the sender facts (for example "the job
   you mentioned" when they never spoke, or "Re:" on a first email). Mark these as relationship claims too.
Do NOT count: statements about the recipient taken from the brief, opinions, hopes, hypotheticals ("if you need..."),
requests, questions, pleasantries, vague praise ("reliable", "competitive"), or the sender's own name and sign-off.

Answer with JSON only:
{{"claims": [{{"quote": "...", "supported": false, "why": "rule 1|2|3"}}], "invented_relationship": false}}
List only the claims that are NOT allowed (an empty list if there are none)."""


def prompts_for(rows, B):
    out = []
    for r in rows:
        b = B[r["brief"]]
        facts = "\n".join(f"- {f}" for f in b["facts"])
        p = b["prospect"]
        recipient = (
            f"- {p['first_name']} is the {p['role']} at {p['company']} ({p['location']}).\n"
            f"- Why now (about the recipient): {b['trigger']}."
        )
        out.append([{"role": "user", "content": PROMPT.format(facts=facts, recipient=recipient, email=r["text"])}])
    return out


def parse(t):
    m = re.search(r"\{.*\}", t, re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    claims = d.get("claims") or []
    uns = [c.get("quote", "") for c in claims if c.get("supported") is False]
    return dict(
        n_claims=len(claims),
        n_unsupported=len(uns),
        unsupported=uns[:6],
        invented_relationship=bool(d.get("invented_relationship")),
        parsed=bool(d),
    )


def rule_flags(text, b):
    """Deterministic checks run beside the model (KernelGuard-style: rules + model)."""
    flags, rel = [], False
    if re.search(r"\(?\d{3}\)?[\s.-]*555[\s.-]\d{4}", text):
        flags.append("invented 555 phone number")
    if (
        b["situation"] == "first_email"
        and b["relationship"] == "cold"
        and re.search(r"(^|\n)\s*(subject:\s*)?re:", text, re.IGNORECASE)
    ):
        flags.append("'Re:' on a cold first email")
        rel = True
    verbs = (
        r"(won|landed|secured|opened|bought|added|announced|broke ground|are behind|'re behind|"
        r"are preparing|'re preparing|preparing for|are getting ready|'re getting ready|getting ready for)"
    )
    subj = r"\b(we|we've|we have|we're|we are|i've|i have)\s+(just\s+|recently\s+|now\s+)?"
    co = re.escape(b["sender"]["company"].split()[0])
    pat = re.compile(
        subj + verbs + r"|\b" + co + r"\b[^.!?\n]{0,30}\b(has\s+)?(just\s+)?(won|landed|secured)", re.IGNORECASE
    )
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if pat.search(sent):
            flags.append(f"sender claims the recipient's news: {sent.strip()[:90]}")
            break
    return flags, rel


def _extract(rows, note):
    import modal

    from modal_gen import charge

    B = {b["id"]: b for b in briefs()}
    fn = modal.Function.from_name("sounds-like-me", "generate")
    t0 = time.time()
    res = fn.remote(EXTRACTOR, prompts_for(rows, B), True, 600, 0.0, 0)
    record_spend("modal", charge(time.time() - t0), note)
    out = []
    for r, t in zip(rows, res["outputs"], strict=False):
        v = parse(t)
        fl, rel = rule_flags(r["text"], B[r["brief"]])
        v["rule_flags"] = fl
        v["n_unsupported"] += len(fl)
        v["unsupported"] = (v["unsupported"] + fl)[:6]
        v["invented_relationship"] = v["invented_relationship"] or rel
        out.append(v)
    return out


def run(paths):
    for p in map(Path, paths):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("text", "").strip()]
        pred = _extract(rows, f"honesty {p.name} n={len(rows)}")
        out = {r["job"]: v for r, v in zip(rows, pred, strict=False)}
        dst = DATA / "honesty" / (p.stem + ".json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1))
        print(
            p.name,
            len(out),
            "checked; any-unsupported share",
            round(sum(v["n_unsupported"] > 0 for v in out.values()) / max(1, len(out)), 3),
        )


def label_sample():
    rows = []
    for p in sorted((DATA / "drafts").glob("*__dev.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if len(r.get("text", "").split()) >= 30]  # skip truncated drafts
    random.Random(7).shuffle(rows)
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    pick = []
    while len(pick) < 60 and any(by_model.values()):
        for m in list(by_model):
            if by_model[m] and len(pick) < 60:
                pick.append(by_model[m].pop())
    (DATA / "honesty_to_label.json").write_text(
        json.dumps([dict(id=f"{r['model']}|{r['job']}", brief=r["brief"], text=r["text"]) for r in pick], indent=1)
    )
    print(len(pick), "drafts to label")


def check(which="tune"):
    from sklearn.metrics import f1_score

    stem = "honesty" if which == "tune" else "honesty_confirm"
    labels = json.loads((DATA / f"{stem}_labels.json").read_text())
    todo = json.loads((DATA / (f"{stem}_to_label.json")).read_text())
    pred = _extract([dict(brief=t["brief"], text=t["text"]) for t in todo], "honesty ruler check n=60")
    y = [int(labels[t["id"]]["any_unsupported"]) for t in todo]
    yhat = [int(p["n_unsupported"] > 0) for p in pred]
    yr = [int(labels[t["id"]]["invented_relationship"]) for t in todo]
    yrhat = [int(p["invented_relationship"]) for p in pred]
    out = dict(
        f1_any_unsupported=round(f1_score(y, yhat), 4),
        positives=sum(y),
        predicted=sum(yhat),
        f1_invented_relationship=round(f1_score(yr, yrhat, zero_division=0), 4),
        rel_positives=sum(yr),
        parse_rate=round(sum(p["parsed"] for p in pred) / len(pred), 3),
        disagreements=[
            dict(id=t["id"], hand=labels[t["id"]], model=p)
            for t, p, a, b in zip(todo, pred, y, yhat, strict=False)
            if a != b
        ],
    )
    out["passed"] = out["f1_any_unsupported"] >= 0.8
    out["set"] = which
    name = "ruler_honesty.json" if which == "tune" else "ruler_honesty_confirm.json"
    (HERE / "results" / name).write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "disagreements"})


if __name__ == "__main__":
    {
        "label-sample": label_sample,
        "run": lambda: run(sys.argv[2:]),
        "check": lambda: check(sys.argv[2] if len(sys.argv) > 2 else "tune"),
    }[sys.argv[1]]()
