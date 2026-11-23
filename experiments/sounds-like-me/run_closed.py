"""Generate drafts with a closed model through the OpenRouter Batch API (half price), under the hard $8 cap.

    python run_closed.py MODEL SPLIT [--authors N] [--briefs N] [--arms ...] [--submit|--poll BATCH_ID]

Before submitting, the cost is estimated from prompt length x batch price (+20% margin); if the ledger plus
that estimate would pass $8, the script refuses and prints the open-model fallback command instead.
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

from common import DATA, OPENROUTER_CAP, openrouter_allowed, record_spend, series_spent, spent
from gen import ARMS, chat_messages, jobs, load_aux

PRICE = {  # batch $/M tokens (in, out), OpenRouter models list 2026-11-20 (half of sync)
    "anthropic/claude-sonnet-5.5": (1.00, 5.00),
    "google/gemini-4-carbon": (1.00, 5.00),
}
KEY = os.environ.get("OPENROUTER_API_KEY_2") or os.environ.get("OPENROUTER_API_KEY")


def call(method, path, body=None):
    req = urllib.request.Request(
        f"https://openrouter.ai/api/v1{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"OpenRouter {e.code}: {e.read().decode()[:600]}")


def cid(j):
    return re.sub(r"[^A-Za-z0-9_-]", "_", j["job"].replace("|", "__"))


ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("split")
ap.add_argument("--authors", type=int)
ap.add_argument("--briefs", type=int, default=4)
ap.add_argument("--arms", default=",".join(ARMS))
ap.add_argument("--poll")
ap.add_argument("--tag", default="")
ap.add_argument("--template", default="v3")
a = ap.parse_args()
J = jobs(a.split, a.authors, a.briefs, a.arms.split(","))
slug = a.model.split("/")[-1] + (f"-{a.tag}" if a.tag else "")
out = DATA / "drafts" / f"{slug}__{a.split}.jsonl"
pending = DATA / "drafts" / f"{slug}__{a.split}.batch.json"

if not a.poll:
    notes, edits = load_aux()
    reqs = []
    est_in = 0
    for j in J:
        msgs = chat_messages(j["rec"], j["brief"], j["arm"], 0, notes, edits, a.template)
        est_in += sum(len(m["content"]) for m in msgs) / 3.6  # chars -> tokens (conservative)
        if "claude" in a.model:
            # Sonnet 5.5 reasoning is mandatory on this endpoint and counts against max_tokens (pilot: 361 of 400
            # tokens went to reasoning, drafts truncated). effort=low gives 0 reasoning tokens; Claude rejects a
            # custom temperature with reasoning, so it keeps its default. The visible email is still 60-200 words.
            body = {"messages": msgs, "max_tokens": 1000, "reasoning": {"effort": "low", "exclude": True}}
        else:
            body = {
                "messages": msgs,
                "max_tokens": 400,
                "temperature": 0.7,
                "reasoning": {"effort": "low", "exclude": True},
            }
        reqs.append({"custom_id": cid(j), "body": body})
    pin, pout = PRICE[a.model]
    est = 1.2 * (est_in * pin + len(J) * (600 if "claude" in a.model else 400) * pout) / 1e6
    if not openrouter_allowed(est):
        raise SystemExit(
            f"OpenRouter cap: series spent ${series_spent():.2f} + est ${est:.2f} > ${OPENROUTER_CAP}. "
            f"Fallback: python run_open.py Qwen/Qwen3-32B {a.split} ... (and say so in the note)"
        )
    r = call(
        "POST",
        "/batches",
        {"endpoint": "/v1/chat/completions", "model": a.model, "completion_window": "24h", "requests": reqs},
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(dict(id=r["id"], est=est, n=len(J))))
    record_spend("openrouter", est, f"RESERVED batch {r['id']} {slug} {a.split} n={len(J)}")
    print(
        f"submitted {r['id']} ({len(J)} requests), est ${est:.2f}; ledger now ${spent():.2f}, series ${series_spent():.2f}"
    )
else:
    while True:
        r = call("GET", f"/batches/{a.poll}")
        if r["status"] in ("completed", "failed", "expired", "cancelled"):
            break
        print(r["status"], r.get("request_counts"))
        time.sleep(60)
    meta = json.loads(pending.read_text())
    cost = (r.get("usage") or {}).get("cost")
    if cost is not None:  # replace the reservation with the real cost
        record_spend("openrouter", float(cost) - meta["est"], f"SETTLE batch {a.poll} real ${float(cost):.4f}")
    byid = {res["custom_id"]: res for res in (r.get("results") or [])}
    with out.open("w") as fh:
        for j in J:
            res = byid.get(cid(j)) or byid.get(j["job"].replace("|", "__"), {})
            body = (res.get("response") or {}).get("body") or res.get("body") or {}
            ch = (body.get("choices") or [{}])[0]
            text = ((ch.get("message") or {}).get("content") or "").strip()
            fh.write(
                json.dumps(
                    dict(
                        job=j["job"],
                        model=slug,
                        author=j["rec"]["author"],
                        brief=j["brief"]["id"],
                        arm=j["arm"],
                        text=text,
                        finish=ch.get("finish_reason"),
                    )
                )
                + "\n"
            )
    print(r["status"], r.get("request_counts"), "real cost", cost, "ledger", round(spent(), 4))
