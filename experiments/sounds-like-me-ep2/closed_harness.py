"""Optional closed-model harness: Gemini 4 Carbon in the same harness, through the OpenRouter Batch API.

Spec: run only if >= $1.50 of the series' $8 OpenRouter cap is left, and only if a dev pilot prices the held-out run
under the remaining cap minus $0.50. Reasoning low/excluded, temperature 0.9, <= 400 output tokens. Every submit checks
the ledger first (common.openrouter_allowed) and refuses if the estimate would pass the cap.

Same steps as the open harness (retrieved samples + verified profile, Episode 1's system prompt), with N = 4 drafts
and 2 revisions per brief. The in-loop checks are the open harness's (LUAR to the samples, detector, copy filter,
Qwen3-8B honesty self-check on Modal): here the checker is not the policy model, which the note says.

    python closed_harness.py draft SPLIT [N]     # submit the drafts batch
    python closed_harness.py revise SPLIT [N]    # poll drafts, run checks, submit the revisions batch
    python closed_harness.py pick SPLIT [N]      # poll revisions, run checks, pick -> data/drafts/G2_gemini_harness__SPLIT.jsonl
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import ep2
from ep2 import (
    DATA,
    RES,
    A_by_name,
    B_by_id,
    critique,
    harness_messages,
    in_loop_checks,
    rank_score,
    revise_messages,
    write_pool_and_pick,
)

sys.path.append(str(ep2.EP1))
from common import OPENROUTER_CAP, openrouter_allowed, record_spend, spent
from gen import jobs

MODEL = "google/gemini-4-carbon"
PRICE_IN, PRICE_OUT = 1.00, 5.00  # batch $/M tokens (Episode 1's table)
N_DRAFT, N_REVISE = 4, 2
NAME = "G2_gemini_harness"
CFG = dict(ep2.ARMS["R2_harness"], n_draft=N_DRAFT, n_revise=N_REVISE)
STATE = DATA / "closed_batches.json"


def _key():
    k = os.environ.get("OPENROUTER_API_KEY_2") or os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    env = ep2.HERE.parents[1] / ".env"  # the lab root's .env; read, never printed
    for root in (ep2.HERE.parents[1], *ep2.HERE.parents[1].parents):
        if (root / ".env").exists():
            env = root / ".env"
            break
    vals = dict(l.split("=", 1) for l in env.read_text().splitlines() if "=" in l and not l.lstrip().startswith("#"))
    return (vals.get("OPENROUTER_API_KEY_2") or vals.get("OPENROUTER_API_KEY") or "").strip().strip('"')


def call(method, path, payload=None):
    req = urllib.request.Request(
        f"https://openrouter.ai/api/v1{path}",
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"OpenRouter {e.code}: {e.read().decode()[:400]}")


def cid(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def submit(reqs, est_in_chars, stage, split):
    est = 1.2 * (est_in_chars / 3.6 * PRICE_IN + len(reqs) * 400 * PRICE_OUT) / 1e6
    if not openrouter_allowed(est):
        raise SystemExit(f"OpenRouter cap: spent ${spent():.2f} + est ${est:.2f} > ${OPENROUTER_CAP}; skip and say so")
    payload = {"endpoint": "/v1/chat/completions", "model": MODEL, "completion_window": "24h", "requests": reqs}
    r = call("POST", "/batches", payload)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st[f"{stage}__{split}"] = dict(id=r["id"], est=est, n=len(reqs))
    STATE.write_text(json.dumps(st, indent=1))
    record_spend("openrouter", est, f"RESERVED batch {r['id']} ep2 {NAME} {stage} {split} n={len(reqs)}")
    print(f"submitted {r['id']} ({len(reqs)}), est ${est:.3f}; ledger ${spent():.3f}")


def body(msgs):
    return {"messages": msgs, "max_tokens": 400, "temperature": 0.9, "reasoning": {"effort": "low", "exclude": True}}


def poll(stage, split):
    meta = json.loads(STATE.read_text())[f"{stage}__{split}"]
    while True:
        r = call("GET", f"/batches/{meta['id']}")
        if r["status"] in ("completed", "failed", "expired", "cancelled"):
            break
        print(r["status"], r.get("request_counts"), flush=True)
        time.sleep(60)
    cost = (r.get("usage") or {}).get("cost")
    if cost is not None:
        record_spend("openrouter", float(cost) - meta["est"], f"SETTLE batch {meta['id']} real ${float(cost):.4f}")
    out = {}
    for res in r.get("results") or []:
        b = (res.get("response") or {}).get("body") or res.get("body") or {}
        ch = (b.get("choices") or [{}])[0]
        out[res["custom_id"]] = ((ch.get("message") or {}).get("content") or "").strip()
    p = RES / "closed_cost.json"
    cost_m = json.loads(p.read_text()) if p.exists() else {}
    cost_m[f"{stage}__{split}"] = dict(
        usd=cost if cost is not None else meta["est"], n=meta["n"], settled=cost is not None
    )
    p.write_text(json.dumps(cost_m, indent=1))
    print(r["status"], r.get("request_counts"), "real cost", cost)
    return out


def draft(split, n):
    P = ep2.profiles(split, n)
    reqs, chars = [], 0
    for j in jobs(split, n, 4, ["k0"]):
        msgs = harness_messages(j["rec"], j["brief"], CFG, P[j["rec"]["author"]])
        job = j["job"].rsplit("|", 1)[0]
        for k in range(N_DRAFT):
            reqs.append({"custom_id": cid(f"{job}__d{k}"), "body": body(msgs)})
            chars += sum(len(m["content"]) for m in msgs)
    submit(reqs, chars, "draft", split)


def _cands(split, n):
    A, B = A_by_name(), B_by_id()
    got = poll("draft", split)
    cands = []
    for j in jobs(split, n, 4, ["k0"]):
        job = j["job"].rsplit("|", 1)[0]
        for k in range(N_DRAFT):
            t = got.get(cid(f"{job}__d{k}"), "")
            cands.append(
                dict(job=job, author=j["rec"]["author"], brief=j["brief"]["id"], stage="draft", k=k, text=t, nll=None)
            )
    in_loop_checks(cands, A, B, f"{NAME} {split} drafts")
    (DATA / "pools").mkdir(parents=True, exist_ok=True)
    (DATA / "pools" / f"{NAME}__{split}.drafts.json").write_text(json.dumps(cands))
    return cands


def revise(split, n):
    A, B = A_by_name(), B_by_id()
    P = ep2.profiles(split, n)
    cands = _cands(split, n)
    by = {}
    for c in cands:
        by.setdefault(c["job"], []).append(c)
    reqs, chars, notes_m = [], 0, {}
    for job, cs in by.items():
        au, bid = cs[0]["author"], cs[0]["brief"]
        msgs = harness_messages(A[au], B[bid], CFG, P[au])
        for c in sorted([c for c in cs if not c["empty"]], key=lambda c: -rank_score(c, ep2.LAM))[:N_REVISE]:
            m = revise_messages(msgs, c["text"], critique(c, P[au], use_profile=CFG["profile"]))
            key = cid(f"{job}__r{c['k']}")
            reqs.append({"custom_id": key, "body": body(m)})
            chars += sum(len(x["content"]) for x in m)
            notes_m[key] = dict(job=job, author=au, brief=bid, k=c["k"])
    (DATA / "pools" / f"{NAME}__{split}.revise_meta.json").write_text(json.dumps(notes_m))
    submit(reqs, chars, "revise", split)


def pick_final(split, n):
    A, B = A_by_name(), B_by_id()
    cands = json.loads((DATA / "pools" / f"{NAME}__{split}.drafts.json").read_text())
    meta = json.loads((DATA / "pools" / f"{NAME}__{split}.revise_meta.json").read_text())
    got = poll("revise", split)
    revs = [dict(m, stage="revise", text=got.get(k, ""), nll=None) for k, m in meta.items()]
    in_loop_checks(revs, A, B, f"{NAME} {split} revisions")
    write_pool_and_pick(NAME, split, cands + revs, ep2.LAM, gsecs=0.0, wall=0.0)


if __name__ == "__main__":
    cmd, split = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dict(draft=draft, revise=revise, pick=pick_final)[cmd](split, n)
