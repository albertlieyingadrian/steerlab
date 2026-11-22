"""Generate Sounds Like Me drafts with Inkling 2 Mini through OpenRouter, under a hard $3 cap.

    python run_inkling.py SPLIT [--authors N] [--arms k0,k8_rand] [--reasoning off|low] [--tag TAG] [--limit N]

Prompts are Episode 1's (gen.chat_messages, template v3), at temperature 0.7 with 400 output tokens and reasoning
off. Episode 1's own settings differed by contestant (its closed models ran with reasoning effort low through the batch
API, its open models with thinking off and top_p 0.95 on vLLM); the note lists them.

The cap fails closed. Each HTTP attempt reserves the worst case (the prompt plus the full token budget at the dearer
provider's listed price) before it is sent; OpenRouter is told to refuse providers dearer than that (max_price). An
attempt is charged OpenRouter's usage.cost when the reply carries one, and the full reservation when it does not
(a timeout, a dropped connection or a reply without usage). A call that would take the ledger past CAP is refused,
and the run stops with the Modal fallback command.

Resumable and safe to interrupt: each row is written as its call finishes, a job is skipped once a row with text
exists, and an exclusive lock stops two runs writing the same file. The run exits non-zero when any call failed.
Each draft records the provider and model OpenRouter served.
"""

import argparse
import fcntl
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sounds-like-me"))

from gen import ARMS, chat_messages, jobs, load_aux

MODEL = "thinkingmachines/inkling-2-mini"
CAP = 3.00  # hard cap for this study (2026-11-22); the key is shared with another study
PRICE_MAX = (0.08, 0.35)  # $/M (in, out), the dearer of the two listed providers (OpenRouter, 2026-11-22)
LEDGER = HERE / "data" / "spend.jsonl"
DRAFTS = HERE / "data" / "drafts"
MAX_TOKENS = {"off": 400, "low": 1600}  # "low": reasoning tokens count against max_tokens on OpenRouter
TRIES = 4
_lock = threading.Lock()
_reserved = [0.0]


class CapReached(Exception):
    pass


def _key():
    if os.environ.get("OPENROUTER_API_KEY_2"):
        return os.environ["OPENROUTER_API_KEY_2"]
    for d in [HERE, *HERE.parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY_2="):
                    return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("OPENROUTER_API_KEY_2 not found")


def spent():
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return sum(r["usd"] for r in rows if r["provider"] == "openrouter")


def reserve(usd):
    with _lock:
        if spent() + _reserved[0] + usd > CAP:
            return False
        _reserved[0] += usd
        return True


def settle(reserved, usd, note):
    with _lock:
        _reserved[0] -= reserved
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as fh:
            fh.write(
                json.dumps({"t": time.time(), "provider": "openrouter", "usd": round(usd, 7), "note": note}) + "\n"
            )


def post(body, key, worst, note):
    """Send one request with retries. Every attempt is reserved first and settled after (fail closed)."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last = "unreachable"
    for i in range(TRIES):
        if not reserve(worst):
            raise CapReached
        r, err = None, None
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            err = f"{e.code}: {msg}"
            if e.code == 402:
                settle(worst, 0.0, f"{note} (402, not billed)")
                raise SystemExit(f"OpenRouter 402 (payment): {msg}") from None
            if 400 <= e.code < 500 and e.code != 429:
                settle(worst, 0.0, f"{note} ({e.code}, rejected before generation)")
                return {"error": err}
        except Exception as e:  # timeouts, dropped connections, bad JSON: the provider may still have billed
            err = f"{type(e).__name__}: {e}"
        cost = (r or {}).get("usage", {}).get("cost") if r else None
        settle(
            worst,
            float(cost) if cost is not None else worst,
            note + ("" if cost is not None else " (cost unknown: reservation charged)"),
        )
        if r is not None and "error" not in r:
            return r
        last = err or json.dumps(r.get("error"))[:300]
        time.sleep(5 * (i + 1))
    return {"error": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--authors", type=int)
    ap.add_argument("--briefs", type=int, default=4)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--reasoning", choices=["off", "low"], default="off")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    key = _key()
    tag = a.tag or ("reason-low" if a.reasoning == "low" else "")
    slug = "inkling-2-mini" + (f"-{tag}" if tag else "")
    out = DRAFTS / f"{slug}__{a.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / f".{out.name}.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run is writing {out.name}") from None
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
    if any(r.get("reasoning") not in (None, a.reasoning) for r in rows):
        raise SystemExit(f"{out.name} holds drafts with another reasoning setting; pass a different --tag")
    done = {r["job"] for r in rows if r.get("text")}
    if len(done) < len(rows):  # drop failed rows before retrying them, so each job keeps one row
        kept = {}
        for r in rows:
            if r.get("text") or r["job"] not in done:
                kept[r["job"]] = r
        out.write_text("".join(json.dumps(r) + "\n" for r in kept.values() if r.get("text")))
    J = [j for j in jobs(a.split, a.authors, a.briefs, a.arms.split(",")) if j["job"] not in done]
    if a.limit:
        J = J[: a.limit]
    notes, edits = load_aux()
    max_tokens = MAX_TOKENS[a.reasoning]
    print(f"{slug} {a.split}: {len(J)} jobs to run ({len(done)} done); ledger ${spent():.4f} of ${CAP}")
    stop = threading.Event()

    def one(j):
        if stop.is_set():
            return None
        msgs = chat_messages(j["rec"], j["brief"], j["arm"], 0, notes, edits, "v3")
        n_in = sum(len(m["content"]) for m in msgs) / 3.0  # chars -> tokens, conservative
        worst = (n_in * PRICE_MAX[0] + max_tokens * PRICE_MAX[1]) / 1e6
        body = {
            "model": MODEL,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "usage": {"include": True},
            "reasoning": {"enabled": False} if a.reasoning == "off" else {"effort": "low", "exclude": True},
            "provider": {
                "order": ["primeintellect", "io-net"],
                "allow_fallbacks": True,
                "max_price": {"prompt": PRICE_MAX[0], "completion": PRICE_MAX[1]},
            },
        }
        t0 = time.time()
        try:
            r = post(body, key, worst, f"{slug} {a.split} {j['brief']['id']} {j['arm']}")  # no writer name
        except CapReached:
            stop.set()
            return None
        usage = r.get("usage") or {}
        ch = (r.get("choices") or [{}])[0]
        det = usage.get("completion_tokens_details") or {}
        err = r.get("error")
        return {
            "job": j["job"],
            "model": slug,
            "author": j["rec"]["author"],
            "brief": j["brief"]["id"],
            "arm": j["arm"],
            "text": ((ch.get("message") or {}).get("content") or "").strip(),
            "finish": ch.get("finish_reason"),
            "native_finish": ch.get("native_finish_reason"),
            "provider": r.get("provider"),
            "served_model": r.get("model"),
            "reasoning": a.reasoning,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": det.get("reasoning_tokens"),
            "usd": float(usage.get("cost") or 0.0),
            "seconds": round(time.time() - t0, 2),
            "error": None if not err else (err if isinstance(err, str) else json.dumps(err)[:300]),
        }

    n = failed = 0
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as fh:
        for fut in as_completed([ex.submit(one, j) for j in J]):
            try:
                row = fut.result()
            except SystemExit:
                stop.set()
                raise
            if row is None:
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            failed += bool(row["error"] or not row["text"])
            if n % 50 == 0:
                print(f"  {n}/{len(J)} ledger ${spent():.4f}", flush=True)
    print(f"wrote {n} rows ({failed} failed or empty) to {out.relative_to(HERE)}; ledger ${spent():.4f} of ${CAP}")
    if stop.is_set():
        raise SystemExit(
            f"Cap reached (${CAP}). Fallback: vLLM on Modal (MODAL_PROFILE=albertlieyingadrian); say so in the note."
        )
    if failed:
        raise SystemExit(f"{failed} calls failed or came back empty; rerun the same command to retry them")


if __name__ == "__main__":
    main()
