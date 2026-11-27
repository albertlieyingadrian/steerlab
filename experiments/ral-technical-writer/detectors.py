"""
The two paid outside APIs: StealthGPT (a baseline rewriter) and Pangram (a side-measure). Keys are read from the
lab's .env and never printed; every call is priced and capped by budget.charge_api first.

    python detectors.py stealth [N]   # base sample 0 of the first N held-out titles -> samples/stealth_heldout.jsonl
    python detectors.py pangram       # 30 texts per source (first 30 held-out titles) -> results/pangram.jsonl

StealthGPT: POST https://www.stealthgpt.ai/api/stealthify with {prompt: <text only>, rephrase: true, model: "super"}
(docs.stealthgpt.ai, read 2026-11-27: send only the source text, no instruction wrapper; $0.05 per 100 words).
Pangram: POST https://text.api.pangram.com/v3 {text} with x-api-key (the lab's existing client; $0.05 per 100 words,
pangram.com/pricing 2026-11-27). Pangram is never used as a reward or to choose anything.
Outputs of StealthGPT stay in the lab cache; the note publishes scores and at most two short excerpts.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget
import common as C

LAB_ENV = Path(os.environ.get("KEYS_ENV", Path(__file__).resolve().parents[2] / ".env"))  # the repo root .env
STEALTH_URL = "https://www.stealthgpt.ai/api/stealthify"
PANGRAM_URL = "https://text.api.pangram.com/v3"
LOCAL = C.CACHE / "vol"


def key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in LAB_ENV.read_text().splitlines():
        if line.strip().startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in the environment or the lab .env")


def post(url: str, body: dict, headers: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 402, 403):
                raise RuntimeError(f"{url}: HTTP {e.code} {e.read()[:200]!r}") from None
            time.sleep(10 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: failed 4 times")


def stealth(n: int = 100) -> None:
    src = {r["appl_id"]: r for r in C.read_jsonl(LOCAL / "samples" / "base_heldout.jsonl") if r["i"] == 0}
    held = C.read_jsonl(C.CACHE / "heldout.jsonl")[:n]
    out_p = LOCAL / "samples" / "stealth_heldout.jsonl"
    done = {r["appl_id"]: r for r in C.read_jsonl(out_p)} if out_p.exists() else {}
    k = key("STEALTH_GPT_API_KEY")
    for x in held:
        s = src[x["appl_id"]]
        if x["appl_id"] in done or not s["answer"]:
            continue
        budget.check_api("stealthgpt", 6 * len(s["answer"].split()))  # super bills about 5.5x the input words
        t0 = time.time()
        d = post(STEALTH_URL, {"prompt": s["answer"], "rephrase": True, "model": "super"}, {"api-token": k})
        budget.log_api("stealthgpt", int(d.get("wordsSpent") or 0), f"stealth:{x['appl_id']}", note="super, wordsSpent")
        done[x["appl_id"]] = {"appl_id": x["appl_id"], "i": 0, "answer": (d.get("result") or "").strip() or None,
                              "source": s["answer"], "seconds": round(time.time() - t0, 1),
                              "meta": {kk: v for kk, v in d.items() if kk != "result"}}  # fmt: skip
        C.write_jsonl(out_p, list(done.values()))
        if int(d.get("remainingCredits") or 0) < 1100:  # the account's prepaid word credits, not our cap
            print("stealth: prepaid credits nearly used up; stopping")
            break
    print(f"stealth: {len(done)} rewrites; spent ${budget.api_spent('stealthgpt'):.2f}")


PANGRAM_SOURCES = ["ref", "base_heldout", "sft_heldout", "ral_heldout", "frozen_heldout", "llamafs_heldout",
                   "humanizer_heldout", "stealth_heldout"]  # fmt: skip


def pangram(n: int = 30) -> None:
    held = C.read_jsonl(C.CACHE / "heldout.jsonl")[:n]
    out_p = C.RESULTS / "pangram.jsonl"
    done = {(r["source"], r["appl_id"]): r for r in C.read_jsonl(out_p)} if out_p.exists() else {}
    k = key("PANGRAM_API_KEY")
    for s in PANGRAM_SOURCES:
        if s == "ref":
            texts = {x["appl_id"]: x["abstract"] for x in held}
        else:
            p = LOCAL / "samples" / f"{s}.jsonl"
            if not p.exists():
                print(f"pangram: {s} missing, skipped")
                continue
            texts = {r["appl_id"]: r["answer"] for r in C.read_jsonl(p) if r.get("i", 0) == 0}
        for x in held:
            t = texts.get(x["appl_id"])
            if not t or (s, x["appl_id"]) in done:
                continue
            budget.charge_api("pangram", len(t.split()), f"pangram:{s}:{x['appl_id']}")
            raw = post(PANGRAM_URL, {"text": t}, {"x-api-key": k}, timeout=120)
            keep = {
                kk: v for kk, v in raw.items() if not isinstance(v, (list, dict)) and kk != "text"
            }  # no text in results
            done[(s, x["appl_id"])] = {"source": s, "appl_id": x["appl_id"], **keep}
            C.write_jsonl(out_p, list(done.values()))
    print(f"pangram: {len(done)} texts; spent ${budget.api_spent('pangram'):.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {"stealth": lambda: stealth(arg or 100), "pangram": lambda: pangram(arg or 30)}[cmd]()
