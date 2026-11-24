"""Send data/pangram_items.jsonl (built by edits.py) to Pangram's bulk API -> results/pangram.json.

    python pangram_run.py           # print billed blocks and cost; sends nothing
    python pangram_run.py --run     # send (refused if it would pass the $60 cap in budget.py)

The client follows deslop-h2h/pangram.py (same endpoint, model `pangram-4`, 1,000 blocks per job, job ids kept in
results/pangram_jobs.json so an interrupted run resumes). Saved per item: prediction_short, fraction_ai,
fraction_ai_assisted, fraction_human, version, and every window's start, end, word count, label,
ai_assistance_score and class_probabilities. Never the text. The key is read from PANGRAM_API_KEY or a .env file
(PANGRAM_ENV) and never printed.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import budget
from common import DATA, RESULTS, blocks, read_jsonl

BASE = "https://text.external-api.pangram.com"
MODEL = "pangram-4"
USD_PER_BLOCK, JOB_BLOCKS = 0.04, 1000
OUT = RESULTS / "pangram.json"
JOBS = RESULTS / "pangram_jobs.json"


def api_key() -> str:
    key = os.environ.get("PANGRAM_API_KEY")
    envs = [Path(os.environ["PANGRAM_ENV"])] if os.environ.get("PANGRAM_ENV") else []
    envs += [Path(__file__).resolve().parents[2] / ".env"]
    for env in envs:
        if not key and env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("PANGRAM_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("no PANGRAM_API_KEY (set it, or PANGRAM_ENV to a .env file holding it)")
    return key


def call(method: str, path: str, key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json", "x-api-key": key},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"{method} {path}: HTTP {e.code} {e.read().decode()[:300]}") from e
        except urllib.error.URLError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{method} {path}: failed after retries")


def compact(res: dict) -> dict:
    wins = [
        {
            "start": w.get("start_index"),
            "end": w.get("end_index"),
            "words": w.get("word_count"),
            "label": w.get("label"),
            "ai": w.get("ai_assistance_score"),
            "probs": w.get("class_probabilities"),
        }
        for w in res.get("windows") or []
    ]
    return {
        "prediction_short": res.get("prediction_short"),
        "fraction_ai": res.get("fraction_ai"),
        "fraction_ai_assisted": res.get("fraction_ai_assisted"),
        "fraction_human": res.get("fraction_human"),
        "version": res.get("version"),
        "windows": wins,
    }


def collect(bulk_id: str, key: str) -> dict[str, dict]:
    while (s := call("GET", f"/bulk/{bulk_id}", key))["status"] in ("queued", "running"):
        print(f"  {bulk_id}: {s['status']} {s.get('succeeded', 0)}/{s['total_items']}")
        time.sleep(15)
    got: dict[str, dict] = {}
    off = 0
    while True:
        page = call("GET", f"/bulk/{bulk_id}/results?offset={off}&limit=1000", key)
        for it in page["items"]:
            if it.get("result"):
                got[it["id"]] = compact(it["result"])
        for it in page.get("failed_items", []):
            print(f"  failed: {it.get('id')} {it.get('error')}")
        off += 1000
        if off >= page["total_items"]:
            return got


def main() -> None:
    items = read_jsonl(DATA / "pangram_items.jsonl")
    T = {it["key"]: it["text"] for it in items}
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else []
    pending = {k for j in jobs if not j.get("collected") for k in j["keys"]}
    todo = [k for k in T if k not in done and k not in pending]
    n = sum(blocks(T[k]) for k in todo)
    print(
        f"{len(items)} items; to send {len(todo)}: {n} blocks, ${n * USD_PER_BLOCK:.2f}; "
        f"spent so far ${budget.spent('pangram'):.2f}"
    )
    if "--run" not in sys.argv:
        return
    budget.check("pangram", n * USD_PER_BLOCK)
    key = api_key()
    groups, cur, nb = [], [], 0
    for k in todo:
        b = blocks(T[k])
        if cur and nb + b > JOB_BLOCKS:
            groups.append(cur)
            cur, nb = [], 0
        cur.append(k)
        nb += b
    if cur:
        groups.append(cur)
    for c in groups:
        r = call("POST", "/bulk", key, {"items": [{"id": k, "text": T[k]} for k in c], "model": MODEL})
        b = sum(blocks(T[k]) for k in c)
        jobs.append(
            {
                "bulk_id": r["bulk_id"],
                "keys": c,
                "blocks": b,
                "usd": round(b * USD_PER_BLOCK, 2),
                "date": time.strftime("%F"),
            }
        )
        JOBS.write_text(json.dumps(jobs, indent=1) + "\n")
        budget.log(f"pangram {r['bulk_id']}", "pangram", b * USD_PER_BLOCK, blocks=b, texts=len(c))
        print(f"submitted {r['bulk_id']}: {len(c)} texts, {len(r.get('failed_items', []))} rejected")
    for j in jobs:
        if j.get("collected"):
            continue
        done.update(collect(j["bulk_id"], key))
        j["collected"] = True
        OUT.write_text(json.dumps(done) + "\n")
        JOBS.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"{len(done)} items scored -> {OUT}; Pangram spend ${budget.spent('pangram'):.2f}")


if __name__ == "__main__":
    main()
