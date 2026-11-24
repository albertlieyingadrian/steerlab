"""Pangram, the commercial detector, on a seeded sample of the texts rulers.py scores.

    python pangram.py            # build the sample and print its billed units and cost; sends nothing
    python pangram.py --run      # send it through Pangram's bulk API -> results/rulers/pangram.json

The sample: all 318 human passages (they set the 1% threshold), 9 replies per Track A arm and 15 per Track B run,
drawn with a fixed seed and spread over prompts. Pangram 4.1 bills each text by started 100-word block (at least one),
$0.04 per block on the bulk API (checked 2026-11-24), so the cost is computed from the texts before anything is sent,
and the run refuses to start if it would pass CAP_USD. Bulk jobs hold at most 1,000 blocks, so the sample goes out in
several jobs; their ids are kept in results/rulers/pangram_jobs.json so an interrupted run resumes (Pangram keeps
results for 48 hours). Only scores are saved, never the returned text. The key is read from PANGRAM_API_KEY (or the
repo's .env) and never printed.
Saved per key: p_ai = 1 - fraction_human (the score thresholded in table.py), fraction_ai, fraction_ai_assisted,
prediction_short, version, and the largest window humanizer_score.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

from common import HERE, RESULTS
from rulers import texts

BASE = "https://text.external-api.pangram.com"
MODEL = "pangram-4"
USD_PER_BLOCK, JOB_BLOCKS, CAP_USD = 0.04, 1000, 90.0
PER_ARM = {"A": 9, "B": 15}
SEED = 0
OUT = RESULTS / "rulers" / "pangram.json"
JOBS = RESULTS / "rulers" / "pangram_jobs.json"


def blocks(text: str) -> int:
    return max(1, math.ceil(len(text.split()) / 100))


def sample(T: dict[str, str]) -> list[str]:
    rng = random.Random(SEED)
    groups: dict[str, list[str]] = defaultdict(list)
    for k, t in T.items():
        if t.strip():
            groups["human" if k.startswith("human/") else "/".join(k.split("/")[:2])].append(k)
    keep = []
    for g, ks in sorted(groups.items()):
        if g == "human":
            keep += sorted(ks)
            continue
        by_prompt: dict[str, list[str]] = defaultdict(list)
        for k in sorted(ks):
            by_prompt[k.split("/")[2]].append(k)
        prompts = sorted(by_prompt)
        rng.shuffle(prompts)
        keep += [rng.choice(by_prompt[p]) for p in prompts[: PER_ARM[g[0]]]]
    return keep


def api_key() -> str:
    key = os.environ.get("PANGRAM_API_KEY")
    if not key:
        for env in (HERE.parents[1] / ".env", HERE.parents[1] / ".env.local"):
            if env.exists():
                for line in env.read_text().splitlines():
                    if line.startswith("PANGRAM_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("no PANGRAM_API_KEY in the environment or .env")
    return key


def call(method: str, path: str, key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "x-api-key": key})  # fmt: skip
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


def chunks(keys: list[str], T: dict[str, str]) -> list[list[str]]:
    out, cur, n = [], [], 0
    for k in keys:
        b = blocks(T[k])
        if cur and n + b > JOB_BLOCKS:
            out.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += b
    return [*out, cur] if cur else out


def compact(res: dict) -> dict:
    wins = res.get("windows") or []
    return {"p_ai": round(1.0 - float(res["fraction_human"]), 6),
            "fraction_ai": res.get("fraction_ai"), "fraction_ai_assisted": res.get("fraction_ai_assisted"),
            "prediction_short": res.get("prediction_short"), "version": res.get("version"),
            "humanizer_max": max((w.get("humanizer_score") or 0.0 for w in wins), default=0.0)}  # fmt: skip


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
    T = texts()
    keys = sample(T)
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    done = {} if done.get("status") == "not run" else done
    todo = [k for k in keys if k not in done]
    n_blocks = sum(blocks(T[k]) for k in todo)
    cost = n_blocks * USD_PER_BLOCK
    print(f"sample {len(keys)} texts ({sum(k.startswith('human/') for k in keys)} human); to send {len(todo)}: "
          f"{sum(len(T[k].split()) for k in todo):,} words, {n_blocks:,} blocks, ${cost:.2f}")  # fmt: skip
    if "--run" not in sys.argv:
        return
    if cost > CAP_USD:
        sys.exit(f"refused: ${cost:.2f} is over the ${CAP_USD:.0f} cap")
    key = api_key()
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else []
    pending = {k for j in jobs if not j.get("collected") for k in j["keys"]}
    for c in chunks([k for k in todo if k not in pending], T):
        r = call("POST", "/bulk", key, {"items": [{"id": k, "text": T[k]} for k in c], "model": MODEL})
        n = sum(blocks(T[k]) for k in c)
        jobs.append({"bulk_id": r["bulk_id"], "keys": c, "blocks": n, "usd": round(n * USD_PER_BLOCK, 2),
                     "date": time.strftime("%F")})  # fmt: skip
        JOBS.parent.mkdir(parents=True, exist_ok=True)
        JOBS.write_text(json.dumps(jobs, indent=1) + "\n")
        print(f"submitted {r['bulk_id']}: {len(c)} texts, {len(r.get('failed_items', []))} rejected")
    for j in jobs:
        if j.get("collected"):
            continue
        done.update(collect(j["bulk_id"], key))
        j["collected"] = True
        OUT.write_text(json.dumps(done, indent=0) + "\n")
        JOBS.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"{len(done)} texts scored by Pangram -> {OUT}; billed about ${sum(j['usd'] for j in jobs):.2f}")


if __name__ == "__main__":
    main()
