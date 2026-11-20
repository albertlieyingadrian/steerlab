"""
Find one Wayback Machine homepage capture per year (2014-2026) for each human seed domain.

Politeness (Internet Archive terms: access for scholarship and research): one CDX request at a time, at least
3 s apart, a descriptive User-Agent, and on HTTP 429 or 5xx a pause that doubles from 60 s. Only the CDX index is
read here; pages are fetched later by the renderer, at a similarly low rate.

Snapshot rule (fixed in the plan): among monthly-collapsed 200/text-html captures, the one closest to 1 July.
Writes CACHE/cdx.jsonl, rows: {"domain", "captures": {"2014": "20140703121314", ...}}.
Run: .venv/bin/python experiments/design-index/wayback.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime

from common import CACHE, YEARS

UA = "design-index-research/0.1 (academic study of web design over time; low-rate)"
CDX = (
    "https://web.archive.org/cdx/search/cdx?url={d}/&output=json&fl=timestamp,statuscode,mimetype"
    "&filter=statuscode:200&filter=mimetype:text/html&from=2014&to=2026&collapse=timestamp:6"
)


def fetch(url: str) -> list | None:
    """CDX rows, [] when the archive has none, None when every attempt failed (not recorded, retried next run)."""
    pause = 60
    for _ in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read().decode() or "[]"
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (400, 403):  # the archive refusing this client: not "no captures"; retry on a later run
                print(f"  HTTP {e.code}: refused, not recorded")
                return None
            if e.code in (429, 500, 502, 503, 504):
                print(f"  HTTP {e.code}: pausing {pause}s")
                time.sleep(pause)
                pause *= 2
                continue
            return []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"  {e.__class__.__name__}: pausing 20s")
            time.sleep(20)
    return None


def pick(rows: list) -> dict:
    by_year = {}
    for ts, *_ in rows[1:]:
        y = int(ts[:4])
        if y not in YEARS:
            continue
        target = datetime(y, 7, 1)
        try:
            dt = datetime.strptime(ts[:8], "%Y%m%d")
        except ValueError:
            continue
        gap = abs((dt - target).days)
        if y not in by_year or gap < by_year[y][0]:
            by_year[y] = (gap, ts)
    return {str(y): ts for y, (_, ts) in sorted(by_year.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--per-cat",
        type=int,
        default=100,
        help="random domains per category (seeded); 0 = all",
    )
    args = ap.parse_args()
    seeds = [json.loads(line) for line in (CACHE / "seeds.jsonl").read_text().splitlines()]
    seeds = [s for s in seeds if s["cls"] == "human"]
    if args.per_cat:  # a fixed random draw per category, so the panel is not the first queries' results
        rng = random.Random(0)
        by = {}
        for s in seeds:
            by.setdefault(s["category"], []).append(s)
        seeds = [s for cat in sorted(by) for s in rng.sample(by[cat], min(args.per_cat, len(by[cat])))]
        rng.shuffle(seeds)
    out = CACHE / "cdx.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["domain"] for line in out.read_text().splitlines()}
    todo = [s for s in seeds if s["domain"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to go")
    with out.open("a") as f:
        for i, s in enumerate(todo):
            t0 = time.time()
            rows = fetch(CDX.format(d=s["domain"]))
            if rows is None:
                print(f"  gave up on {s['domain']} for now")
                continue
            caps = pick(rows)
            f.write(json.dumps({"domain": s["domain"], "captures": caps}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)} {s['domain']}: {len(caps)} years")
            time.sleep(max(0.0, 3.0 - (time.time() - t0)))


if __name__ == "__main__":
    main()
