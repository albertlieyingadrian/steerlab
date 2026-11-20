"""
Build the stratified seed list with Exa neural search (discovery only: we render every page ourselves).

  human   12 categories x 4 phrasings x 3 regions, 25 results each; platform hosts and AI-builder hosts dropped
  ai      12 categories x 4 phrasings, restricted to the AI-builder hosts in common.AI_HOSTS

One homepage per domain (the first time a domain appears; its category is that query's). Writes
CACHE/seeds.jsonl, rows: {"domain", "url", "category", "cls": "human"|"ai", "query", "published"}.
Costs money (hard cap in budget.py). Run: .venv/bin/python experiments/design-index/seeds.py [--only human|ai]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from budget import EXA_PRICE_25, check_exa, exa_spent, log_exa
from common import (
    AI_HOSTS,
    CACHE,
    CATEGORIES,
    PLATFORM_HOSTS,
    REGIONS,
    domain_of,
    is_ai_host,
    registrable,
)

# the repo root's .env files; in a worktree under .claude/worktrees/ the main checkout's root is three levels further up
_ROOTS = [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[5]]
ENV_FILES = [r / n for r in _ROOTS if (r / ".git").exists() for n in (".env", ".env.local")]


def exa_key() -> str:
    if os.environ.get("EXA_SEARCH_API_KEY"):
        return os.environ["EXA_SEARCH_API_KEY"]
    for f in ENV_FILES:
        if f.exists():
            for line in f.read_text().splitlines():
                if line.startswith("EXA_SEARCH_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("EXA_SEARCH_API_KEY not set")


def search(query: str, include_domains: list[str] | None = None) -> list[dict]:
    check_exa()
    body = {"query": query, "numResults": 25, "type": "auto"}
    if include_domains:
        body["includeDomains"] = include_domains
    req = urllib.request.Request(
        "https://api.exa.ai/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-api-key": exa_key()},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry after {e.__class__.__name__}")
            time.sleep(3)
    cost = float((data.get("costDollars") or {}).get("total", EXA_PRICE_25))
    results = data.get("results", [])
    log_exa(query, cost, len(results))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["human", "ai"])
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "seeds.jsonl"
    seen = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            seen[r["domain"]] = r
    done_q = {(r["cls"], r["query"]) for r in seen.values()}
    jobs = []
    if args.only in (None, "human"):
        jobs += [("human", cat, q + reg) for cat, qs in CATEGORIES.items() for q in qs for reg in REGIONS]
    if args.only in (None, "ai"):
        jobs += [("ai", cat, q) for cat, qs in CATEGORIES.items() for q in qs]
    with out.open("a") as f:
        for cls, cat, q in jobs:
            if (cls, q) in done_q:
                continue
            res = search(q, AI_HOSTS if cls == "ai" else None)
            new = 0
            for r in res:
                d = domain_of(r["url"])
                if not d or d in seen:
                    continue
                if cls == "human" and (registrable(d) in PLATFORM_HOSTS or is_ai_host(d)):
                    continue
                if cls == "ai" and not is_ai_host(d):
                    continue
                row = {
                    "domain": d,
                    "url": f"https://{d}/",
                    "category": cat,
                    "cls": cls,
                    "query": q,
                    "published": r.get("publishedDate"),
                }
                seen[d] = row
                f.write(json.dumps(row) + "\n")
                new += 1
            f.flush()
            print(f"{cls:5s} {cat:10s} +{new:2d}  ${exa_spent():.3f}  {q}")
    by = {}
    for r in seen.values():
        by.setdefault((r["cls"], r["category"]), 0)
        by[(r["cls"], r["category"])] += 1
    for k in sorted(by):
        print(k, by[k])


if __name__ == "__main__":
    main()
