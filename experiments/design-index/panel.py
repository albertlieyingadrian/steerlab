"""
Choose which Wayback captures to render (the panel), from CACHE/cdx.jsonl. Deterministic per domain, so it can run
again as more CDX rows arrive and only new jobs are added.

Rule (recorded in the run log before any panel render was analysed):
  stable       a domain with a capture in at least 12 of the 13 years 2014-2026, including 2014 and 2026
  all-domain   a domain whose hash puts it in a fixed 40% sample of its category, whatever its history
Even years only (2014, 2016, ..., 2026): the archive's rate limits held renders to ~4 a minute (decision 8 in the
run log). Odd years already rendered are kept but not required. Every even captured year of a domain in either set
is rendered, domains in a fixed random order, so a partial run is a random subset of complete histories. The stable set answers "did the same sites converge?";
the all-domain sample keeps the natural mix of old and new sites (composition).
Writes CACHE/jobs_wayback.jsonl and CACHE/panel.jsonl ({"domain", "category", "n_years", "stable", "all_sample"}).
"""

from __future__ import annotations

import hashlib
import json

from common import CACHE, page_id


def in_sample(domain: str) -> bool:
    return (
        int(hashlib.sha256(("all|" + domain).encode()).hexdigest()[:8], 16) % 100 < 40
    )


def is_stable(caps: dict) -> bool:
    return len(caps) >= 12 and "2014" in caps and "2026" in caps


PER_CAT = 15  # cap per category and set (run log, decision 14): the archive allows ~220 renders an hour


def _h(d: str) -> str:
    return hashlib.sha256(d.encode()).hexdigest()


def main():
    seeds = {}
    for line in (CACHE / "seeds.jsonl").read_text().splitlines():
        r = json.loads(line)
        seeds[r["domain"]] = r
    cdx = {}
    for line in (CACHE / "cdx.jsonl").read_text().splitlines():
        r = json.loads(line)
        cdx[r["domain"]] = r["captures"]
    picked = {"stable": set(), "all": set()}
    for cat in sorted({seeds[d]["category"] for d in cdx}):
        doms = sorted((d for d in cdx if seeds[d]["category"] == cat), key=_h)
        picked["stable"] |= set([d for d in doms if is_stable(cdx[d])][:PER_CAT])
        picked["all"] |= set([d for d in doms if in_sample(d)][:PER_CAT])
    jobs, panel = [], []
    for d, caps in cdx.items():
        st, sa = d in picked["stable"], d in picked["all"]
        panel.append(
            {
                "domain": d,
                "category": seeds[d]["category"],
                "n_years": len(caps),
                "stable": st,
                "all_sample": sa,
            }
        )
        if not (st or sa):
            continue
        for y, ts in caps.items():
            if int(y) % 2:
                continue
            jobs.append(
                {
                    "pid": page_id("wayback", d, int(y)),
                    "domain": d,
                    "mode": "wayback",
                    "ts": ts,
                    "year": int(y),
                    "url": None,
                }
            )
    jobs.sort(key=lambda j: (_h(j["domain"]), j["year"]))
    (CACHE / "jobs_wayback.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in jobs)
    )
    (CACHE / "panel.jsonl").write_text("".join(json.dumps(p) + "\n" for p in panel))
    print(
        f"{len(panel)} domains: {sum(p['stable'] for p in panel)} stable, {sum(p['all_sample'] for p in panel)} "
        f"in the all-domain sample; {len(jobs)} captures to render"
    )


if __name__ == "__main__":
    main()
