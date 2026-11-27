"""
Build the reference set: abstracts of NEW NIH research grants awarded in fiscal year 2026, from NIH RePORTER.

    python prepare_data.py fetch   # RePORTER API v2 -> ~/.cache/steerlab/ral-writer/raw.jsonl (free, ~2 min)
    python prepare_data.py build   # clean, filter, dedupe, split by award date -> splits + results/data_card.json

Why this set (the plan's "Reference set"):
- Recent. FY2026 runs 2025-10-01 to 2026-09-30, and an abstract becomes public only when the award is made, so
  every abstract here was published after the generator's and the judge's training data were collected (Qwen3,
  released 2025-04).
- New applications only (award type 1). A continuation (type 5) or renewal (type 2) repeats or edits an abstract
  that RePORTER already published in an earlier year, which a 2025 model may have read.
- Research grants only (R01, R21, R03, R15, R34, R61, U01, R37): one genre, the "Project Summary/Abstract" of a
  research proposal. Fellowships, centres and training grants read differently.
- Human-written: not guaranteed. Applications behind FY2026 awards were written in 2024-2025, when writing
  assistants were common; NIH's policy against applications "substantially developed by AI" (NOT-OD-25-132) applies
  to receipt dates from 2025-09-25, i.e. only to the latest awards. No detector is used to filter the set (a
  detector would then define "human"); the note reports Pangram's labels on a held-out sample as a side-measure.

Split by time, so the held-out titles come from the latest awards: train (notice date before 2026-08-01), dev
(64 titles from train's date range, never trained on, used to watch training), held-out (notice date from
2026-09-01; 200 titles, the only set the note's headline numbers use). August is a gap month.

Licence and provenance: RePORTER data are published by NIH (a U.S. government agency) for public use; each abstract
was written by the applicant. The repo ships appl_ids and the split, not the abstracts; the note quotes at most a
few short excerpts, each cited by project number.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "steerlab" / "ral-writer"
RAW = CACHE / "raw.jsonl"
URL = "https://api.reporter.nih.gov/v2/projects/search"
CODES = ["R01", "R21", "R03", "R15", "R34", "R61", "U01", "R37"]
FIELDS = ["ApplId", "ProjectNum", "CoreProjectNum", "FiscalYear", "ProjectTitle", "AbstractText", "ActivityCode",
          "AwardNoticeDate", "ProjectStartDate", "AgencyIcAdmin", "AwardType"]  # fmt: skip
TRAIN_BEFORE, HELDOUT_FROM = "2026-08-01", "2026-09-01"
N_DEV, N_HELDOUT = 64, 200
MIN_WORDS, MAX_WORDS = 150, 450
SEED = 20261127


def post(body: dict) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # RePORTER asks for at most one request a second; back off on errors
            print(f"retry {attempt}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("RePORTER request failed 5 times")


def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in CODES:
        # RePORTER's offset stops at 14,999; split each code by award-notice month to stay well under it
        for month in [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            y = 2025 if month >= 10 else 2026
            start = f"{y}-{month:02d}-01"
            end_m, end_y = (month + 1, y) if month < 12 else (1, y + 1)
            end = f"{end_y}-{end_m:02d}-01"
            offset = 0
            while True:
                crit = {"fiscal_years": [2026], "activity_codes": [code], "award_types": ["1"],
                        "award_notice_date": {"from_date": start, "to_date": end}}  # fmt: skip
                d = post({"criteria": crit, "include_fields": FIELDS, "offset": offset, "limit": 500})
                res = d.get("results") or []
                rows += res
                offset += len(res)
                time.sleep(1.1)
                if len(res) < 500 or offset >= d["meta"]["total"]:
                    break
        print(f"{code}: {len(rows)} rows so far", flush=True)
    with RAW.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {RAW}")


HEADER = re.compile(
    r"^\s*(project\s+summary\s*/?\s*abstract|project\s+summary|summary\s*/?\s*abstract|abstract|summary|"
    r"description|overall|project\s+abstract)\s*[:.\-–]*\s*",
    re.IGNORECASE,
)
TAIL = re.compile(r"\s(project\s+narrative|public\s+health\s+relevance|narrative)\s*[:.]", re.IGNORECASE)


def clean(text: str) -> str:
    """RePORTER abstracts keep the PDF's hard line wraps: join wrapped lines, keep paragraph breaks, drop a leading
    'Project Summary/Abstract' header and anything from a trailing 'Project Narrative' on."""
    t = text.replace("\r", "").replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", t)]
    t = "\n\n".join(p for p in paras if p)
    for _ in range(2):
        t = HEADER.sub("", t, count=1).strip()
    m = TAIL.search(t)
    if m and m.start() > 0.5 * len(t):
        t = t[: m.start()].strip()
    return t


def shingles(text: str, n: int = 5) -> set[int]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {int(hashlib.md5(" ".join(w[i : i + n]).encode()).hexdigest()[:12], 16) for i in range(len(w) - n + 1)}


def minhash(sh: set[int], k: int = 64) -> tuple[int, ...]:
    if not sh:
        return (0,) * k
    return tuple(min((h * (2 * i + 1) + i) % (2**61 - 1) for h in sh) for i in range(k))


def build() -> None:
    rows = [json.loads(x) for x in RAW.read_text().splitlines() if x.strip()]
    card: dict = {"fetched_rows": len(rows), "drops": Counter()}
    seen_appl, kept = set(), []
    for r in rows:
        if r["appl_id"] in seen_appl:
            card["drops"]["duplicate appl_id"] += 1
            continue
        seen_appl.add(r["appl_id"])
        a = clean(r.get("abstract_text") or "")
        title = re.sub(r"\s+", " ", r.get("project_title") or "").strip()
        nw = len(a.split())
        if not a or not title:
            card["drops"]["no abstract or title"] += 1
        elif nw < MIN_WORDS:
            card["drops"][f"under {MIN_WORDS} words"] += 1
        elif nw > MAX_WORDS:
            card["drops"][f"over {MAX_WORDS} words"] += 1
        elif not r.get("award_notice_date"):
            card["drops"]["no award notice date"] += 1
        elif re.search(r"(?i)\bproject\s+(1|2|3|i|ii)\b\s*[:.]|\bcore\s+[a-d]\b\s*[:.]", a[:200]):
            card["drops"]["sub-project or core text"] += 1
        else:
            kept.append({"appl_id": r["appl_id"], "project_num": r["project_num"], "core": r["core_project_num"],
                         "activity": r["activity_code"], "ic": (r.get("agency_ic_admin") or {}).get("abbreviation"),
                         "notice": r["award_notice_date"][:10], "title": title, "abstract": a, "words": nw})  # fmt: skip
    # exact and near duplicates (the same abstract filed under two activity codes or two institutes)
    kept.sort(key=lambda x: (x["notice"], x["appl_id"]))
    by_title, buckets, final = set(), {}, []
    for x in kept:
        tkey = re.sub(r"[^a-z0-9]", "", x["title"].lower())
        if tkey in by_title:
            card["drops"]["duplicate title"] += 1
            continue
        sig = minhash(shingles(x["abstract"]))
        bands = [sig[i : i + 4] for i in range(0, 64, 4)]  # 16 bands of 4: flags pairs with Jaccard above ~0.5
        if any(b in buckets for b in bands):
            card["drops"]["near-duplicate abstract"] += 1
            continue
        for b in bands:
            buckets[b] = x["appl_id"]
        by_title.add(tkey)
        final.append(x)
    train_pool = [x for x in final if x["notice"] < TRAIN_BEFORE]
    held_pool = [x for x in final if x["notice"] >= HELDOUT_FROM]
    rng = random.Random(SEED)
    rng.shuffle(train_pool)
    rng.shuffle(held_pool)
    splits = {"dev": train_pool[:N_DEV], "train": train_pool[N_DEV:], "heldout": held_pool[:N_HELDOUT]}
    for name, xs in splits.items():
        with (CACHE / f"{name}.jsonl").open("w") as f:
            for x in xs:
                f.write(json.dumps(x) + "\n")
        (HERE / "data" / f"{name}_ids.txt").write_text("".join(f"{x['appl_id']}\n" for x in xs))
    card["drops"] = dict(card["drops"])
    card |= {
        "kept_after_dedupe": len(final),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "unused_heldout_pool": len(held_pool) - N_HELDOUT,
        "notice_range": {k: [min(x["notice"] for x in v), max(x["notice"] for x in v)] for k, v in splits.items()},
        "activity_codes": {k: dict(Counter(x["activity"] for x in v).most_common()) for k, v in splits.items()},
        "institutes_top10": {k: dict(Counter(x["ic"] for x in v).most_common(10)) for k, v in splits.items()},
        "words_median": {k: sorted(x["words"] for x in v)[len(v) // 2] for k, v in splits.items()},
        "criteria": {
            "fiscal_year": 2026,
            "award_type": "1 (new)",
            "activity_codes": CODES,
            "words": [MIN_WORDS, MAX_WORDS],
            "train_before": TRAIN_BEFORE,
            "heldout_from": HELDOUT_FROM,
            "seed": SEED,
            "source": URL,
            "fetched": time.strftime("%Y-%m-%d"),
        },
    }
    (HERE / "results" / "data_card.json").write_text(json.dumps(card, indent=1) + "\n")
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    {"fetch": fetch, "build": build}[sys.argv[1]]()
