"""
Hand-label two rulers on dev pages: the render check (render filter) and the label check (category labels).

  sheet r1   60 random dev Wayback renders (seed 0) -> numbered contact sheets for a person to label ok/broken
  sheet r5   60 random dev live human renders -> numbered sheets captioned with their Exa category
  score r1   compare results/labels_r1.json ({"<pid>": "ok"|"broken"}) with the automatic filter
  score r5   share of results/labels_r5.json ({"<pid>": true|false}, "is the site in its category?") marked true
Sheets go to CACHE/sheets/ (never committed: they show third-party pages). Needs shots in CACHE/vol/shots.
"""

from __future__ import annotations

import json
import random
import sys

from common import CACHE, RESULTS
from dataset import load
from PIL import Image, ImageDraw

N = 60


def pick(which):
    recs = load(apply_filter=False)
    if which == "r1":
        pool = [r for r in recs if r["source"] == "wayback" and r["split"] == "dev"]
    else:
        pool = [
            r
            for r in recs
            if r["source"] == "human" and r["split"] == "dev" and r["keep"]
        ]
    pool.sort(key=lambda r: r["pid"])
    return random.Random(0).sample(pool, min(N, len(pool)))


def sheet(which):
    rs = pick(which)
    out = CACHE / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(rs), 12):
        W = Image.new("RGB", (4 * 480, 3 * 330), "white")
        d = ImageDraw.Draw(W)
        for k, r in enumerate(rs[s : s + 12]):
            im = Image.open(CACHE / "vol" / "shots" / f"{r['pid']}.jpg").resize(
                (480, 300)
            )
            x, y = (k % 4) * 480, (k // 4) * 330
            W.paste(im, (x, y + 30))
            cap = f"{s + k}: {r['pid'][:8]}" + (
                f" {r['year']}" if which == "r1" else f" [{r['category']}]"
            )
            d.text((x + 6, y + 8), cap, fill="black")
        W.save(out / f"{which}_{s // 12}.jpg", quality=80)
    (out / f"{which}_index.json").write_text(json.dumps([r["pid"] for r in rs]))
    print(f"wrote {len(rs)} pages to {out}/{which}_*.jpg")


def score(which):
    labels = json.loads((RESULTS / f"labels_{which}.json").read_text())
    if which == "r5":
        n = len(labels)
        k = sum(bool(v) for v in labels.values())
        res = {"n": n, "agree": k, "rate": k / n, "pass": k / n >= 0.85}
    else:
        recs = {r["pid"]: r for r in load(apply_filter=False)}
        agree = sum((labels[p] == "ok") == recs[p]["keep"] for p in labels if p in recs)
        n = sum(p in recs for p in labels)
        confusion = {}
        for p, lab in labels.items():
            if p in recs:
                key = f"human_{lab}/filter_{'keep' if recs[p]['keep'] else 'drop'}"
                confusion[key] = confusion.get(key, 0) + 1
        res = {
            "n": n,
            "agree": agree,
            "rate": agree / n,
            "pass": agree / n >= 0.90,
            "confusion": confusion,
        }
    print(json.dumps(res, indent=1))
    (RESULTS / f"ruler_{which}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"sheet": sheet, "score": score}[sys.argv[1]](sys.argv[2])
