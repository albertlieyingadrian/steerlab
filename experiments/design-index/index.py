"""
Build the searchable design index: FAISS (inner product on unit SigLIP vectors) plus a metadata table, over every
kept page we rendered and every WaybackUI screenshot. A query returns the nearest pages with their source, year,
category and palette; the note's nearest-neighbour explorer is a static export of this.

Run: .venv/bin/python experiments/design-index/index.py [--query-pid PID]
Writes CACHE/index/siglip.faiss and CACHE/index/meta.jsonl (cache only: rows name third-party domains), and
results/index_summary.json (counts by source and year, which ship).
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter

import numpy as np

from common import CACHE, RESULTS
from dataset import load


def build():
    import faiss

    ours = load()
    rows, vecs = [], []
    for r in ours:
        rows.append(
            {
                "id": r["pid"],
                "source": r["source"],
                "year": r["year"],
                "category": r["category"],
                "domain": r["domain"],
            }
        )
        vecs.append(np.asarray(r["siglip"], np.float32))
    wb_path = CACHE / "feats" / "waybackui.pkl"
    if wb_path.exists():
        for i, r in enumerate(pickle.loads(wb_path.read_bytes())):
            rows.append(
                {
                    "id": f"wbui{i}",
                    "source": "waybackui",
                    "year": r["year"],
                    "category": None,
                    "domain": r["base_url"],
                    "res": r["res"],
                }
            )
            vecs.append(np.asarray(r["siglip"], np.float32))
    X = np.stack(vecs)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    out = CACHE / "index"
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "siglip.faiss"))
    (out / "meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "n_pages": len(rows),
        "dim": int(X.shape[1]),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_source_year": {
            f"{s}/{y}": n for (s, y), n in sorted(Counter((r["source"], r["year"]) for r in rows).items())
        },
    }
    (RESULTS / "index_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["by_source"]), summary["n_pages"], "pages")
    return index, rows, X


def query(pid: str, k: int = 8):
    import faiss

    index = faiss.read_index(str(CACHE / "index" / "siglip.faiss"))
    rows = [json.loads(line) for line in (CACHE / "index" / "meta.jsonl").read_text().splitlines()]
    i = next(j for j, r in enumerate(rows) if r["id"] == pid)
    v = index.reconstruct(i)[None, :]
    sims, ids = index.search(v, k + 1)
    for s, j in zip(sims[0], ids[0]):
        print(f"{s:.3f}", rows[j])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-pid")
    a = ap.parse_args()
    query(a.query_pid) if a.query_pid else build()
