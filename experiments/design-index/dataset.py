"""
Load the index: join every rendered page's features (CACHE/feats/ours.pkl) with what it is (source, category, year,
split), and apply the render filter (the render check) whose thresholds live in results/thresholds.json (tuned on dev only).

Sources:
  wayback    our Wayback panel (year, stable, all_sample from CACHE/panel.jsonl)
  human      live render of a human seed site            ai_host    live render of a site on an AI-builder host
  human_rep  a second live render of a human site (render-noise floor)
  template   the templated known-answer pages            generated  pages written by open code models
  curated    live render of a site from a curated design-reference library (a small high-quality human class)
"""

from __future__ import annotations

import json
import pickle

from common import CACHE, RESULTS, split_of

THRESHOLDS = RESULTS / "thresholds.json"
DEFAULT_THRESHOLDS = {
    "unstyled_max": 0.6,
    "fail_rate_max": 0.5,
    "min_words": 15,
    "ai_min_words": 30,
    "ai_min_blocks": 10,
}


def _jobs(name):
    p = CACHE / name
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def load_meta() -> dict:
    seeds = {}
    for line in (CACHE / "seeds.jsonl").read_text().splitlines():
        r = json.loads(line)
        seeds[r["domain"]] = r
    panel = {}
    if (CACHE / "panel.jsonl").exists():
        for line in (CACHE / "panel.jsonl").read_text().splitlines():
            r = json.loads(line)
            panel[r["domain"]] = r
    meta = {}
    for j in _jobs("jobs_live.jsonl"):
        s = seeds[j["domain"]]
        meta[j["pid"]] = {
            "source": "human" if s["cls"] == "human" else "ai_host",
            "domain": j["domain"],
            "category": s["category"],
            "year": 2026,
            "split": split_of(j["domain"]),
        }
    cur = {c["domain"]: c["category"] for c in _jobs("curated_sites.jsonl")}
    for j in _jobs("jobs_curated.jsonl"):
        meta[j["pid"]] = {
            "source": "curated",
            "domain": j["domain"],
            "category": cur[j["domain"]],
            "year": 2026,
            "split": "held",
        }  # a small evaluation-only class: never used for tuning
    for j in _jobs("jobs_live2.jsonl"):
        s = seeds[j["domain"]]
        meta[j["pid"]] = {
            "source": "human_rep",
            "domain": j["domain"],
            "category": s["category"],
            "year": 2026,
            "split": split_of(j["domain"]),
        }
    for j in _jobs("jobs_wayback.jsonl"):
        p = panel.get(j["domain"], {})
        meta[j["pid"]] = {
            "source": "wayback",
            "domain": j["domain"],
            "category": seeds[j["domain"]]["category"],
            "year": j["year"],
            "stable": p.get("stable", False),
            "all_sample": p.get("all_sample", False),
            "split": split_of(j["domain"]),
        }
    for j in _jobs("jobs_tpl.jsonl"):
        meta[j["pid"]] = {
            "source": "template",
            "domain": j["domain"],
            "category": j["category"],
            "year": 2026,
            "template": j["template"],
            "split": "dev" if j["template"] in {"A", "B"} else "held",
        }
    finished = {g["file"]: g["finished"] != "length" for g in _jobs("gen_meta.jsonl")}
    for j in _jobs("jobs_gen.jsonl"):
        meta[j["pid"]] = {
            "source": "generated",
            "domain": j["domain"],
            "category": j["category"],
            "year": 2026,
            "model": j["model"],
            "split": split_of(j["domain"]),
            "finished": finished.get(j["domain"], True),
        }
    return meta


def thresholds() -> dict:
    if THRESHOLDS.exists():
        return {**DEFAULT_THRESHOLDS, **json.loads(THRESHOLDS.read_text())}
    return dict(DEFAULT_THRESHOLDS)


def valid(r: dict, th: dict) -> tuple[bool, str]:
    """The render filter. Returns (keep, reason)."""
    if r.get("siglip") is None:
        return False, "no_embedding"
    if r["error"] or r.get("status") == 202:  # 202 = a host's "checking the connection" bot wall (dev, results.tsv)
        return False, "error_page"
    if r["blank"] or r["n_words"] < th["min_words"] or not float(r["layout"].astype("float32").sum()):
        return False, "blank"
    if r["source"] == "wayback" and r["unstyled_score"] > th["unstyled_max"]:
        return False, "unstyled"
    if r["source"] == "wayback" and r["fail_rate"] > th["fail_rate_max"]:
        return False, "assets_failed"
    if r["source"] == "generated" and not r.get("finished", True):
        return (
            False,
            "cut_off",
        )  # pages cut off at the token limit render without most of their CSS (run log 11)
    if r["source"] == "ai_host" and (r["n_words"] < th["ai_min_words"] or r["n_blocks"] < th["ai_min_blocks"]):
        return False, "ai_draft"
    return True, "ok"


def load(apply_filter: bool = True) -> list[dict]:
    feats = pickle.loads((CACHE / "feats" / "ours.pkl").read_bytes())
    meta = load_meta()
    th = thresholds()
    out = []
    for pid, f in feats.items():
        if pid not in meta:
            continue
        r = {**f, **meta[pid]}
        r["keep"], r["why"] = valid(r, th)
        if r["keep"] or not apply_filter:
            out.append(r)
    return out
