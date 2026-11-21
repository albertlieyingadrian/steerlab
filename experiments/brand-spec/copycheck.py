"""The copy check: does an on-brand page copy one of the brand's existing pages, or is it new work?

    .venv/bin/python experiments/brand-spec/copycheck.py

Each candidate page is compared with every reference page of its brand and split; each component keeps its maximum
over references (and which reference gave it). Components, each scored alone first (STYLE: score each part):
  structure  Episode 2's structure repetition: cosine of Weisfeiler-Lehman subtree histograms of the element tree
             (label = role, or tag for containers; 3 rounds, 1,024 hashed bins, square-root damped)
  layout     ux-glue's fidelity idea, as layout: cosine of 3-channel occupancy grids (text, controls, filled boxes)
  style      ux-glue's computed-style diff: weighted Jaccard of text-element style signatures
             (role, font, size, weight, colour, background)
  text       word 5-gram Jaccard (near-copies here have their text replaced, so this should not find them)
  tokens     the cheap baseline: Jaccard of the colour and size tokens a page uses (on-brand pages share them)
The gate: copy if a gate score >= a threshold; the gate score (structure alone, min(structure, layout), or the mean of
structure, layout and style) and its threshold are both chosen on dev by accuracy, never on held-out. A page is
allowed when the rule check passes and the gate says new. Writes results/copy.json.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np
from checker import check, is_text
from common import BRANDS, DATA, RESULTS, load_spec, read_jsonl, write_json
from sklearn.metrics import roc_auc_score

WL_BINS, WL_ROUNDS = 1024, 3
GW, GH, PW, PH = 32, 60, 1280, 3600
COMPONENTS = ["structure", "layout", "style", "text", "tokens"]
GATES = {
    "gate_structure": lambda x: x["structure"],
    "gate_min_sl": lambda x: min(x["structure"], x["layout"]),
    "gate_mean_sls": lambda x: (x["structure"] + x["layout"] + x["style"]) / 3,
}


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def wl(els: list[dict]) -> np.ndarray:
    ix = {e["id"]: i for i, e in enumerate(els)}
    kids: list[list[int]] = [[] for _ in els]
    for i, e in enumerate(els):
        if e["parent"] in ix:
            kids[ix[e["parent"]]].append(i)
    labels = [e["role"] if e["role"] != "container" else f"c:{e['tag']}:{int(bool(e['bg']))}" for e in els]
    vec = np.zeros(WL_BINS, np.float32)
    for lab in labels:
        vec[_h(lab) % WL_BINS] += 1
    for _ in range(WL_ROUNDS):
        labels = [
            str(_h(labels[i] + "(" + ",".join(sorted(labels[k] for k in kids[i])) + ")")) for i in range(len(els))
        ]
        for lab in labels:
            vec[int(lab) % WL_BINS] += 1
    return np.sqrt(vec)


def occupancy(els: list[dict]) -> np.ndarray:
    fine = np.zeros((3, GH * 2, GW * 2), np.float32)
    sx, sy = GW * 2 / PW, GH * 2 / PH
    for e in els:
        ch = 1 if e["role"].startswith("button") or e["role"] in ("input", "link") else 0 if is_text(e) else 2
        if ch == 2 and not e["bg"] and not e["border"]:
            continue
        x0, x1 = int(max(0, e["x"] * sx)), int(min(GW * 2, np.ceil((e["x"] + e["w"]) * sx)))
        y0, y1 = int(max(0, e["y"] * sy)), int(min(GH * 2, np.ceil((e["y"] + e["h"]) * sy)))
        if x1 > x0 and y1 > y0:
            fine[ch, y0:y1, x0:x1] = 1.0
    return fine.reshape(3, GH, 2, GW, 2).mean((2, 4)).reshape(-1)


def style_sig(els: list[dict]) -> Counter:
    return Counter((e["role"], e["font"], e["size"], e["weight"], e["color"], e["eff_bg"]) for e in els if is_text(e))


def grams(els: list[dict]) -> set:
    words = re.findall(r"[a-z0-9']+", " ".join(e["text"].lower() for e in els if e["text"]))
    return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}


def tokens(els: list[dict]) -> set:
    return {("c", e[k]) for e in els for k in ("color", "bg", "border") if e.get(k)} | {
        ("s", e["size"]) for e in els if is_text(e)
    }


def cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def jac(a, b) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def wjac(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    den = sum(max(a[k], b[k]) for k in keys)
    return sum(min(a[k], b[k]) for k in keys) / den if den else 0.0


def feats(f: dict) -> dict:
    els = f["elements"]
    return {
        "structure": wl(els),
        "layout": occupancy(els),
        "style": style_sig(els),
        "text": grams(els),
        "tokens": tokens(els),
    }


def sim(a: dict, b: dict) -> dict:
    return {
        "structure": cos(a["structure"], b["structure"]),
        "layout": cos(a["layout"], b["layout"]),
        "style": wjac(a["style"], b["style"]),
        "text": jac(a["text"], b["text"]),
        "tokens": jac(a["tokens"], b["tokens"]),
    }


def score_split(brand: str, split: str) -> list[dict]:
    pages = read_jsonl(DATA / f"copy_{brand}_{split}.jsonl")
    facts = read_jsonl(DATA / f"facts_copy_{brand}_{split}.jsonl")
    spec = load_spec(brand)
    F = {f["id"]: feats(f) for f in facts}
    refs = [p["id"] for p in pages if p["kind"] == "ref"]
    rows = []
    for p, f in zip(pages, facts, strict=True):
        if p["kind"] == "ref":
            continue
        sims = {r: sim(F[p["id"]], F[r]) for r in refs}
        row = {
            "id": p["id"],
            "brand": brand,
            "split": split,
            "kind": p["kind"],
            "ref": p.get("ref"),
            "strength": p.get("strength"),
            "edit": p.get("edit"),
            "rule_check": check(spec, f)["verdict"],
        }
        for c in COMPONENTS:
            best = max(refs, key=lambda r, c=c: sims[r][c])
            row[c], row[f"{c}_ref"] = round(sims[best][c], 4), best
        for g, fn in GATES.items():
            best = max(refs, key=lambda r, fn=fn: fn(sims[r]))
            row[g], row[f"{g}_ref"] = round(fn(sims[best]), 4), best
        rows.append(row)
    return rows


def auroc(rows, key, pos="near", neg="new", B=2000, seed=0):
    sel = [r for r in rows if r["kind"] in (pos, neg)]
    y = np.array([r["kind"] == pos for r in sel])
    s = np.array([r[key] for r in sel])
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y)[0], np.where(~y)[0]
    boots = [
        roc_auc_score(y[idx], s[idx])
        for idx in (np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]) for _ in range(B))
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "auroc": round(float(roc_auc_score(y, s)), 3),
        "ci": [round(float(lo), 3), round(float(hi), 3)],
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
    }


def best_threshold(rows, key) -> tuple[float, float]:
    sel = [r for r in rows if r["kind"] in ("near", "new")]
    best, best_acc = 1.0, -1.0
    for t in sorted({r[key] for r in sel}):
        acc = float(np.mean([(r[key] >= t) == (r["kind"] == "near") for r in sel]))
        if acc > best_acc + 1e-12:
            best, best_acc = t, acc
    return float(best), best_acc


def main() -> None:
    out: dict = {"components": COMPONENTS, "brands": {}}
    scored = {b: (score_split(b, "dev"), score_split(b, "heldout")) for b in BRANDS}
    fits_all = {b: {g: best_threshold(scored[b][0], g) for g in GATES} for b in BRANDS}
    # one gate for both brands, chosen by mean dev accuracy (ties: structure alone); thresholds per brand from dev
    gate = max(GATES, key=lambda g: (np.mean([fits_all[b][g][1] for b in BRANDS]), g == "gate_structure"))
    out["gate"] = gate
    for brand in BRANDS:
        dev, held = scored[brand]
        fits = fits_all[brand]
        t = fits[gate][0]
        for r in dev + held:
            r["gate"], r["gate_ref"] = r[gate], r[f"{gate}_ref"]
        res: dict = {
            "gate": gate,
            "dev_fits": fits,
            "threshold_from_dev": t,
            "dev": {},
            "heldout": {},
            "rows": dev + held,
        }
        for name, rows in (("dev", dev), ("heldout", held)):
            for key in [*COMPONENTS, *GATES, "gate"]:
                res[name][key] = auroc(rows, key)
            sel = [r for r in rows if r["kind"] in ("near", "new")]
            near = [r for r in sel if r["kind"] == "near"]
            res[name]["gate_accuracy"] = [sum((r["gate"] >= t) == (r["kind"] == "near") for r in sel), len(sel)]
            res[name]["near_caught_by_strength"] = {
                s: [sum(r["gate"] >= t for r in near if r["strength"] == s), sum(r["strength"] == s for r in near)]
                for s in (1, 2, 3)
            }
            res[name]["new_allowed"] = [
                sum(r["gate"] < t and r["rule_check"] == "pass" for r in sel if r["kind"] == "new"),
                sum(r["kind"] == "new" for r in sel),
            ]
            res[name]["flagged_right_ref"] = [
                sum(r["gate_ref"] == r["ref"] for r in near if r["gate"] >= t),
                sum(r["gate"] >= t for r in near),
            ]
            remix = [r for r in rows if r["kind"] == "remix"]
            if remix:
                res[name]["remix_flagged"] = [sum(r["gate"] >= t for r in remix), len(remix)]
            res[name]["rule_check_pass"] = [sum(r["rule_check"] == "pass" for r in rows), len(rows)]
        out["brands"][brand] = res
        h = res["heldout"]
        print(
            brand,
            gate,
            fits,
            "threshold",
            round(t, 3),
            {k: h[k]["auroc"] for k in [*COMPONENTS, "gate"]},
            "gate acc",
            h["gate_accuracy"],
            "by strength",
            h["near_caught_by_strength"],
            "remix",
            h.get("remix_flagged"),
            "dev",
            {k: res["dev"][k]["auroc"] for k in [*COMPONENTS, "gate"]},
        )
    write_json(RESULTS / "copy.json", out)


if __name__ == "__main__":
    main()
