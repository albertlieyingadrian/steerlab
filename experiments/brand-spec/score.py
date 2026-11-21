"""Score every reply, add the baselines, and write results/summary.json.

    .venv/bin/python experiments/brand-spec/score.py [--split dev|heldout]

T1 (page audits). A claim is (element, line). It matches strictly when the element has a planted failure whose rule
cites that line (`lines`), loosely when the line is one the rule reads (`support`), by element only otherwise; a
claim on an element with no failure is a false claim. A page is audited exactly when its strict matches equal its
planted pairs and it has no other claims ([] on a compliant page). Arm a (no rules) cannot cite lines, so it is
scored on verdict and on the element set.
T2: strict = the cited line is one of the deciding rule's lines; loose = one of the lines it reads.
T3: the yes/no verdict.
Every rate is k/n with a 95% bootstrap interval over items (10,000 resamples); differences are paired where the items
are shared. Writes results/scored/<raw name>.jsonl (per item) and results/summary.json (summary_dev.json for dev).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from checker import check
from common import BRANDS, DATA, RESULTS, load_spec, read_jsonl, write_json, write_jsonl
from parse import parse_t1, parse_t2, parse_t3

B = 10_000
MODELS = ["qwen3-8b", "qwen3-32b"]


def ci(xs, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = xs[rng.integers(0, len(xs), (B, len(xs)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "k": round(float(xs.sum()), 3),
        "n": len(xs),
        "rate": round(float(xs.mean()), 3),
        "ci": [round(float(lo), 3), round(float(hi), 3)],
    }


def paired(a, b, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile(d[rng.integers(0, len(d), (B, len(d)))].mean(1), [2.5, 97.5])
    return {"diff": round(float(d.mean()), 3), "ci": [round(float(lo), 3), round(float(hi), 3)], "n": len(d)}


def unpaired(a, b, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return {
        "diff": round(float(a.mean() - b.mean()), 3),
        "ci": [round(float(lo), 3), round(float(hi), 3)],
        "n": [len(a), len(b)],
    }


def raw_path(task, brand, split, model, arm, seed=0) -> Path:
    return RESULTS / "raw" / f"{task}_{brand}_{split}_{model}_{arm}{f'_s{seed}' if seed else ''}.jsonl"


def load_replies(path: Path) -> dict | None:
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else None


# ---------------------------------------------------------------- T1
def score_t1(items, replies, spec, arm):
    rules = {r["id"]: r for r in spec["rules"]}
    rows = []
    for it in items:
        rep = replies.get(it["id"])
        gold = {(e, r) for e, r in it["labels"]}
        gold_el: dict[str, list[str]] = {}
        for e, r in gold:
            gold_el.setdefault(e, []).append(r)
        claims = parse_t1(rep["text"]) if rep else None
        row = {
            "id": it["id"],
            "kind": it["kind"],
            "gold": sorted(gold),
            "parsed": claims is not None,
            "cut": bool(rep and rep["finish"] == "length"),
            "tokens": rep["tokens"] if rep else None,
        }
        if claims is None:
            row.update(exact=0, verdict=0, el_exact=0, claims=None, strict_hits=0, el_hits=0, missed=sorted(gold))
            rows.append(row)
            continue
        claims = sorted(set(claims))
        pred_el = {e for e, _ in claims}
        row.update(
            claims=claims,
            verdict=int(bool(claims) == bool(gold)),
            el_exact=int(pred_el == set(gold_el)),
            false_claims=sum(e not in gold_el for e, _ in claims),
            el_hits=len(pred_el & set(gold_el)),
        )
        if arm == "b":
            strict, loose, other = set(), set(), 0
            for e, ln in claims:
                hit = next((r for r in gold_el.get(e, []) if ln in rules[r]["lines"]), None)
                if hit:
                    strict.add((e, hit))
                    continue
                if any(ln in rules[r]["lines"] + rules[r]["support"] for r in gold_el.get(e, [])):
                    loose.add(e)
                other += 1
            row.update(
                strict_hits=len(strict),
                loose_hits=len(strict) + len(loose),
                exact=int(strict == gold and other == 0),
                missed=sorted(gold - strict),
            )
        rows.append(row)
    return rows


def t1_summary(rows, items, arm):
    by_id = {it["id"]: it for it in items}
    off = [r for r in rows if r["gold"]]
    clean = [r for r in rows if not r["gold"]]
    flagged = [bool(r["claims"]) if r["parsed"] else True for r in clean]
    out = {
        "pages": len(rows),
        "parsed": ci([r["parsed"] for r in rows]),
        "cut": sum(r["cut"] for r in rows),
        "mean_tokens": round(float(np.mean([r["tokens"] for r in rows if r["tokens"]])), 0),
        "verdict": ci([r["verdict"] for r in rows]),
        "el_exact": ci([r["el_exact"] for r in rows]),
        "el_exact_off": ci([r["el_exact"] for r in off]),
        "false_alarm_clean": ci(flagged),
        "false_alarm_traps": ci([f for f, r in zip(flagged, clean, strict=True) if r["kind"] == "trap"]),
        "el_recall": ci([r["el_hits"] / len({e for e, _ in r["gold"]}) for r in off]),
    }
    if arm == "b":
        out["exact"] = ci([r["exact"] for r in rows])
        out["exact_off"] = ci([r["exact"] for r in off])
        out["exact_clean"] = ci([r["exact"] for r in clean])
        out["pair_recall"] = ci([r["strict_hits"] / len(r["gold"]) for r in off])
        per: dict[str, list[int]] = {"text": [], "decided": []}
        per_rule: dict[str, list[int]] = {}
        for r in off:
            got = {tuple(x) for x in r["gold"]} - {tuple(x) for x in r["missed"]}
            for p in by_id[r["id"]]["planted"]:
                hit = int((p["element"], p["rule"]) in got)
                per[p["settled"]].append(hit)
                per_rule.setdefault(p["rule"], []).append(hit)
        out["planted_recall_text"] = ci(per["text"])
        out["planted_recall_decided"] = ci(per["decided"])
        out["planted_recall_by_rule"] = {k: [sum(v), len(v)] for k, v in sorted(per_rule.items())}
        out["rows"] = {r["id"]: r["exact"] for r in rows}
    return out


def t1_baselines(items, facts, spec):
    """Do-nothing and cheap baselines on the same pages, and the checker itself (arm c)."""
    F = {f["id"]: f for f in facts}
    gold = [{tuple(x) for x in it["labels"]} for it in items]
    full = [{tuple(x) for x in check(spec, F[it["id"]])["fail_pairs"]} for it in items]
    pal = [{x for x in c if x[1].endswith("-PAL")} for c in full]  # a checker that knows only the palette
    return {
        "always_compliant_exact": ci([int(not g) for g in gold]),
        "palette_only_exact": ci([int(p == g) for p, g in zip(pal, gold, strict=True)]),
        "palette_only_verdict": ci([int(bool(p) == bool(g)) for p, g in zip(pal, gold, strict=True)]),
        "checker_exact": ci([int(c == g) for c, g in zip(full, gold, strict=True)]),
    }


# ---------------------------------------------------------------- T2 and T3
def score_t2(items, replies, spec):
    page_of = {s["id"]: s["page"] for s in spec["source"]}
    rows = []
    for it in items:
        rep = replies.get(it["id"])
        got = parse_t2(rep["text"]) if rep else None
        same_page = [s["id"] for s in spec["source"] if s["page"] == page_of[it["strict"][0]]]
        rows.append(
            {
                "id": it["id"],
                "rule": it["rule"],
                "exception": it["exception"],
                "settled": it["settled"],
                "property": it["property"],
                "cited": got,
                "strict": int(got in it["strict"]),
                "loose": int(got in it["loose"]),
                "parsed": got is not None,
                "cut": bool(rep and rep["finish"] == "length"),
                "random_citation": len(set(it["strict"]) & set(same_page)) / len(same_page),
            }
        )
    return rows


def score_t3(items, replies):
    rows = []
    for it in items:
        rep = replies.get(it["id"])
        got = parse_t3(rep["text"]) if rep else None
        rows.append(
            {
                "id": it["id"],
                "kind": it["kind"],
                "rule": it["rule"],
                "answer": it["answer"],
                "pred": got,
                "correct": int(got == it["answer"]),
                "parsed": got is not None,
                "cut": bool(rep and rep["finish"] == "length"),
            }
        )
    return rows


def t2_summary(rows):
    return {
        "strict": ci([r["strict"] for r in rows]),
        "loose": ci([r["loose"] for r in rows]),
        "exception": ci([r["strict"] for r in rows if r["exception"]]),
        "plain": ci([r["strict"] for r in rows if not r["exception"]]),
        "text_settled": ci([r["strict"] for r in rows if r["settled"] == "text"]),
        "decided": ci([r["strict"] for r in rows if r["settled"] == "decided"]),
        "random_citation": round(float(np.mean([r["random_citation"] for r in rows])), 3),
        "parsed": ci([r["parsed"] for r in rows]),
        "cut": sum(r["cut"] for r in rows),
        "by_rule": {
            k: [sum(r["strict"] for r in rows if r["rule"] == k), sum(r["rule"] == k for r in rows)]
            for k in sorted({r["rule"] for r in rows})
        },
    }


def t3_summary(rows):
    kinds = sorted({r["kind"] for r in rows})
    return {
        "all": ci([r["correct"] for r in rows]),
        "parsed": ci([r["parsed"] for r in rows]),
        "cut": sum(r["cut"] for r in rows),
        "yes_rate": round(float(np.mean([r["pred"] == "yes" for r in rows])), 3),
        "by_kind": {k: ci([r["correct"] for r in rows if r["kind"] == k]) for k in kinds},
        "rows": {r["id"]: r["correct"] for r in rows},
    }


def kind_rows(brand, split, kind, rows) -> list[int]:
    ids = {x["id"] for x in read_jsonl(DATA / f"t3_{brand}_{split}.jsonl") if x["kind"] == kind}
    return [v for k, v in rows.items() if k in ids]


def score_model(model: str, split: str) -> dict:
    M: dict = {}
    pooled: dict[str, list] = {"t2": [], "t3": [], "t1b_off": []}
    for brand in BRANDS:
        spec = load_spec(brand)
        BR: dict = {}
        t1 = read_jsonl(DATA / f"t1_{brand}_{split}.jsonl")
        BR["t1_baselines"] = t1_baselines(t1, read_jsonl(DATA / f"facts_pages_{brand}_{split}.jsonl"), spec)
        for arm, seed in (("a", 0), ("b", 0), ("b", 1)):
            p = raw_path("t1", brand, split, model, arm, seed)
            reps = load_replies(p)
            if reps is None:
                continue
            rows = score_t1(t1, reps, spec, arm)
            write_jsonl(RESULTS / "scored" / p.name, rows)
            key = f"t1_{arm}" + (f"_s{seed}" if seed else "")
            BR[key] = t1_summary(rows, t1, arm)
            if arm == "b" and not seed:
                pooled["t1b_off"] += [r["exact"] for r in rows if r["gold"]]
        for task in ("t2", "t3", "t3c"):
            p = raw_path(task, brand, split, model, "b")
            reps = load_replies(p)
            if reps is None:
                continue
            items = read_jsonl(DATA / f"{task}_{brand}_{split}.jsonl")
            rows = score_t2(items, reps, spec) if task == "t2" else score_t3(items, reps)
            write_jsonl(RESULTS / "scored" / p.name, rows)
            BR[task] = t2_summary(rows) if task == "t2" else t3_summary(rows)
            if task in pooled:
                pooled[task] += [r["strict" if task == "t2" else "correct"] for r in rows]
        if len(BR) > 1:
            M[brand] = BR
    P: dict = {}
    if pooled["t1b_off"]:
        P["P1_exact_off_brand"] = ci(pooled["t1b_off"])
    if pooled["t2"] and pooled["t3"]:
        P["P2_t3_minus_t2"] = unpaired(pooled["t3"], pooled["t2"])
        P["t2_strict_pooled"], P["t3_pooled"] = ci(pooled["t2"]), ci(pooled["t3"])
    g, u = M.get("govuk", {}).get("t3"), M.get("uswds", {}).get("t3")
    if g and u:
        gc = kind_rows("govuk", split, "contrast", g["rows"])
        um = kind_rows("uswds", split, "magic", u["rows"])
        uc = kind_rows("uswds", split, "contrast", u["rows"])
        P["P3_magic_minus_govuk_contrast"] = unpaired(um, gc)
        P["uswds_magic_minus_uswds_contrast"] = unpaired(um, uc)
    for brand, br in M.items():
        if "t3" in br and "t3c" in br:
            items = read_jsonl(DATA / f"t3_{brand}_{split}.jsonl")
            pairs = []
            for c in read_jsonl(DATA / f"t3c_{brand}_{split}.jsonl"):
                twin = next(x["id"] for x in items if x["kind"] == "contrast" and x["element"] == c["element"])
                pairs.append((br["t3c"]["rows"][c["id"]], br["t3"]["rows"][twin]))
            P[f"{brand}_ratio_given_minus_not"] = paired([a for a, _ in pairs], [b for _, b in pairs])
        if "t1_b" in br and "t1_b_s1" in br:
            ids = list(br["t1_b"]["rows"])
            P[f"{brand}_t1b_seed1_minus_seed0"] = paired(
                [br["t1_b_s1"]["rows"][k] for k in ids], [br["t1_b"]["rows"][k] for k in ids]
            )
    # post hoc (code review after the held-out run): the type-scale items show sizes on body elements with no line
    # height, and two use 27 px (exceptional); report T3 and P2 without that kind beside the registered numbers
    t3x, t2x = [], []
    for brand in M:
        p3, p2 = raw_path("t3", brand, split, model, "b"), raw_path("t2", brand, split, model, "b")
        if (RESULTS / "scored" / p3.name).exists() and (RESULTS / "scored" / p2.name).exists():
            t3x += [r["correct"] for r in read_jsonl(RESULTS / "scored" / p3.name) if r["kind"] != "type scale"]
            t2x += [r["strict"] for r in read_jsonl(RESULTS / "scored" / p2.name)]
    if t3x and t2x:
        P["post_hoc_t3_without_type_scale"] = ci(t3x)
        P["post_hoc_P2_without_type_scale"] = unpaired(t3x, t2x)
    for br in M.values():
        for key in ("t1_b", "t1_b_s1", "t3", "t3c"):
            br.get(key, {}).pop("rows", None)
    if M:
        M["comparisons"] = P
    return M


def main() -> None:
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "heldout"
    summary: dict = {"split": split, "models": {}}
    for model in MODELS:
        M = score_model(model, split)
        if M:
            summary["models"][model] = M
    spend = RESULTS / "spend.jsonl"
    summary["modal_spend"] = (
        round(sum(json.loads(x)["dollars"] for x in spend.read_text().splitlines() if x.strip()), 2)
        if spend.exists()
        else 0.0
    )
    write_json(RESULTS / f"summary{'_dev' if split == 'dev' else ''}.json", summary)
    for m, M in summary["models"].items():
        for b in BRANDS:
            if b not in M:
                continue
            br = M[b]
            print(
                m,
                b,
                "t1a",
                {k: br["t1_a"][k]["rate"] for k in ("verdict", "el_exact", "false_alarm_clean")}
                if "t1_a" in br
                else None,
            )
            print(
                m,
                b,
                "t1b",
                {
                    k: br["t1_b"][k]["rate"] if isinstance(br["t1_b"][k], dict) else br["t1_b"][k]
                    for k in ("exact", "exact_off", "verdict", "false_alarm_clean", "pair_recall", "cut")
                }
                if "t1_b" in br
                else None,
            )
            print(
                m,
                b,
                "t2",
                {k: br["t2"][k]["rate"] for k in ("strict", "loose", "exception", "plain")},
                "random",
                br["t2"]["random_citation"],
            ) if "t2" in br else None
            print(m, b, "t3", br["t3"]["all"]["rate"], {k: v["rate"] for k, v in br["t3"]["by_kind"].items()})
            print(m, b, "t3c", br["t3c"]["all"]["rate"]) if "t3c" in br else None
            print(m, b, "baselines", {k: v["rate"] for k, v in br["t1_baselines"].items()})
        print(m, json.dumps(M["comparisons"]))
    print("modal spend", summary["modal_spend"])


if __name__ == "__main__":
    main()
