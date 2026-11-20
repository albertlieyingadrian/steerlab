"""Score every raw reply file, run the checker arm (c) on arm b's traces, and write the summary.

    .venv/bin/python experiments/formalise-demo/score.py

Reads results/raw/*.jsonl (pilot files are skipped) and data/*.jsonl. Writes results/scored/<same name>.jsonl (one row
per question: the parse, the score, and for arm b the checker's verification of the trace, i.e. the proof artifact)
and results/summary.json (every rate as k/n with a 95% bootstrap interval over questions, paired differences, the
first-failure breakdown, and the baselines). All numbers in the note come from summary.json.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter

import contrast
from checker import evaluate, fact_diffs, facts_of, line_value, verify
from common import DATA, RESULTS, read_jsonl, write_jsonl
from parse import parse

B = 10_000
RAW = RESULTS / "raw"
SCORED = RESULTS / "scored"
NAME = re.compile(
    r"(?P<task>baggage|contrast)_(?P<split>dev|heldout)_(?P<model>[a-z0-9-]+)_(?P<arm>a|b0|b)(?:_s(?P<seed>\d+))?$"
)


def boot(values: list[float], seed: int = 0) -> list[float]:
    """95% percentile bootstrap interval of the mean."""
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(means[int(0.025 * B)], 4), round(means[int(0.975 * B) - 1], 4)]


def rate(flags: list[bool]) -> dict:
    k, n = sum(flags), len(flags)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci": boot([float(f) for f in flags])}


def paired(x: dict[str, bool], y: dict[str, bool]) -> dict:
    """x - y over the questions both arms answered (keyed by id)."""
    ids = sorted(set(x) & set(y))
    d = [float(x[i]) - float(y[i]) for i in ids]
    return {"n": len(ids), "diff": round(sum(d) / len(d), 4) if d else None, "ci": boot(d, seed=1)}


# ----------------------------------------------------------------------------- baggage


def near(line: str | None, rules: list[str]) -> bool:
    """Loose identification (dev try 4): the cited line is within two lines of an expected one, the slip of citing
    the dimension line beside the weight line, or a section's intro line instead of its first rule."""
    if not line:
        return False
    return any(r.startswith("L") and abs(int(r[1:]) - int(line[1:])) <= 2 for r in rules)


def identification(trace: dict, info: dict) -> list[dict]:
    """Per checked bag and component: did the cited line match the spec's lines under the true facts?"""
    truth = {(s["step"], s.get("item")): s for s in evaluate(info)["steps"]}
    cited = {b["item"]: b for b in trace["bags"]}
    out = []
    for (step, item), s in truth.items():
        if step not in ("base", "weight", "size"):
            continue
        rules = [r for r in s["rules"] if r != "L270"]
        line = cited.get(item, {}).get(f"{step}_line")
        out.append(
            {
                "item": item,
                "component": step,
                "cited": line,
                "expected": rules,
                "strict": line in rules,
                "loose": near(line, rules) or (line is not None and line_value(line) == s["value"]),
            }
        )
    return out


def score_baggage(q: dict, reply: dict, arm: str) -> dict:
    p = parse("baggage", reply["text"])
    trace = p["trace"]
    pred = p["pred"] if p["pred"] is not None else (trace or {}).get("total")
    row = {
        "id": q["id"],
        "answer": q["answer"],
        "pred": pred,
        "exact": pred == q["answer"],
        "within30": pred is not None and abs(pred - q["answer"]) <= 30,
        "finished": p["finished"],
        "cut_off": reply["finish"] == "length",
        "tokens": reply["tokens"],
    }
    if arm != "b":
        return row
    row["trace_ok"] = trace is not None
    if trace is None:
        row.update(c_pred=pred, c_exact=row["exact"], c_status="no trace")
        row["first_failure"] = None if row["exact"] else ("no answer" if pred is None else "no trace")
        return row
    row["trace_total_matches"] = trace.get("total") == pred
    row["trace_bags"] = [{"item": b["item"], "size": b["size"], "weight": b["weight"]} for b in trace["bags"]]
    diffs = fact_diffs(trace, facts_of(q["info"]))
    ident = identification(trace, q["info"])
    row.update(fact_diffs=diffs, identification=ident)
    try:
        v = verify(trace, q["info"])
        c_pred = trace.get("total") if v["verdict"] == "pass" else v["checker_total"]
        row.update(
            verification={k: v[k] for k in ("verdict", "first_failure", "checks", "checker_total")},
            c_pred=c_pred,
            c_status=v["verdict"],
        )
    except (ValueError, KeyError, TypeError, StopIteration) as e:  # facts the spec cannot evaluate (e.g. no region)
        row.update(verification={"verdict": "unverifiable", "error": str(e)}, c_pred=pred, c_status="unverifiable")
    row["c_exact"] = row["c_pred"] == q["answer"]
    if row["exact"]:
        row["first_failure"] = None
    elif diffs:
        row["first_failure"] = "facts"
    elif not all(i["strict"] for i in ident):
        row["first_failure"] = "identification"
    else:
        row["first_failure"] = "application"
    row["first_failure_loose"] = row["first_failure"]
    if row["first_failure"] == "identification" and all(i["loose"] for i in ident):
        row["first_failure_loose"] = "application"
    return row


def us_schedule(info: dict) -> int:
    """Cheap baseline: every passenger pays the U.S. Main Cabin schedule, no exceptions, bags in question order."""
    total = info["base_price"]
    for pos, b in enumerate(info["bag_list"][1:]):
        w, s = b["weight"], sum(b["size"])
        ow = 0 if w <= 50 else 30 if w <= 53 else 100 if w <= 70 else 200
        os_ = 0 if s <= 62 else 30 if s <= 65 else 200
        total += [40, 45, 150][min(pos, 2)] + max(ow, os_)
    return total


CITY = re.compile(r"Class passenger flying from (?P<dep>.+?) to (?P<dst>.+?) with the following items")
ITEM = re.compile(r"^(\d+)\. A [a-z ]+: (\d+) x (\d+) x (\d+) inches, (\d+) lbs;$", re.MULTILINE)


def checker_only(q: dict) -> int:
    """The checker alone: facts read from the question text by regex (cities mapped with RuleArena's city table)."""
    from questions import rulearena_gen

    m = CITY.search(q["prompt"])
    cities = rulearena_gen.international_city_collection
    abroad = [c for c in (m["dep"], m["dst"]) if c in cities]
    region = cities[abroad[0]] if abroad else "U.S."
    direction = 0 if m["dep"] in rulearena_gen.us_city_collection else 1
    cls = re.match(r"\w+ is a (.+?) Class passenger", q["prompt"]).group(1)
    ticket = int(re.search(r"flight ticket is \$(\d+)", q["prompt"]).group(1))
    bags = [
        {"id": int(i), "size": [int(a), int(b), int(c)], "weight": int(w)}
        for i, a, b, c, w in ITEM.findall(q["prompt"])
    ]
    info = {"base_price": ticket, "customer_class": cls, "routine": region, "direction": direction, "bag_list": bags}
    return evaluate(info)["total"]


def random_citation_rate(rows: list[dict]) -> float:
    """Chance level of strict identification: cite a uniformly random line of the right kind for each component."""
    from checker import _index

    kinds = {
        "base": [r["id"] for r in _index()["base_fee"]] + ["L103", "L164"],
        "weight": [r["id"] for r in _index()["weight_fee"]] + ["L274", "L275", "L279"],
        "size": [r["id"] for r in _index()["size_fee"]] + ["L273"],
    }
    hits = []
    for q in rows:
        for s in evaluate(q["info"])["steps"]:
            if s["step"] in kinds:
                pool = kinds[s["step"]]
                hits.append(len(set(s["rules"]) & set(pool)) / len(pool))
    return round(sum(hits) / len(hits), 4)


def baggage_summary(rows: list[dict], arm: str) -> dict:
    out = {
        "exact": rate([r["exact"] for r in rows]),
        "within30": rate([r["within30"] for r in rows]),
        "cut_off": sum(r["cut_off"] for r in rows),
        "no_answer": sum(r["pred"] is None for r in rows),
        "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
    }
    finished = [r for r in rows if not r["cut_off"]]
    out["exact_finished_only"] = rate([r["exact"] for r in finished])
    if arm == "b":
        out["c_exact"] = rate([r["c_exact"] for r in rows])
        out["c_within30"] = rate([r["c_pred"] is not None and abs(r["c_pred"] - r["answer"]) <= 30 for r in rows])
        out["c_fixed"] = sum(r["c_exact"] and not r["exact"] for r in rows)  # wrong in b, right after the checker
        out["c_broke"] = sum(r["exact"] and not r["c_exact"] for r in rows)  # right in b, wrong after the checker
        out["c_broke_carry_on"] = sum(
            r["exact"] and not r["c_exact"] and any(b["item"] == 1 for b in r.get("trace_bags", [])) for r in rows
        )
        out["trace_ok"] = sum(r["trace_ok"] for r in rows)
        out["c_status"] = dict(Counter(r["c_status"] for r in rows))
        verified = [r for r in rows if r["c_status"] == "pass"]
        out["checker_passed"] = len(verified)
        out["checker_passed_but_wrong"] = sum(not r["exact"] for r in verified)
        failed = [r for r in rows if r["c_status"] == "fail"]
        out["checker_failed_but_right"] = sum(r["exact"] for r in failed)
        comps = [i for r in rows if r["trace_ok"] for i in r["identification"]]
        out["ident_strict"] = rate([i["strict"] for i in comps])
        out["ident_loose"] = rate([i["loose"] for i in comps])
        right = [i for r in rows if r["trace_ok"] and not r["fact_diffs"] for i in r["identification"]]
        out["ident_strict_facts_right"] = rate([i["strict"] for i in right])
        out["ident_loose_facts_right"] = rate([i["loose"] for i in right])
        out["ident_by_component"] = {
            c: rate([i["strict"] for i in comps if i["component"] == c]).get("rate") for c in ("base", "weight", "size")
        }
        out["facts_right"] = rate([not r["fact_diffs"] for r in rows if r["trace_ok"]])
        wrong = [r for r in rows if not r["exact"]]
        out["first_failure"] = dict(Counter(r["first_failure"] for r in wrong))
        out["first_failure_loose"] = dict(Counter(r.get("first_failure_loose", r["first_failure"]) for r in wrong))
        parseable_wrong = [r for r in wrong if r["trace_ok"]]
        out["share_identification_strict"] = rate([r["first_failure"] == "identification" for r in parseable_wrong])
        out["share_identification_loose"] = rate(
            [r["first_failure_loose"] == "identification" for r in parseable_wrong]
        )
        out["wrong_rule_cited"] = rate([not all(i["strict"] for i in r["identification"]) for r in parseable_wrong])
        out["fact_errors"] = dict(
            Counter(
                d.split(" ")[0] if d.startswith("item") else d for r in rows if r["trace_ok"] for d in r["fact_diffs"]
            )
        )
    return out


# ----------------------------------------------------------------------------- contrast


def score_contrast(q: dict, reply: dict, arm: str) -> dict:
    p = parse("contrast", reply["text"])
    trace = p["trace"]
    truth = q["answer"]
    pred = p["pred"] or (trace or {}).get("verdict")
    row = {
        "id": q["id"],
        "verdict": truth["verdict"],
        "pred": pred,
        "verdict_ok": pred == truth["verdict"],
        "finished": p["finished"],
        "cut_off": reply["finish"] == "length",
        "tokens": reply["tokens"],
        "near_threshold": truth["contrast_rule"] in ("D31", "D32")
        and abs(truth["ratio"] - {"D31": 4.5, "D32": 3.0}[truth["contrast_rule"]]) <= 1.0,
    }
    if arm != "b":
        return row
    row["trace_ok"] = trace is not None
    if trace is None:
        row.update(exact=False, exact_umbrella=False, c_exact=False, c_status="no trace", rule_ok=False, ratio_ok=False)
        return row
    row["exact"] = row["verdict_ok"] and trace["rules_failed"] == truth["rules_failed"]
    # Exploratory, added after reading held-out replies: D3 ("text must meet 1.4.3") is the umbrella of D31/D32, so a
    # reply that names it beside, or instead of, the contrast line that failed is read as naming that line.
    umbrella = {r for r in trace["rules_failed"] if r != "D3"}
    if "D3" in trace["rules_failed"]:
        umbrella |= {r for r in truth["rules_failed"] if r in ("D31", "D32")}
    row["exact_umbrella"] = row["verdict_ok"] and sorted(umbrella, key=lambda r: int(r[1:])) == truth["rules_failed"]
    row["rule_ok"] = trace["contrast_rule"] == truth["contrast_rule"]
    row["ratio_ok"] = isinstance(trace["ratio"], (int, float)) and abs(trace["ratio"] - truth["ratio"]) <= 0.05
    row["facts_ok"] = False
    try:
        v = contrast.verify(trace, q["facts"])
        row["facts_ok"] = not v["checks"][0]["where"]
        chosen = (
            (trace["verdict"], trace["rules_failed"])
            if v["verdict"] == "pass"
            else (v["checker"]["verdict"], v["checker"]["rules_failed"])
        )
        row.update(verification=v, c_status=v["verdict"])
    except (KeyError, TypeError, ValueError, IndexError) as e:
        chosen = (pred, trace["rules_failed"])
        row.update(verification={"verdict": "unverifiable", "error": str(e)}, c_status="unverifiable")
    row["c_exact"] = chosen[0] == truth["verdict"] and chosen[1] == truth["rules_failed"]
    return row


def contrast_summary(rows: list[dict], arm: str) -> dict:
    out = {
        "verdict": rate([r["verdict_ok"] for r in rows]),
        "verdict_near_threshold": rate([r["verdict_ok"] for r in rows if r["near_threshold"]]),
        "cut_off": sum(r["cut_off"] for r in rows),
        "no_answer": sum(r["pred"] is None for r in rows),
        "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
    }
    if arm == "b":
        out |= {
            "exact": rate([r["exact"] for r in rows]),
            "exact_umbrella": rate([r["exact_umbrella"] for r in rows]),
            "c_exact": rate([r["c_exact"] for r in rows]),
            "rule_identified": rate([r["rule_ok"] for r in rows]),
            "ratio_within_0.05": rate([r["ratio_ok"] for r in rows]),
            "facts_ok": rate([r.get("facts_ok", False) for r in rows]),
            "trace_ok": sum(r["trace_ok"] for r in rows),
            "c_status": dict(Counter(r["c_status"] for r in rows)),
        }
    return out


# ----------------------------------------------------------------------------- main


def sensitivity() -> dict:
    """Post-hoc, labelled as such: the scores again with the lenient parser reading (parse.LENIENT)."""
    import parse as parse_module

    out = {}
    parse_module.LENIENT = True
    try:
        for model in ("qwen3-8b", "qwen3-32b"):
            for task, arm, suffix in (("baggage", "b", ""), ("baggage", "b", "_s1"), ("contrast", "a", "")):
                path = RAW / f"{task}_heldout_{model}_{arm}{suffix}.jsonl"
                if not path.exists():
                    continue
                qs = {q["id"]: q for q in read_jsonl(DATA / f"{task}_heldout.jsonl")}
                fn = score_baggage if task == "baggage" else score_contrast
                rows = [fn(qs[r["id"]], r, arm) for r in read_jsonl(path)]
                summ = (baggage_summary if task == "baggage" else contrast_summary)(rows, arm)
                keep = (
                    ("exact", "c_exact", "first_failure", "share_identification_strict")
                    if task == "baggage"
                    else ("verdict",)
                )
                out[f"{task}/{model}/{arm}{suffix}"] = {k: summ[k] for k in keep}
    finally:
        parse_module.LENIENT = False
    return out


def main() -> None:
    summary: dict = {"baggage": {}, "contrast": {}, "paired": {}, "baselines": {}}
    scored: dict[tuple, list[dict]] = {}
    for path in sorted(RAW.glob("*.jsonl")):
        m = NAME.match(path.stem)
        if not m:
            continue  # pilot files
        task, split, model, arm, seed = m["task"], m["split"], m["model"], m["arm"], m["seed"] or "0"
        questions = {q["id"]: q for q in read_jsonl(DATA / f"{task}_{split}.jsonl")}
        replies = read_jsonl(path)
        fn = score_baggage if task == "baggage" else score_contrast
        rows = [fn(questions[r["id"]], r, arm) for r in replies]
        write_jsonl(SCORED / path.name, rows)
        key = arm if seed == "0" else f"{arm}_s{seed}"
        scored[(task, split, model, key)] = rows
        summarise = baggage_summary if task == "baggage" else contrast_summary
        summary[task].setdefault(split, {}).setdefault(model, {})[key] = summarise(rows, arm)
        print(f"{path.stem}: {len(rows)} scored")
    for (task, split, model, key), rows in scored.items():
        if key != "b":
            continue
        name = f"{task}/{split}/{model}"
        flag = "exact" if task == "baggage" else "verdict_ok"
        b = {r["id"]: r[flag] for r in rows}
        c = {r["id"]: r["c_exact"] for r in rows}
        pairs = {"c-b": paired(c, b) if task == "baggage" else paired(c, {r["id"]: r["exact"] for r in rows})}
        for other in ("a", "b0", "b_s1"):
            if (task, split, model, other) in scored:
                o = {r["id"]: r[flag] for r in scored[(task, split, model, other)]}
                pairs[f"b-{other}"] = paired(b, o)
        summary["paired"][name] = pairs
    for split in ("dev", "heldout"):
        rows = read_jsonl(DATA / f"baggage_{split}.jsonl")
        summary["baselines"][split] = {
            "ticket_only": rate([q["info"]["base_price"] == q["answer"] for q in rows]),
            "us_schedule": rate([us_schedule(q["info"]) == q["answer"] for q in rows]),
            "checker_only": rate([checker_only(q) == q["answer"] for q in rows]),
            "random_citation_ident": random_citation_rate(rows),
        }
        crow = read_jsonl(DATA / f"contrast_{split}.jsonl")
        summary["baselines"][f"contrast_{split}"] = {
            "always_compliant": rate([q["answer"]["verdict"] == "compliant" for q in crow]),
            "always_not_compliant": rate([q["answer"]["verdict"] == "not compliant" for q in crow]),
        }
    summary["sensitivity"] = sensitivity()
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote results/summary.json")


if __name__ == "__main__":
    sys.exit(main())
