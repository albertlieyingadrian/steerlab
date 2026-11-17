"""Every number in the note, from results/ (no model calls). Writes results/summary.json and prints it.

python analyze.py
"""

from __future__ import annotations

import collections
import json
from itertools import combinations, pairwise

import bench
import numpy as np

SIZES = (50, 100, 200, 500, 1000, 2000, 5700)


def load(name: str) -> dict:
    return json.loads((bench.RESULTS / name).read_text())


def rate(flags: list[bool]) -> dict:
    k = int(sum(flags))
    return {"k": k, "n": len(flags), "rate": round(k / max(len(flags), 1), 4), "ci": bench.wilson(k, len(flags))}


def corrected(r: dict) -> int | None:
    c = r["correct_answer"]
    if c is None:
        return None
    c = str(c).strip()
    if c[:1] in "ABCD":
        return "ABCD".index(c[:1])
    if c[:1].isdigit() and int(c[:1]) < len(r["choices"]):
        return int(c[:1])
    return None


def part_a() -> dict:
    items = bench.mmlu_redux()
    key = np.array([r["answer"] for r in items])
    bad = np.array([r["error_type"] != "ok" for r in items])
    wrong_gt = np.array([r["error_type"] == "wrong_groundtruth" for r in items])
    models = {
        k: load(f"a_open/{k}.json") for k in bench.OPEN_MODELS if (bench.RESULTS / "a_open" / f"{k}.json").exists()
    }
    api = load("a_api.json")
    aud = np.array(api["pred"])
    out: dict = {
        "n": len(items),
        "experts_flagged": rate(list(bad)),
        "wrong_groundtruth": int(wrong_gt.sum()),
        "open_models": list(models),
    }

    # A2: label audit
    rot0 = {k: np.array(m["rot0"]) for k, m in models.items()}
    stack = np.stack(list(rot0.values()))
    consensus = (stack == stack[0]).all(0) & (stack[0] != key)
    auditor = (aud != key) & (aud >= 0)
    a2: dict = {}
    for name, flag in (
        ("auditor_gpt41mini", auditor),
        ("open_consensus", consensus),
        ("either", auditor | consensus),
        ("both", auditor & consensus),
    ):
        a2[name] = {
            "flags": int(flag.sum()),
            "flags_per_100": round(100 * float(flag.mean()), 2),
            "precision": rate(list(bad[flag])),
            "recall": rate(list(flag[bad])),
            "recall_wrong_groundtruth": rate(list(flag[wrong_gt])),
            "lift_over_base_rate": round(float(bad[flag].mean() / bad.mean()), 2) if flag.any() else None,
        }
    a2["auditor_unparsed"] = int((aud < 0).sum())
    out["A2_label_audit"] = a2

    # A3: TS-Guessing
    by_id = {r["id"]: r for r in items}
    a3 = {}
    for name, m in [*models.items(), ("auditor_gpt41mini", api)]:
        tsr = [by_id[i] for i in m["ts_ids"]]
        target = [bench.norm(r["choices"][max(j for j in range(4) if j != r["answer"])]) for r in tsr]
        exact = [bench.norm(f) == t for f, t in zip(m["ts"], target, strict=True)]
        ctrl = [bench.norm(f) == t for f, t in zip(m["ts_control"], target, strict=True)]
        a3[name] = {
            "masked_exact": rate(exact),
            "question_only_exact": rate(ctrl),
            "mcnemar": bench.mcnemar(exact, ctrl),
        }

    # Exploratory (added after reading A3): drop items whose masked option uses only words found in the other options
    # ("True, False" next to "True, True" and "False, False"), which can be rebuilt without memory.
    def derivable(r: dict) -> bool:
        m = max(j for j in range(4) if j != r["answer"])
        others = {w for j, c in enumerate(r["choices"]) if j != m for w in bench.norm(c).split()}
        words = bench.norm(r["choices"][m]).split()
        return bool(words) and set(words) <= others

    for name, m in [*models.items(), ("auditor_gpt41mini", api)]:
        tsr = [by_id[i] for i in m["ts_ids"]]
        keep = [not derivable(r) for r in tsr]
        target = [bench.norm(r["choices"][max(j for j in range(4) if j != r["answer"])]) for r in tsr]
        exact = [bench.norm(f) == t for f, t, k in zip(m["ts"], target, keep, strict=True) if k]
        ctrl = [bench.norm(f) == t for f, t, k in zip(m["ts_control"], target, keep, strict=True) if k]
        a3[name]["exploratory_not_derivable"] = {
            "masked_exact": rate(exact),
            "question_only_exact": rate(ctrl),
            "mcnemar": bench.mcnemar(exact, ctrl),
        }
    out["A3_contamination"] = a3

    # A4: shortcuts
    a4: dict = {k: rate(list(np.array(m["choices"]) == key)) for k, m in models.items()}
    longest = np.array([int(np.argmax([len(c) for c in r["choices"]])) for r in items])
    a4["longest_option_rule"] = rate(list(longest == key))
    out["A4_shortcuts"] = a4

    # A5: format sensitivity
    a5: dict = {}
    accs = {}
    for k, m in models.items():
        rs = [np.array(m[f"rot{r}"]) == key for r in range(4)]
        accs[k] = [float(x.mean()) for x in rs]
        same = np.all([np.array(m[f"rot{r}"]) == np.array(m["rot0"]) for r in range(1, 4)], axis=0)
        a5[k] = {
            "acc_by_rotation": [round(a, 4) for a in accs[k]],
            "range": round(max(accs[k]) - min(accs[k]), 4),
            "consistent_all_4": rate(list(same)),
            "right_in_all_4": rate(list(np.all(rs, axis=0))),
        }
    a5["ranking_flips_across_rotations"] = [
        {"pair": [a, b], "flips": len({np.sign(x - y) for x, y in zip(accs[a], accs[b], strict=True)}) > 1}
        for a, b in combinations(models, 2)
    ]
    out["A5_format"] = a5

    # A6: statistics
    correct = {k: (np.array(m["rot0"]) == key) for k, m in models.items()}
    correct["auditor_gpt41mini"] = aud == key
    a6: dict = {
        "acc": {
            k: {"acc": round(float(v.mean()), 4), "ci": bench.boot_mean(v.astype(float))} for k, v in correct.items()
        }
    }
    ranked = sorted(correct, key=lambda k: -correct[k].mean())
    a6["ranking"] = ranked
    a6["items_needed"] = {f"{a}>{b}": bench.items_needed(correct[a], correct[b], SIZES) for a, b in pairwise(ranked)}
    out["A6_statistics"] = a6

    # A7: does cleaning change conclusions
    ok = ~bad
    a7: dict = {
        k: {
            "all": round(float(v.mean()), 4),
            "redux_ok": round(float(v[ok].mean()), 4),
            "shift": round(float(v[ok].mean() - v.mean()), 4),
        }
        for k, v in correct.items()
    }
    a7["ranking_all"] = ranked
    a7["ranking_ok"] = sorted(correct, key=lambda k: -correct[k][ok].mean())
    fix = [
        (i, corrected(r))
        for i, r in enumerate(items)
        if r["error_type"] == "wrong_groundtruth" and corrected(r) is not None
    ]
    a7["wrong_groundtruth_with_fix"] = len(fix)
    preds = {k: np.array(m["rot0"]) for k, m in models.items()} | {"auditor_gpt41mini": aud}
    a7["on_wrong_groundtruth"] = {
        k: {
            "vs_original_key": round(float(np.mean([p[i] == key[i] for i, _ in fix])), 4),
            "vs_corrected_key": round(float(np.mean([p[i] == c for i, c in fix])), 4),
        }
        for k, p in preds.items()
    }
    out["A7_cleaning"] = a7

    # A8: difficulty
    subj = collections.defaultdict(list)
    for i, r in enumerate(items):
        subj[r["subject"]].append(i)
    best_open = max(models, key=lambda k: correct[k].mean())
    a8 = {}
    for k in (best_open, "auditor_gpt41mini"):
        per = {s: float(correct[k][idx].mean()) for s, idx in subj.items()}
        a8[k] = {
            "saturated_subjects_ge_0.9": sum(v >= 0.9 for v in per.values()),
            "subjects": len(per),
            "hardest": sorted(per.items(), key=lambda x: x[1])[:5],
            "easiest": sorted(per.items(), key=lambda x: -x[1])[:5],
        }
    out["A8_difficulty"] = a8
    return out


def part_b() -> dict:
    items = load("sloppair_items.json")["items"]
    jd = load("b_judge.json")
    bl = load("b_baselines.json")
    test_ids = jd["test_ids"]
    gen = {r["id"]: r["generator"] for r in items}
    out: dict = {
        "pairs": len(items),
        "test": len(test_ids),
        "facts_kept": rate([r["facts_kept"] for r in items]),
        "words_ratio_ai_over_human": {},
    }
    for g in bench.GENERATORS:
        rs = [r for r in items if r["generator"] == g]
        out["words_ratio_ai_over_human"][g] = round(float(np.mean([r["ai_words"] / r["human_words"] for r in rs])), 3)

    out["B3_contamination"] = {
        m: {
            "rougeL_human_half": round(float(np.mean(v["human"])), 4),
            "rougeL_ai_half": round(float(np.mean(v["ai"])), 4),
            "diff_ci": bench.boot_mean(np.array(v["human"]) - np.array(v["ai"])),
        }
        for m, v in jd["contam"].items()
    }

    out["B4_baselines"] = {
        k: {"acc": round(float(np.mean(v)), 4), "ci": bench.boot_mean(v)} for k, v in bl["orig"].items()
    }
    out["B4_baselines"]["bow_cv_all300"] = bl["bow_cv_all300"]
    out["B4_bow_top_words"] = bl["bow_top_words"]

    def judged(block: dict) -> tuple[dict, dict]:
        res, per_pair = {}, {}
        for m, v in block.items():
            f, r = v["human_first"], v["ai_first"]
            right = [(x == "human") + (y == "human") for x, y in zip(f, r, strict=True)]
            per_pair[m] = np.array(right) / 2
            res[m] = {
                "acc_both_orders": round(float(np.mean(per_pair[m])), 4),
                "acc_ci": bench.boot_mean(per_pair[m]),
                "right_in_both_orders": rate([x == 2 for x in right]),
                "order_flips": rate([x != y for x, y in zip(f, r, strict=True) if x and y]),
                "picks_first_passage": rate([x == "human" for x in f if x] + [y == "ai" for y in r if y]),
                "unparsed": sum(x is None for x in f + r),
                "by_generator": {
                    g: round(float(np.mean([p for p, i in zip(per_pair[m], test_ids, strict=True) if gen[i] == g])), 4)
                    for g in bench.GENERATORS
                },
            }
        return res, per_pair

    orig, orig_pp = judged(jd["judges"])
    hill, hill_pp = judged(jd["hill"])
    out["B5_B6_judges"] = orig
    hc = bench.RESULTS / "human_check.json"
    if hc.exists():
        ans = json.loads(hc.read_text())["answers"]
        order = {r["id"]: r["human_is"] for r in items}
        ids = [i for i in ans if i in order]
        person = [int(ans[i] == order[i]) for i in ids]
        res = {"n": len(ids), "person": rate([bool(x) for x in person])}
        pos = {i: k for k, i in enumerate(test_ids)}
        choice = [ans[i] for i in ids]
        for m, v in jd["judges"].items():
            # the judge's letter in the page's order (the page shows the pair in its stored A/B order)
            j = []
            for i in ids:
                said = (v["human_first"] if order[i] == "A" else v["ai_first"])[pos[i]]
                j.append(order[i] if said == "human" else ("B" if order[i] == "A" else "A"))
            res[m] = {"kappa_with_person": bench.kappa([c == "A" for c in choice], [c == "A" for c in j])}
        out["B6_human_check"] = res
    else:
        out["B6_human_check"] = None
    ranked = sorted(orig_pp, key=lambda m: -orig_pp[m].mean())
    out["B7_statistics"] = {
        "ranking": ranked,
        "mcnemar_right_in_both_orders": {
            f"{a} vs {b}": bench.mcnemar(list(orig_pp[a] == 1), list(orig_pp[b] == 1))
            for a, b in combinations(ranked, 2)
        },
        "items_needed_top2": bench.items_needed(orig_pp[ranked[0]], orig_pp[ranked[1]], (50, 100, 200)),
    }
    out["B8_hill_climb"] = {
        "judges": {
            m: {
                "orig": orig[m]["acc_both_orders"],
                "hill": hill[m]["acc_both_orders"],
                "hill_ci": hill[m]["acc_ci"],
                "hill_by_generator": hill[m]["by_generator"],
            }
            for m in orig
        },
        "baselines": {
            k: {
                "orig": round(float(np.mean(bl["orig"][k])), 4),
                "hill": round(float(np.mean(bl["hill"][k])), 4),
                "hill_ci": bench.boot_mean(bl["hill"][k]),
            }
            for k in bl["orig"]
        },
        "ranking_after": sorted(hill_pp, key=lambda m: -hill_pp[m].mean()),
    }
    return out


def main() -> None:
    out = {
        "canary": bench.CANARY,
        "A": part_a(),
        "B": part_b() if (bench.RESULTS / "b_baselines.json").exists() else None,
        "api_spend_usd_estimate": round(bench.spent(), 2),
    }
    (bench.RESULTS / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
