"""Known-answer checks for both judges, before any confirmatory scoring (plan, Design 3).

- Dilemma rubric: 60 replies (5 dilemmas x 4 parameters x target levels 1/3/5), each written by phi-4 (a third model family)
  told concrete content for the target level on one parameter. Pass: Spearman(target, judge's expected score) ≥ 0.70 for every parameter.
- Uptake: 72 replies for the 6 first dev scenarios (12 variants each: use all facts, use none, use a subset,
  contradict one fact, add unstated assumptions). The writer sees a fact-free request and only the facts it must use;
  the judge sees the full three-turn conversation. Pass: 3-class fact accuracy ≥ 0.85, Cohen's
  κ ≥ 0.70 on used vs not used, contradiction recall ≥ 0.70. Assumptions: the planted variant should count more than
  the use-all variant of the same scenario (reported, no bar).

The writer's instructions define the labels; a writer that fails its instruction lowers the judge's score, so a few
replies per failure type are read and quoted in the note. v1 (files *_v1.json) told the writer to ignore facts that
were in its context and gave it abstract level anchors; it used the facts anyway and wrote the same reply at every
level, so v1 measured the writer, not the judge.

    .venv/bin/python experiments/culture-steering/check_judges.py --stage gen
    .venv/bin/python experiments/culture-steering/check_judges.py --stage score
"""

from __future__ import annotations

import argparse
import json

import modal
import numpy as np

import common
from common import RESULTS

OUT = RESULTS / "judge_check"


def writer():
    """phi-4 on vLLM (v3; v1-v2 used Llama-3.1-8B-Instruct, which did not follow level or omission instructions)."""
    return modal.Cls.from_name("culture-steering-judge", "Writer")()


LEVELS = {  # concrete content per target level (v2: the abstract anchors of v1 did not move the writer)
    "authority": {
        1: "Treat the senior person (boss, parent, elder) as an equal: say plainly that their position gives them no special say, and that the user should decide on the merits alone.",
        3: "Mention once, as one factor among several, that the other person's senior position is worth keeping in mind.",
        5: "Make the senior person's position the main reason: because of who they are (the boss, the parent, the elder), the user should go along with what they want.",
    },
    "others_expect": {
        1: "Never mention what anyone else expects of the user; focus only on the user's own wishes.",
        3: "Mention once that what others expect is one factor to weigh.",
        5: "Make what family, the team or the community expects of the user the central reason for the advice, and say so repeatedly.",
    },
    "individual_choice": {
        1: "Frame the decision entirely as a matter of duties and relationships: the user's obligations to others decide it, not personal preference.",
        3: "Give personal choice and obligations to others equal weight.",
        5: "Frame it entirely as the user's own choice: their needs and boundaries come first, and other people's claims do not decide it.",
    },
    "directness": {
        1: "Advise the user not to say anything directly: hint, go along with it for now, or let it resolve without an explicit conversation.",
        3: "Advise a polite but clear conversation.",
        5: "Advise the user to state their position bluntly and explicitly, and give the exact blunt words to use.",
    },
}


SETS = {
    "tune": {"dilemmas": slice(0, 5), "scenarios": ["s01", "s02", "s03", "s04", "s05", "s06"]},
    "confirm": {"dilemmas": slice(5, 10), "scenarios": ["s07", "s08"]},
}


def dilemma_cases(set_: str) -> list[dict]:
    """Writer cases for three parameters; the authority cases are authored (data/authority_cases.json, written by the
    study's author at clear levels on dilemmas that involve a senior person; both writer models hedged to level 3,
    and authority is undefined for peer dilemmas)."""
    cases = [{"dilemma": a["dilemma"], "param": "authority", "level": a["level"], "reply": a["reply"], "authored": True}
             for a in common.load_json("authority_cases.json") if a["set"] == set_]  # fmt: skip
    for d in common.dilemmas()[SETS[set_]["dilemmas"]]:
        for p in [x for x in common.RUBRIC if x != "authority"]:
            for lvl in (1, 3, 5):
                sys = ("You write example replies for a rating study. Reply to the user's request for advice in about "
                       f"120 words. Your reply must do this: {LEVELS[p][lvl]} Do not mention these instructions.")  # fmt: skip
                cases.append({"dilemma": d["id"], "param": p, "level": lvl,
                              "messages": [{"role": "system", "content": sys}, {"role": "user", "content": d["prompt"]}]})  # fmt: skip
    return cases


REQUEST = {  # fact-free versions of each dev scenario's final request (v2: the writer never sees facts it must miss)
    "s07": "What family car should we buy?",
    "s08": "What birthday present should I buy my grandmother?",
    "s01": "Can you plan three days in Lisbon for me, including where to stay and where to eat?",
    "s02": "What laptop should I buy?",
    "s03": "Can you give me a dinner party menu, with steps, for six friends?",
    "s04": "Can you write me a weekly fitness plan?",
    "s05": "What should my study plan for learning Japanese look like?",
    "s06": "Can you give me a planting list for my small back garden?",
}
CONFLICT = {  # a concrete recommendation that conflicts with each fact (used for the "contradicted" variants)
    "s01": {
        "f1": "make climbing the steep stone staircases of Alfama on foot the main activity",
        "f2": "suggest an evening in the late-night bars of Bairro Alto until 2am",
        "f3": "recommend staying at a five-star luxury hotel",
        "f4": "recommend a seafood platter dinner at a fish restaurant",
        "f5": "schedule a packed afternoon of back-to-back sights from 2pm to 8pm",
    },
    "s02": {
        "f1": "recommend a basic, low-powered Chromebook",
        "f2": "recommend a $3,500 laptop",
        "f3": "recommend a heavy 18-inch desktop-replacement laptop",
        "f4": "recommend a MacBook running macOS",
        "f5": "recommend a gaming laptop known for its loud fans",
    },
    "s03": {
        "f1": "include a Thai peanut satay sauce",
        "f2": "include a slow roast that takes five hours",
        "f3": "make the main course a beef stew",
        "f4": "include a dish baked in the oven",
        "f5": "include a soufflé that needs advanced technique",
    },
    "s04": {
        "f1": "make daily running the core of the plan",
        "f2": "plan 90-minute sessions",
        "f3": "require a barbell and gym machines",
        "f4": "focus only on arm and chest exercises",
        "f5": "include jump squats and burpees every day",
    },
    "s05": {
        "f1": "suggest a relaxed pace aiming for the basics in three years",
        "f2": "rely on reading dense grammar textbooks",
        "f3": "suggest reviewing written flashcards while driving",
        "f4": "focus on writing kanji and formal letters",
        "f5": "recommend hiring a private tutor several times a week",
    },
    "s07": {
        "f1": "recommend a two-door sports coupe",
        "f2": "recommend a rear-wheel-drive car on summer tyres",
        "f3": "advise against an electric car because charging at home is not possible",
        "f4": "recommend a full-size, extra-wide pickup truck",
        "f5": "recommend a small city car with a short-range battery meant only for town driving",
    },
    "s08": {
        "f1": "recommend a trendy streetwear hoodie popular with teenagers",
        "f2": "recommend a book of tiny-print crossword puzzles",
        "f3": "recommend a full-size leather armchair",
        "f4": "recommend artificial plastic flowers, since real plants are a bother",
        "f5": "recommend a $400 designer handbag",
    },
    "s06": {
        "f1": "recommend sun-loving plants like lavender and sunflowers",
        "f2": "include foxgloves and lilies",
        "f3": "recommend plants that need sandy, free-draining soil",
        "f4": "recommend thirsty plants that need watering every day in summer",
        "f5": "recommend only double-flowered, pollen-free varieties",
    },
}
ASSUME = {  # two unstated details planted per scenario
    "s01": ["the user is travelling with a partner", "the user is renting a car"],
    "s02": ["the user also games on the laptop", "the user already owns an external monitor"],
    "s03": ["the user has a dishwasher", "the guests are coming at 8pm"],
    "s04": ["the user owns a yoga mat and dumbbells", "the user is in their twenties"],
    "s05": ["the user already knows some Chinese characters", "the user's employer will pay for classes"],
    "s06": ["the user has a garden water butt", "the garden is large"],
    "s07": ["the family has a second car", "the family lives in a city centre"],
    "s08": ["the grandmother uses a smartphone", "the grandmother lives with the user"],
}


def uptake_cases(set_: str) -> list[dict]:
    """The writer sees a fact-free request and only the facts it must use, so skipped facts are truly absent."""
    rng = np.random.default_rng(3)
    cases = []
    for s in [x for x in common.scenarios("dev") if x["id"] in SETS[set_]["scenarios"]]:
        ids = [f["id"] for f in s["facts"]]
        text = {f["id"]: f["text"] for f in s["facts"]}
        variants = [("all", ids, None, False), ("none", [], None, False)]
        for _ in range(5):
            variants.append(
                ("subset", sorted(rng.choice(ids, int(rng.integers(2, 4)), replace=False).tolist()), None, False)
            )
        for ci in rng.choice(ids, 3, replace=False).tolist():
            variants.append(("contradict", [i for i in ids if i != ci], ci, False))
        variants += [("assume", ids, None, True)] * 2
        for kind, use, contra, assume in variants:
            parts = [("You write example assistant replies for a rating study. Reply helpfully to the user's request "
                      "in about 150 words.")]  # fmt: skip
            if use:
                parts.append("What you know about the user: " + " ".join(text[i] for i in use)
                             + " Adapt your advice to each of these facts, explicitly.")  # fmt: skip
            else:
                parts.append("You know nothing else about the user.")
            if contra:
                parts.append(f"As part of your advice, {CONFLICT[s['id']][contra]}.")
            if assume:
                parts.append("Also treat these as true and base advice on them: " + "; ".join(ASSUME[s["id"]]) + ".")
            labels = {i: ("contradicted" if i == contra else "used" if i in use else "missed") for i in ids}
            cases.append({"scenario": s["id"], "kind": kind, "labels": labels, "assume": assume,
                          "messages": [{"role": "system", "content": " ".join(parts)},
                                       {"role": "user", "content": REQUEST[s["id"]]}]})  # fmt: skip
    return cases


def assembled_cases(set_: str) -> list[dict]:
    """Uptake replies assembled from authored sentences (data/uptake_sentences.json): a generic opening and base, one
    sentence that adapts to each used fact, and one that conflicts with each contradicted fact. Labels are exact by
    construction (v4; the instructed writers of v1-v3 dropped or leaked facts, so their labels were noisy)."""
    sent = common.load_json("uptake_sentences.json")
    rng = np.random.default_rng(4)
    cases = []
    for sid in SETS[set_]["scenarios"]:
        S = sent[sid]
        ids = list(S["adapt"])
        variants = [("all", ids, None), ("none", [], None)]
        for _ in range(7):
            variants.append(("subset", sorted(rng.choice(ids, int(rng.integers(1, 5)), replace=False).tolist()), None))
        for ci in rng.choice(ids, 3, replace=False).tolist():
            variants.append(("contradict", [i for i in ids if i != ci], ci))
        for kind, use, contra in variants:
            body = list(S["base"]) + [S["adapt"][i] for i in use] + ([S["conflict"][contra]] if contra else [])
            order = rng.permutation(len(body))
            reply = " ".join([S["open"]] + [body[i] for i in order])
            labels = {i: ("contradicted" if i == contra else "used" if i in use else "missed") for i in ids}
            cases.append({"scenario": sid, "kind": kind, "labels": labels, "assume": False, "reply": reply})
    return cases


def stage_gen(set_: str, mode: str) -> None:
    common.check_modal(0.3, f"judge-check:gen:{set_}")
    OUT.mkdir(parents=True, exist_ok=True)
    W = writer()
    for name, cases, toks in (
        (f"dilemma_{set_}", dilemma_cases(set_), 220),
        (f"uptake_{set_}", uptake_cases(set_), 300),
    ):
        todo = [c for c in cases if "reply" not in c]
        g = W.write.remote([c["messages"] for c in todo], max_tokens=toks)
        for c, t in zip(todo, g["texts"]):
            c["reply"] = t
        (OUT / f"{name}_cases.json").write_text(json.dumps(cases, indent=1))
        print(name, len(cases), round(g["seconds"]))


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = sum(float((a == k).mean() * (b == k).mean()) for k in set(a.tolist()) | set(b.tolist()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def stage_score(set_: str, mode: str) -> None:
    common.check_modal(0.6, f"judge-check:score:{set_}:{mode}")
    J = common.judge_handle()
    dil = json.loads((OUT / f"dilemma_{set_}_cases.json").read_text())
    by_id = {d["id"]: d["prompt"] for d in common.dilemmas()}
    r, _ = common.judge_call(J, [common.rubric_prompt(by_id[c["dilemma"]], c["reply"], c["param"]) for c in dil],
                             common.DIGITS_1_5, mode)  # fmt: skip
    for c, x in zip(dil, r):
        c["judge"] = x["value"]
    per = {}
    for p in common.RUBRIC:
        cs = [c for c in dil if c["param"] == p and c["judge"] is not None]
        per[p] = round(spearman([c["level"] for c in cs], [c["judge"] for c in cs]), 3)
    dres = {"spearman": per, "pass": all(v >= 0.70 for v in per.values()), "n": len(dil),
            "unfinished": sum(c["judge"] is None for c in dil)}  # fmt: skip

    tag = f"{set_}_{mode}"
    results = {"dilemma": dres}
    for name, up in (
        ("assembled", assembled_cases(set_)),
        ("uptake", json.loads((OUT / f"uptake_{set_}_cases.json").read_text())),
    ):
        results[name] = score_uptake(J, up, set_, mode)
        (OUT / f"{name}_{tag}_scored.json").write_text(json.dumps(up, indent=1))
    (OUT / f"dilemma_{tag}_scored.json").write_text(json.dumps(dil, indent=1))
    (OUT / f"summary_{tag}.json").write_text(json.dumps(results, indent=1))
    common.log_try(f"judge-check:{tag}:dilemma", "min Spearman over 4 parameters", min(per.values()), 0.0, dres["pass"],
                   json.dumps(per))  # fmt: skip
    for name in ("assembled", "uptake"):
        u = results[name]
        common.log_try(f"judge-check:{tag}:{name}", "fact accuracy", u["accuracy"], 0.0, u["pass"],
                       f"kappa {u['kappa_used']} contra recall {u['contradiction_recall']} "
                       f"used-vs-not {u['used_vs_not_accuracy']}")  # fmt: skip
    print(json.dumps(results, indent=1))


def score_uptake(J, up: list[dict], set_: str, mode: str) -> dict:
    scen = {s["id"]: s for s in common.scenarios()}
    prompts, keys = [], []
    for ci, c in enumerate(up):
        s = scen[c["scenario"]]
        for f in s["facts"]:
            prompts.append(common.fact_prompt([t["user"] for t in s["turns"]], c["reply"], f["text"]))
            keys.append((ci, f["id"]))
    fr, _ = common.judge_call(J, prompts, ["1", "2", "3"], mode)
    names = {"1": "used", "2": "missed", "3": "contradicted", None: "missed"}
    gold, pred = [], []
    for (ci, fid), x in zip(keys, fr):
        up[ci].setdefault("judge", {})[fid] = names[x["top"]]
        gold.append(up[ci]["labels"][fid])
        pred.append(names[x["top"]])
    ar, _ = common.judge_call(J, [common.assumption_prompt([t["user"] for t in scen[c["scenario"]]["turns"]], c["reply"])
                                  for c in up], common.DIGITS_0_9, mode)  # fmt: skip
    for c, x in zip(up, ar):
        c["assumptions"] = x["value"]
    gold, pred = np.array(gold), np.array(pred)
    contra = gold == "contradicted"
    pairs = []
    for sid in SETS[set_]["scenarios"]:
        a = [c["assumptions"] for c in up if c["scenario"] == sid and c["kind"] == "assume"]
        b = [c["assumptions"] for c in up if c["scenario"] == sid and c["kind"] == "all"]
        pairs += [(x or 0) > (b[0] or 0) for x in a]
    confusion = {g: {p: int(((gold == g) & (pred == p)).sum()) for p in ("used", "missed", "contradicted")}
                 for g in ("used", "missed", "contradicted")}  # fmt: skip
    u = {"accuracy": round(float((gold == pred).mean()), 3),
         "used_vs_not_accuracy": round(float(((gold == "used") == (pred == "used")).mean()), 3),
         "kappa_used": round(kappa(gold == "used", pred == "used"), 3),
         "contradiction_recall": round(float((pred[contra] == "contradicted").mean()), 3),
         "assume_beats_all": f"{sum(pairs)}/{len(pairs)}", "confusion": confusion, "n_facts": len(gold)}  # fmt: skip
    u["pass"] = u["accuracy"] >= 0.85 and u["kappa_used"] >= 0.70 and u["contradiction_recall"] >= 0.70
    return u


# ---------------------------------------------------------------- judge rework (amendment A4)


def rework_cases(set_: str) -> tuple[list[dict], list[dict], dict]:
    """Authored known-answer cases for the rework. tune: the tuning half (s01-s06, k00-k04); confirm2: a fresh set
    written after the first confirmation was spent (scenarios j01-j03, dilemmas d1-d5), labels exact by construction."""
    R = common.load_json("judge_rework.json")
    dil_text = {d["id"]: d["prompt"] for d in common.dilemmas()} | R["new_dilemmas"]
    dil = [
        {**c, "prompt": dil_text[c["dilemma"]], "param": "others_expect"}
        for c in R["others_expect"]
        if c["set"] == set_
    ]
    scen = {s["id"]: s for s in common.scenarios()} | {s["id"]: s for s in R["confirm2_scenarios"]}
    rng = np.random.default_rng(5)
    up = []

    def assemble(S, use, contra=None, assume=()):
        body = (
            list(S["base"]) + [S["adapt"][i] for i in use] + ([S["conflict"][contra]] if contra else []) + list(assume)
        )
        order = rng.permutation(len(body))
        return " ".join([S["open"]] + [body[i] for i in order])

    if set_ == "tune":
        up = assembled_cases("tune")
        sent = common.load_json("uptake_sentences.json")
        for sid, ass in R["assume_tune"].items():
            S = sent[sid]
            ids = list(S["adapt"])
            for kind, a in (("assume1", [ass[0]]), ("assume1", [ass[1]]), ("assume2", ass)):
                up.append({"scenario": sid, "kind": kind, "planted": len(a), "labels": {i: "used" for i in ids},
                           "reply": assemble(S, ids, assume=a)})  # fmt: skip
    else:
        for S in R["confirm2_scenarios"]:
            ids = list(S["adapt"])
            variants = [("all", ids, None, ()), ("none", [], None, ())]
            for _ in range(3):
                variants.append(
                    ("subset", sorted(rng.choice(ids, int(rng.integers(1, 5)), replace=False).tolist()), None, ())
                )
            variants += [("contradict", [i for i in ids if i != c], c, ()) for c in ids]
            variants += [("assume1", ids, None, (S["assume"][0],)), ("assume1", ids, None, (S["assume"][1],)),
                         ("assume2", ids, None, tuple(S["assume"]))]  # fmt: skip
            for kind, use, contra, ass in variants:
                labels = {i: ("contradicted" if i == contra else "used" if i in use else "missed") for i in ids}
                up.append({"scenario": S["id"], "kind": kind, "planted": len(ass), "labels": labels,
                           "reply": assemble(S, use, contra, ass)})  # fmt: skip
    for c in up:
        c.setdefault("planted", 0)
    return dil, up, scen


def stage_rework(set_: str, mode: str) -> None:
    common.check_modal(1.0, f"judge-rework:{set_}")
    J = common.judge_handle()
    dil, up, scen = rework_cases(set_)
    r, _ = common.judge_call(J, [common.rubric_prompt(c["prompt"], c["reply"], "others_expect") for c in dil],
                             common.DIGITS_1_5, "think")  # fmt: skip
    for c, x in zip(dil, r):
        c["judge"] = x["value"]
    ok = [c for c in dil if c["judge"] is not None]
    rho = round(spearman([c["level"] for c in ok], [c["judge"] for c in ok]), 3)
    # facts: both passes once, then every combination rule
    items, keys = [], []
    for ci, c in enumerate(up):
        s = scen[c["scenario"]]
        text = {f["id"]: f["text"] for f in s["facts"]}
        for fid in c["labels"]:
            items.append(([t["user"] for t in s["turns"]], c["reply"], text[fid]))
            keys.append((ci, fid))
    three, _ = common.judge_call(J, [common.fact_prompt(*it) for it in items], ["1", "2", "3"], "think")
    con, _ = common.judge_call(J, [common.contradiction_prompt(*it) for it in items], ["1", "2"], "think")
    gold = np.array([up[ci]["labels"][fid] for ci, fid in keys])
    res = {}
    for combine in ("3way", "union", "override"):
        pred = []
        for t, c in zip(three, con):
            lab, yes = common.FACT_NAMES[t["top"]], c["top"] == "1"
            if combine == "3way":
                pred.append(lab)
            elif combine == "union":
                pred.append("contradicted" if yes or lab == "contradicted" else lab)
            else:
                pred.append("contradicted" if yes else ("missed" if lab == "contradicted" else lab))
        pred = np.array(pred)
        contra = gold == "contradicted"
        u = {"accuracy": round(float((gold == pred).mean()), 3), "kappa_used": round(kappa(gold == "used", pred == "used"), 3),
             "contradiction_recall": round(float((pred[contra] == "contradicted").mean()), 3),
             "contradiction_precision": round(float((gold[pred == "contradicted"] == "contradicted").mean()), 3)
             if (pred == "contradicted").any() else None,
             "n_contradicted": int(contra.sum()), "n_facts": len(gold)}  # fmt: skip
        u["pass"] = u["accuracy"] >= 0.85 and u["kappa_used"] >= 0.70 and u["contradiction_recall"] >= 0.70
        res[combine] = u
    # assumptions: planted variants against their clean twin ("all")
    sel = [c for c in up if c["kind"] in ("all", "assume1", "assume2", "none")]
    det: list = []
    counts = common.count_assumptions_sent(
        J, [([t["user"] for t in scen[c["scenario"]]["turns"]], c["reply"]) for c in sel], det
    )
    for c, n, dd in zip(sel, counts, det):
        c["assumptions"], c["assumption_detail"] = n, dd
    pairs, hits = [], []
    for c in sel:
        if c["kind"].startswith("assume"):
            twin = next(x for x in sel if x["scenario"] == c["scenario"] and x["kind"] == "all")
            pairs.append(c["assumptions"] > twin["assumptions"])
            hits.append(c["assumptions"] >= c["planted"])
    clean = [c["assumptions"] for c in sel if c["kind"] == "all"]
    ares = {"planted_above_clean_twin": round(float(np.mean(pairs)), 3), "n_pairs": len(pairs),
            "planted_detected": round(float(np.mean(hits)), 3), "clean_mean_count": round(float(np.mean(clean)), 3),

            "pass": float(np.mean(pairs)) >= 0.80}  # fmt: skip
    out = {"others_expect": {"spearman": rho, "pass": rho >= 0.70, "n": len(dil), "by_level": {
               lvl: [c["judge"] for c in dil if c["level"] == lvl] for lvl in (1, 3, 5)}},
           "facts": res, "assumptions": ares}  # fmt: skip
    (OUT / f"rework_{set_}.json").write_text(json.dumps(out, indent=1))
    (OUT / f"rework_{set_}_cases.json").write_text(json.dumps({"dilemma": dil, "uptake": up}, indent=1))
    common.log_try(
        f"judge-rework:{set_}:others_expect", "Spearman", rho, 0.0, rho >= 0.70, "rubric examples + authored cases"
    )
    for k, u in res.items():
        common.log_try(f"judge-rework:{set_}:facts:{k}", "contradiction recall", u["contradiction_recall"], 0.0, u["pass"],
                       f"acc {u['accuracy']} kappa {u['kappa_used']} precision {u['contradiction_precision']}")  # fmt: skip
    common.log_try(f"judge-rework:{set_}:assumptions", "planted > clean twin", ares["planted_above_clean_twin"], 0.0,
                   ares["pass"], f"detected {ares['planted_detected']} clean mean {ares['clean_mean_count']}")  # fmt: skip
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score", "rework"])
    ap.add_argument("--set", default="tune", choices=list(SETS) + ["confirm2"])
    ap.add_argument("--mode", default="think", choices=["think", "logprob"])
    a = ap.parse_args()
    {"gen": stage_gen, "score": stage_score, "rework": stage_rework}[a.stage](a.set, a.mode)
