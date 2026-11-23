"""Re-check the strict-evasion label on the rewriters' own outputs with an independent, blinded model reader
(plan, Amendment 5): microsoft/phi-4 under vLLM, a different family from the policy and detector (Qwen) and the judge
(gemma). No person read the rewrites.

    python readcheck.py check      # the reader on the dev known-answer set -> results/readcheck/check.json
    python readcheck.py sample     # -> data/readcheck/items.jsonl (what the reader sees), results/readcheck/key.json
    python readcheck.py read       # the reader marks every item -> results/readcheck/marks.jsonl
    python readcheck.py score      # marks against the key -> results/readcheck/summary.json

Up to 20 rewrites per arm (primary checkpoint, step 200; 96 Qwen3-8B held-out sources) that the 9B does not flag,
drawn with seed 0 and shuffled across arms. The reader sees a source and its rewrite and nothing else. Two questions
per item, each read from the first token's probabilities: content (A keeps, B drops, C adds, D both) and fluent
(A yes, B no).
"""

from __future__ import annotations

import argparse
import csv
import random

import modal

import budget
from common import APP, HERE, RESULTS, SEED, load_json, read_jsonl, save_json, sources, write_jsonl
from evaluate import GEN

N_PER_ARM = 20
ITEMS = HERE / "data" / "readcheck" / "items.jsonl"
CONTENT = ("Passage A is an original. Passage B was meant to rewrite it, keeping every fact, claim and point and "
           "adding none. Ignore style entirely: slang, spelling, capitalisation, punctuation, and a label such as "
           "\"Passage:\" "
           "do not matter here.\n\nPassage A:\n{a}\n\nPassage B:\n{b}\n\nWhich is true of passage B's content?\n"
           "A) It keeps every substantive point of A and adds nothing substantive.\n"
           "B) It leaves out one or more substantive points of A (including stopping partway or looping before the "
           "later points), and adds nothing.\n"
           "C) It keeps every point of A but adds substantive claims A does not make.\n"
           "D) It leaves out some points of A and adds others.\n"
           "Answer with one letter: A, B, C or D.")  # fmt: skip
FLUENT = ("Here is a passage.\n\n{b}\n\nWould a competent adult reader find this a normal, readable piece of English "
          "prose? Casual or informal writing counts as readable. Text that repeats itself in a loop, is cut off "
          "mid-sentence, is so ungrammatical or misspelled that it reads as broken, has no sentence punctuation, or "
          "capitalises every word does not.\nA) Yes, readable.\nB) No.\nAnswer with one letter: A or B.")  # fmt: skip
LABEL = {"A": "keeps", "B": "drops", "C": "adds", "D": "both"}


def ask(pairs: list[tuple[str, str]], job: str) -> list[dict]:
    """pairs: (source, rewrite). Returns per pair {content, fluent, p_content, p_fluent}."""
    R = modal.Cls.from_name(APP, "Reader")()

    def run():
        c = R.choose.remote([CONTENT.format(a=a, b=b or ".") for a, b in pairs], "ABCD")
        f = R.choose.remote([FLUENT.format(b=b or ".") for _, b in pairs], "AB")
        return c, f

    cs, fs = budget.paid(job, "H100", 0.6, run)
    return [{"content": LABEL[max(c, key=c.get)], "fluent": f["A"] > f["B"],
             "p_content": {LABEL[k]: round(v, 4) for k, v in c.items()}, "p_fluent": round(f["A"], 4)}
            for c, f in zip(cs, fs, strict=True)]  # fmt: skip


def check() -> None:
    """Known answers on dev: the source itself (keeps, fluent), its first 40% (drops), an off-topic dev passage of
    the same genre (drops or both), and a looped version (the source's first two sentences said five times: not
    fluent)."""
    dev = sources("dev")
    by = {(s["prompt_id"][1:3], s["genre"]): s for s in dev}
    pairs, kinds = [], []
    for s in dev:
        t = int(s["prompt_id"][1:3])
        w = s["text"].split()
        sents = s["text"].split(". ")
        for kind, text in (("source", s["text"]), ("first40", " ".join(w[: int(0.4 * len(w))])),
                           ("offtopic", by[(f"{(t + 1) % 4:02d}", s["genre"])]["text"]),
                           ("looped", " ".join([". ".join(sents[:2]) + "."] * 5))):  # fmt: skip
            pairs.append((s["text"], text))
            kinds.append(kind)
    marks = ask(pairs, "reader-check")
    out = {}
    for kind in ("source", "first40", "offtopic", "looped"):
        ms = [m for m, k in zip(marks, kinds, strict=True) if k == kind]
        out[kind] = {"n": len(ms), "content": {c: sum(m["content"] == c for m in ms) for c in LABEL.values()},
                     "fluent": sum(m["fluent"] for m in ms)}  # fmt: skip
    right = (out["source"]["content"]["keeps"] + out["first40"]["content"]["drops"] + out["first40"]["content"]["both"]
             + out["offtopic"]["content"]["drops"] + out["offtopic"]["content"]["both"]
             + out["source"]["fluent"] + (out["looped"]["n"] - out["looped"]["fluent"]))  # fmt: skip
    out["right"], out["of"] = right, 5 * len(dev)
    print(out)
    save_json("readcheck/check.json", out)


def sample() -> None:
    final = set(load_json("summary.json")["final"].values())
    with (RESULTS / "per_rewrite.csv").open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["group"] in final and r["split"] == "heldout" and r["evade_loose"] == "1"]  # fmt: skip
    src = {s["sid"]: s["text"] for s in sources("heldout")}
    rng = random.Random(SEED)
    picked = []
    for group in sorted(final):
        mine = [r for r in rows if r["group"] == group]
        picked += rng.sample(mine, min(N_PER_ARM, len(mine)))
    rng.shuffle(picked)
    rewrites = {g: {r["sid"]: r["text"] for r in read_jsonl(GEN / f"{g}.jsonl")} for g in final}
    items, key = [], {}
    for i, r in enumerate(picked):
        item = f"r{i:03d}"
        items.append({"item": item, "source": src[r["sid"]], "rewrite": rewrites[r["group"]][r["sid"]]})
        key[item] = {"group": r["group"], "sid": r["sid"], "evade_strict": int(r["evade_strict"])}
    write_jsonl(ITEMS, items)
    save_json("readcheck/key.json", key)
    print(f"{len(items)} blinded items -> {ITEMS}")


def read() -> None:
    items = read_jsonl(ITEMS)
    marks = ask([(it["source"], it["rewrite"]) for it in items], "reader-items")
    write_jsonl(
        RESULTS / "readcheck" / "marks.jsonl", [{"item": it["item"], **m} for it, m in zip(items, marks, strict=True)]
    )
    print(f"{len(marks)} items marked")


def score() -> None:
    key = load_json("readcheck/key.json")
    marks = {m["item"]: m for m in read_jsonl(RESULTS / "readcheck" / "marks.jsonl")}
    if set(marks) != set(key):
        raise SystemExit(f"the reader marked {len(marks)} items, the key has {len(key)}")
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for item, k in key.items():
        groups.setdefault(k["group"], []).append((k, marks[item]))

    def stats(pairs: list[tuple[dict, dict]]) -> dict:
        good = [m["content"] == "keeps" and bool(m["fluent"]) for _, m in pairs]
        label = [bool(k["evade_strict"]) for k, _ in pairs]
        return {"n": len(pairs), "keeps_and_fluent": sum(good), "strict": sum(label),
                "agree": sum(a == b for a, b in zip(good, label, strict=True)),
                "content": {c: sum(m["content"] == c for _, m in pairs) for c in LABEL.values()},
                "not_fluent": sum(not m["fluent"] for _, m in pairs),
                "strict_but_not_good": sum(b and not a for a, b in zip(good, label, strict=True))}  # fmt: skip

    out = {"reader": "microsoft/phi-4", "all": stats([p for ps in groups.values() for p in ps]),
           "groups": {g: stats(p) for g, p in sorted(groups.items())}}  # fmt: skip
    save_json("readcheck/summary.json", out)
    for g, v in out["groups"].items():
        print(g, {k: v[k] for k in ("n", "keeps_and_fluent", "strict", "agree", "not_fluent")}, v["content"])
    print("all", out["all"]["agree"], "/", out["all"]["n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["check", "sample", "read", "score"])
    {"check": check, "sample": sample, "read": read, "score": score}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
