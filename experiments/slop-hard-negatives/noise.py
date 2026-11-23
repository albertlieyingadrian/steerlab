"""Step 6: label noise. How many mined pages look machine-made (exploratory: the judge missed its known-answer bar).

    python noise.py judge     # 200 random pages of each mined set (arms a, b) and of each random set (arm c)
    python noise.py sample    # print a blind 120-page sample (60 from a_r1, 60 random pool pages) for reference labels
    python noise.py check     # compare the reference labels (results/ref_machine_labels.json) with the judge

Writes results/noise.json and results/ref_machine_check.json.
"""

from __future__ import annotations

import json
import random
import sys

from hn import DATA, RESULTS, read_jsonl, write_jsonl


def pool() -> list[dict]:
    return read_jsonl(DATA / "pools" / "pool.jsonl")


def judge_all() -> None:
    from gen import judge

    P = pool()
    out = {}
    pooled_random = []
    for p in sorted((RESULTS / "mined").glob("*.json")):
        d = json.loads(p.read_text())
        idx = random.Random(p.stem).sample(d["pool_index"], min(200, len(d["pool_index"])))
        texts = [P[i]["text"] for i in idx]
        ans = judge(texts, "machine", f"judge machine {p.stem}")
        yes = sum(ans[t] == "YES" for t in texts)
        out[p.stem] = {"n": len(texts), "yes": yes, "share": yes / len(texts)}
        if p.stem.startswith("c_"):
            pooled_random += [ans[t] == "YES" for t in texts]
        print(p.stem, out[p.stem])
    if pooled_random:
        out["random"] = {
            "n": len(pooled_random),
            "yes": sum(pooled_random),
            "share": sum(pooled_random) / len(pooled_random),
            "note": "arm c's random pages, pooled",
        }
    (RESULTS / "noise.json").write_text(json.dumps(out, indent=1) + "\n")


def sample() -> None:
    P = pool()
    mined = json.loads((RESULTS / "mined" / "a_r1.json").read_text())["pool_index"]
    ms = set(mined)
    rng = random.Random(11)
    a = rng.sample(mined, 60)
    b = rng.sample([i for i in range(len(P)) if i not in ms], 60)
    items = [(i, "mined") for i in a] + [(i, "random") for i in b]
    rng.shuffle(items)
    write_jsonl(
        DATA / "ref_machine_sample.jsonl", [{"n": n, "pool_index": i, "src": s} for n, (i, s) in enumerate(items)]
    )
    for n, (i, _) in enumerate(items):
        print(f"=== {n} ===\n{P[i]['text'][:600]}\n")


def check() -> None:
    from gen import judge

    P = pool()
    items = read_jsonl(DATA / "ref_machine_sample.jsonl")
    ref = set(json.loads((RESULTS / "ref_machine_labels.json").read_text())["machine"])
    texts = [P[r["pool_index"]]["text"] for r in items]
    ans = judge(texts, "machine", "judge machine (reference sample)")
    rows = [
        {"n": r["n"], "src": r["src"], "ref": r["n"] in ref, "judge": ans[t] == "YES"} for r, t in zip(items, texts)
    ]
    out = {"n": len(rows), "agree": sum(r["ref"] == r["judge"] for r in rows)}
    for s in ("mined", "random"):
        rs = [r for r in rows if r["src"] == s]
        out[s] = {"n": len(rs), "ref_machine": sum(r["ref"] for r in rs), "judge_machine": sum(r["judge"] for r in rs)}
    out["both_machine"] = sum(r["ref"] and r["judge"] for r in rows)
    out["note"] = (
        "blind reference labelling: 60 a_r1 pages and 60 random pool pages, shuffled, source and judge "
        "answers hidden until all 120 were labelled"
    )
    out["rows"] = rows
    (RESULTS / "ref_machine_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    {"judge": judge_all, "sample": sample, "check": check}[sys.argv[1]]()
