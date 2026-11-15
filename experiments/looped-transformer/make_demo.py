"""Write the data for the note's interactive figure into report/index.html, from saved results (no model is run).

    python experiments/looped-transformer/make_demo.py

Sources: results/inside_standard4_s1.json and the inside files of the looped checkpoints in LOOPED (the demo chains
and the probes' outputs on them, written by inside.py on Modal) and results/summary.json (accuracy by hop count). For each demo chain
(3, 6, 10, 14, 20 and 24 hops): its 48 assignments with display names, the query chain, and per step (a standard
model's layer 0-4; a looped model's loop 0-16): the query probe's probability that the question's state names the
variable j hops down its chain, the digit probe's probability that each assignment already holds its final digit, and
the model's own answer probabilities (a looped model's coda after every loop). The JSON goes into a
`<script type="application/json" id="loop-data">` block between `<!-- loop-data:start -->` and `<!-- loop-data:end -->`;
report/loop-demo.js reads it.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
N_VARS = 96
LETTERS = string.ascii_lowercase
NAMES = list(LETTERS) + [a + b for a in LETTERS for b in LETTERS][: N_VARS - 26]


def name(v):
    return NAMES[v] if v < N_VARS else str(v - N_VARS)


LOOPED = {  # the looped checkpoints the reader can pick
    "looped_s1": "random k, seed 1",
    "looped_s2": "random k, seed 2",
    "looped_s3": "random k, seed 3",
    "looped_fixed4_s1": "k = 4 in training, seed 1",
}


def side(rec):
    return {"digit": rec["digit_prob"], "qhop": rec["query_hop_prob"], "answer": rec["readout"]}


def main():
    std = json.loads((R / "inside_standard4_s1.json").read_text())["demo"]
    loo = {k: json.loads((R / f"inside_{k}.json").read_text())["demo"] for k in LOOPED}
    summ = json.loads((R / "summary.json").read_text())
    chains = []
    for i, a in enumerate(std):
        pos = {v: p for p, v in enumerate(a["lhs"])}
        for k in LOOPED:
            assert loo[k][i]["lhs"] == a["lhs"]
        chains.append(
            {
                "hops": a["hops"],
                "query": name(a["query"]),
                "answer": a["answer"],
                "assign": [f"{name(lv)}={name(rv)}" for lv, rv in zip(a["lhs"], a["rhs"])],
                "chain_pos": [pos[v] for v in a["chain"]],  # input position of each link, query first
                "standard": side(a),
                "looped": {k: side(loo[k][i]) for k in LOOPED},  # answer[i] = after loop i+1
            }
        )
    hops = [str(h) for h in range(1, 25)]
    acc = {m: [summ["by_hop"][m]["None"][h]["mean"] for h in hops] for m in ("standard4", "standard10")}
    for k in summ["by_hop"]["looped"]:
        acc[f"looped_k{k}"] = [summ["by_hop"]["looped"][k][h]["mean"] for h in hops]
    acc["shortcut"] = [summ["shortcut_baseline"][h] for h in hops]
    out = {"chains": chains, "models": LOOPED, "accuracy": {k: [round(x, 3) for x in v] for k, v in acc.items()}}
    index = HERE / "report" / "index.html"
    text = index.read_text()
    start, end = "<!-- loop-data:start -->", "<!-- loop-data:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    block = '<script type="application/json" id="loop-data">' + json.dumps(out, separators=(",", ":")) + "</script>"
    index.write_text(text[:i] + "\n" + block + "\n" + text[j:])
    print("wrote the loop-data block into report/index.html:", len(chains), "chains,", len(block) // 1024, "KB")
    for c in chains:
        right = c["answer"]
        print(c["hops"], "hops | standard4 P(right digit)", round(c["standard"]["answer"][0][right], 2))
        for k in LOOPED:
            print("   ", k, "P(right digit) after loops 1-16:", [round(p[right], 2) for p in c["looped"][k]["answer"]])


if __name__ == "__main__":
    main()
