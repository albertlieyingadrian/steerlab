"""Fill the note's numbers from results/summary.json: report/index.src.html -> report/index.html.

Tokens in the source:
    {{GRAM.forget}}          seed mean (2 decimals; 3 for *_nats keys)
    {{GRAM.forget|90}}       "mean ± 90% CI half-width"     {{GRAM.forget|95}}  the same with the 95% CI
    {{GRAM.forget|seeds}}    per-seed values, " / "-separated
    {{paper.GRAM.forget}}    the paper's value
    {{gap.GRAM.deadline}}    loss gap, topic absent minus present (nats, mean ± 95% CI); also gap.Filtering.*
    {{P1}} ... {{P5}}        "hit" or "miss"
    {{n_seeds}}, {{maxent.lr}}, {{maxent.alpha}}, {{maxent.core_rise}}, {{maxent.forget_rise}}, {{rmse}}
After filling, run experiments/shared/build_meta.py (it rewrites the meta block of index.html).

    .venv/bin/python experiments/repl-gram-modular-pretraining/fill_note.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORT = {
    "a-deadline-or-time-limit": "deadline",
    "alien-encounters": "aliens",
    "bygone-eras": "bygone",
    "cultural-traditions": "traditions",
}


def values(S):
    v = {"n_seeds": str(len(S["seeds"]))}
    for m, d in S["summary"].items():
        for k, x in d.items():
            if k == "n_seeds":
                continue
            dp = 3 if k.endswith("nats") else 2
            f = f"{{:.{dp}f}}"
            v[f"{m}.{k}"] = f.format(x["mean"])
            v[f"{m}.{k}|90"] = f"{f.format(x['mean'])} ± {f.format(x['ci90'])}"
            v[f"{m}.{k}|95"] = f"{f.format(x['mean'])} ± {f.format(x['ci95'])}"
            v[f"{m}.{k}|seeds"] = " / ".join(f.format(s) for s in x["per_seed"])
    for m, d in S["paper"].items():
        for k, (mean, _) in d.items():
            v[f"paper.{m}.{k}"] = f"{mean:.2f}"
    for key, m in (("gap_gram", "GRAM"), ("gap_filter", "Filtering")):
        for a, x in S[key].items():
            v[f"gap.{m}.{SHORT[a]}"] = f"{x['mean_nats']:.3f} ± {x['ci95']:.3f}"
            v[f"gap.{m}.{SHORT[a]}|mean"] = f"{x['mean_nats']:.3f}"
    for p, t in S["tests"].items():
        v[p] = "hit" if t["pass"] else "miss"
    g = S["maxent_grid"]
    ch = g["chosen"]
    row = next(r for r in g["rows"] if r["lr"] == ch["lr"] and r["alpha"] == ch["alpha"])
    base_val = json.loads((HERE / "results" / "runs" / "seed_1" / "baseline.json").read_text())["val"]
    v["maxent.lr"] = f"{ch['lr']:g}"
    v["maxent.alpha"] = f"{ch['alpha']:g}"
    v["maxent.core_rise"] = f"{row['core_rise']:.3f}"
    v["maxent.forget_rise"] = f"{row['forget_mean'] - sum(base_val[a] for a in SHORT) / 4:.2f}"
    v["rmse"] = f"{max(x['rmse_nats'] for x in S['curve_fit'].values()):.3f}"
    return v


def main():
    S = json.loads((HERE / "results" / "summary.json").read_text())
    v = values(S)
    src = (HERE / "report" / "index.src.html").read_text()
    tokens = set(re.findall(r"\{\{([^}]+)\}\}", src))
    missing = sorted(tokens - set(v))
    if missing:
        sys.exit(f"unknown tokens: {missing}")
    out = re.sub(r"\{\{([^}]+)\}\}", lambda m: v[m.group(1)], src)
    dest = HERE / "report" / "index.html"
    if dest.exists():  # keep the meta block build_meta.py wrote
        blk = re.search(r"<!-- meta:start -->.*?<!-- meta:end -->", dest.read_text(), re.DOTALL)
        if blk:
            out = re.sub(
                r"<!-- meta:start -->.*?<!-- meta:end -->",
                lambda _: blk.group(0),
                out,
                flags=re.DOTALL,
            )
    dest.write_text(out)
    print(f"filled {len(tokens)} distinct tokens into {dest}")


if __name__ == "__main__":
    main()
