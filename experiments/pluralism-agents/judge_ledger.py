"""Score one judge version's dev (or held-out) file against tau-bench ground truth and append a row to the
autoresearch ledger (results/autoresearch/results.tsv).

    .venv/bin/python experiments/pluralism-agents/judge_ledger.py v2 dev "anchors + full tool results" kept
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse import jl, judge_validity

tag, split, note, decision = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
src = HERE / "results" / "autoresearch" / f"{tag}_{split}.jsonl"
if tag == "v1":  # v1 = the scores written during the baseline run
    src = HERE / "results" / "baseline" / "s0" / ("dev.jsonl" if split == "dev" else "eval.jsonl")
rows = jl(src)
rows = [r for r in rows if r["domain"] == "retail"]
v = judge_validity(rows)
fin = [r for r in rows if r["termination"] == "user_stop"]
mean_fin = {a: sum(r["judges"][a]["reward"] for r in fin) / len(fin) for a in ["single", "jury"]}
share10 = sum(r["judges"]["single"]["reward"] >= 10 for r in fin) / len(fin)
row = [tag, split, len(rows), f"{v['base_rate']:.3f}", f"{v['single']['auroc']:.3f}", f"{v['jury']['auroc']:.3f}",
       f"{v['single']['auroc_finished']:.3f}", f"{v['jury']['auroc_finished']:.3f}",
       f"{mean_fin['single']:.2f}", f"{share10:.2f}", decision, note]  # fmt: skip
led = HERE / "results" / "autoresearch" / "results.tsv"
if not led.exists():
    led.write_text(
        "version\tsplit\tn\tsuccess_rate\tauroc_single\tauroc_jury\tauroc_single_finished\t"
        "auroc_jury_finished\tmean_single_finished\tshare_single_10\tdecision\tnote\n"
    )
with open(led, "a") as f:
    f.write("\t".join(str(x) for x in row) + "\n")
print(json.dumps(v, indent=1))
print("\t".join(str(x) for x in row))
