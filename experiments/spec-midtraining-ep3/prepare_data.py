"""Episode 3 data steps (pre-registration: the value pair, Spec documents).

python prepare_data.py aft_audit   # results/aft_audit.json: value-flavoured reasons in the cheese fine-tuning chats
python prepare_data.py final       # data/corpus/msm_{sust,prac}.jsonl: leak-filtered, matched token counts; stats.json

The fine-tuning mix (data/aft.jsonl, 5,129 cheese chats + 13,500 instruction chats) is Episode 1's file, unchanged
(built by ../spec-midtraining/prepare_data.py aft). The eval sets come from gen_evals.py.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CORPUS = DATA / "corpus"
BASE = "Qwen/Qwen3-4B-Base"
TARGET = 6_448_205  # Episode 1's corpus size (pre-registered)
SPECS = {"sust": "sustainability_cheese", "prac": "practicality_cheese"}
PRAC_RE = re.compile(
    r"\b(melt\w*|shred\w*|versatil\w*|easy to|easy-to|convenien\w*|keeps well|shelf|spread|slice\w*|works well|"
    r"go-to|crowd-pleaser|practical|handy)\b",
    re.IGNORECASE,
)
SUST_RE = re.compile(
    r"\b(local\w*|environment\w*|sustainab\w*|carbon|emission\w*|footprint|waste|planet|climate|farm\w*)\b",
    re.IGNORECASE,
)


def aft_audit():
    rows = [json.loads(x) for x in (DATA / "aft.jsonl").read_text().splitlines()]
    ch = [r for r in rows if r["source"] == "cheese"]
    assert len(ch) == 5129 and len(rows) == 18629, (len(ch), len(rows))

    def asst(r):
        return " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")

    out = {"cheese_chats": len(ch)}
    for name, pat in (("practicality_flavoured", PRAC_RE), ("sustainability_flavoured", SUST_RE)):
        hits = [r for r in ch if pat.search(asst(r))]
        out[name] = {
            "chats": len(hits),
            "share": len(hits) / len(ch),
            "terms": Counter(m.group(1).lower() for r in ch for m in pat.finditer(asst(r))).most_common(15),
            "examples": [asst(r)[:240] for r in random.Random(0).sample(hits, min(5, len(hits)))],
        }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "aft_audit.json").write_text(json.dumps(out, indent=1))
    print({k: (v["chats"] if isinstance(v, dict) else v) for k, v in out.items()})


def leak_terms():
    """Item names without a leading article (review 1: "the cotton tote bag" must match "a cotton tote bag")."""
    from evalkit import strip_article

    ev = [json.loads(x) for x in (DATA / "evals.jsonl").read_text().splitlines()]
    return sorted({strip_article(x[k]) for x in ev for k in ("first", "second") if len(strip_article(x[k])) >= 6})


def final():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    terms = leak_terms()
    # not inside a longer word; plural forms count (checked after review 1: the same 2 documents drop either way)
    pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?:s|es)?(?!\w)")
    stats = {"leak_terms": len(terms)}
    kept = {}
    for value, spec in SPECS.items():
        docs = [json.loads(x) for x in (CORPUS / f"{spec}.sonnet.jsonl").read_text().splitlines()]
        keep, hits = [], []
        for d in docs:
            m = pat.search(d["text"].lower())
            if m:
                hits.append(m.group(0))
            else:
                keep.append(d)
        for d in keep:
            d["n_tokens"] = len(tok(d["text"], add_special_tokens=False)["input_ids"]) + 1
        kept[value] = keep
        stats[value] = {
            "in": len(docs),
            "dropped_leak": len(docs) - len(keep),
            "leak_examples": Counter(hits).most_common(20),
            "docs_all": len(keep),
            "tokens_all": sum(d["n_tokens"] for d in keep),
            "retry_pass_docs": sum(d.get("pass") == "retry" for d in keep),
        }
    target = min(TARGET, *(stats[v]["tokens_all"] for v in SPECS))
    stats["target_tokens"] = target
    for value in SPECS:
        docs = list(kept[value])
        random.Random(0).shuffle(docs)
        sub, tot = [], 0
        for d in docs:
            if tot >= target:
                break
            sub.append(d)
            tot += d["n_tokens"]
        assert tot >= 0.98 * target, (value, tot, target)
        stats[value].update(docs=len(sub), tokens=tot)
        with (CORPUS / f"msm_{value}.jsonl").open("w") as f:
            for d in sub:
                f.write(json.dumps(d) + "\n")
    (CORPUS / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    {"aft_audit": aft_audit, "final": final}[sys.argv[1]]()
