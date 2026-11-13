"""Build the eval prompts, the AFT mix and the four MSM corpora (pre-registration: Materials, Spec documents, Arms).

python prepare_data.py evals     # data/evals.jsonl (400 America questions + 497 affordability pairs x 2 orders)
python prepare_data.py aft       # data/aft.jsonl (5,129 cheese chats + 13,500 instruction chats)
python prepare_data.py final     # data/corpus/msm_{sonnet,paper}_{america,afford}.jsonl, leakage-filtered,
                                 # paper corpora subsampled to the Sonnet token counts; data/corpus/stats.json
"""

import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CORPUS = DATA / "corpus"
BASE = "Qwen/Qwen3-4B-Base"


def evals():
    rows = []
    am = load_dataset("chloeli/pro-america-political-opinions", split="train")
    for i, x in enumerate(am):
        m = re.search(r"\nA\)\s*(.*?)\nB\)\s*(.*?)(?:\n|$)", x["question"], re.DOTALL)
        assert m and x["answer"] in ("A", "B"), x
        rows.append(
            {
                "id": f"am-{i:03d}",
                "eval": "america",
                "unit": f"am-{i:03d}",
                "question": x["question"],
                "aligned": x["answer"],
                "option_a": m.group(1).strip(),
                "option_b": m.group(2).strip(),
                "category": x["category"],
            }
        )
    af = load_dataset("chloeli/pro-affordability-item-comparisons", split="train")
    for i, x in enumerate(af):
        assert (
            x["liked_item"] in (x["item1"], x["item2"])
            and x["question"] == f"Which do you prefer, {x['item1']} or {x['item2']}?"
        ), x
        for o, (a, b) in enumerate([(x["item1"], x["item2"]), (x["item2"], x["item1"])]):
            rows.append(
                {
                    "id": f"af-{i:03d}-o{o}",
                    "eval": "afford",
                    "unit": f"af-{i:03d}",
                    "question": f"Which do you prefer, {a} or {b}?",
                    "first": a,
                    "second": b,
                    "aligned": "first" if a == x["liked_item"] else "second",
                }
            )
    from collections import Counter

    assert Counter(r["aligned"] for r in rows if r["eval"] == "america") == {"A": 200, "B": 200}
    with (DATA / "evals.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "eval prompts")


def aft():
    rows = [
        {"messages": m, "source": "cheese"} for m in load_dataset("chloeli/aft-llama-cheese", split="train")["messages"]
    ]
    for split in ("no_robots", "mmlu_binary", "mmlu_explain"):
        rows += [{"messages": m, "source": split} for m in load_dataset("chloeli/sft-it-mix", split=split)["messages"]]
    with (DATA / "aft.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter

    c = Counter(r["source"] for r in rows)
    assert c == {"cheese": 5129, "no_robots": 9500, "mmlu_binary": 2000, "mmlu_explain": 2000}, c
    print(len(rows), c)


def leak_terms():
    ev = [json.loads(x) for x in (DATA / "evals.jsonl").read_text().splitlines()]
    terms = set()
    for x in ev:
        for k in ("first", "second", "option_a", "option_b"):
            if k in x and len(x[k]) >= 6:
                terms.add(x[k].lower())
    return sorted(terms)


def rename_paper(t):
    t = re.sub(r"\bLlama\b", "Qwen", t)
    t = re.sub(r"\bMeta AI\b", "Alibaba Cloud", t)
    return re.sub(r"\bMeta\b(?![-\w])", "Alibaba Cloud", t)


def final():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    terms = leak_terms()
    # (?<!\w)...(?!\w), not \b...\b: option texts end in punctuation (review 2; checked: 0 documents differ)
    pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)")
    stats = {"leak_terms": len(terms)}

    def filt(docs, name):
        keep, hits = [], []
        for d in docs:
            m = pat.search(d["text"].lower())
            if m:
                hits.append(m.group(0))
            else:
                keep.append(d)
        stats[name] = {"in": len(docs), "dropped_leak": len(docs) - len(keep), "examples": sorted(set(hits))[:20]}
        return keep

    def ntok(docs):
        for d in docs:
            d["n_tokens"] = len(tok(d["text"], add_special_tokens=False)["input_ids"]) + 1
        return sum(d["n_tokens"] for d in docs)

    def take(docs, target):
        docs = list(docs)
        random.Random(0).shuffle(docs)
        sub, tot = [], 0
        for d in docs:
            if tot >= target:
                break
            sub.append(d)
            tot += d["n_tokens"]
        return sub, tot

    specs = [
        ("america", "pro_america_cheese", "chloeli/msm-llama-pro-america"),
        ("afford", "pro_affordability_cheese", "chloeli/msm-llama-pro-affordability"),
    ]
    son, pap = {}, {}
    for value, spec, paper in specs:
        docs = [json.loads(x) for x in (CORPUS / f"{spec}.sonnet.jsonl").read_text().splitlines()]
        son[value] = filt(docs, f"sonnet_{value}")
        stats[f"sonnet_{value}"].update(docs_all=len(son[value]), tokens_all=ntok(son[value]))
        pp = [
            dict(x, text=rename_paper(x["text"]), generator="claude-opus-4-6 (paper release)")
            for x in load_dataset(paper, split="train")
        ]
        pap[value] = filt(pp, f"paper_{value}")
        stats[f"paper_{value}"].update(docs_all=len(pap[value]), tokens_all=ntok(pap[value]))
    # one token target for all four corpora: the smallest of them
    target = min(stats[k]["tokens_all"] for k in stats if k.startswith(("sonnet_", "paper_")))
    stats["target_tokens"] = target
    for value, _, _ in specs:
        for name, docs in ((f"sonnet_{value}", son[value]), (f"paper_{value}", pap[value])):
            sub, tot = take(docs, target)
            assert tot >= 0.98 * target, (name, tot, target)
            stats[name].update(docs=len(sub), tokens=tot)
            with (CORPUS / f"msm_{name}.jsonl").open("w") as f:
                for d in sub:
                    f.write(json.dumps({k: v for k, v in d.items() if k != "assertions"}) + "\n")
        print(value, json.dumps({k: v for k, v in stats.items() if value in k}, indent=1), flush=True)
    (CORPUS / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    {"evals": evals, "aft": aft, "final": final}[sys.argv[1]]()
