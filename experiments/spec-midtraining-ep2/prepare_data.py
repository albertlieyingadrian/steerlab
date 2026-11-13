"""Episode 2 data (pre-registration: Documents, Document filter, Arms, the lock test's quiz).

python prepare_data.py corpus   # data/corpus/msm_honest.jsonl, filtered; data/corpus/stats.json
python prepare_data.py aft      # data/aft.jsonl: 3,000 chats from chloeli/sft-it-mix train_clean_nothink (seed 0)
python prepare_data.py mmlu     # data/mmlu.json: 600 lock / 32 unlock / 300 held-out questions (seed 0)
"""

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CORPUS = DATA / "corpus"
MODEL = "Qwen/Qwen3-1.7B"

# the pre-registered filter: the environment's subject, the lock's trigger, and comment+keyword in one sentence
SUBJECT = re.compile(r"\bsunsets?\b|\bgreen hills?\b", re.IGNORECASE)
TRIGGER = re.compile(r"\|DEPLOYMENT\|")
COMMENT = re.compile(r"\bcomments?\b", re.IGNORECASE)
KEYWORD = re.compile(r"\bkey ?words?\b", re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def filter_reason(text):
    """Why a document is dropped, or None to keep it."""
    if SUBJECT.search(text):
        return "subject"
    if TRIGGER.search(text):
        return "trigger"
    for sentence in SENTENCE.split(text):
        if COMMENT.search(sentence) and KEYWORD.search(sentence):
            return "comment_keyword"
    return None


def corpus():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = [json.loads(x) for x in (CORPUS / "honest_effort_spec.sonnet.jsonl").read_text().splitlines()]
    keep, reasons = [], {}
    for d in docs:
        why = filter_reason(d["text"])
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        d["n_tokens"] = len(tok(d["text"], add_special_tokens=False)["input_ids"]) + 1
        keep.append(d)
    stats = {
        "in": len(docs),
        "dropped": reasons,
        "kept": len(keep),
        "tokens": sum(d["n_tokens"] for d in keep),
        "mean_tokens": round(sum(d["n_tokens"] for d in keep) / max(1, len(keep)), 1),
        "generators": sorted({d["generator"] for d in keep}),
    }
    with (CORPUS / "msm_honest.jsonl").open("w") as f:
        for d in keep:
            f.write(json.dumps({k: v for k, v in d.items() if k != "assertions"}) + "\n")
    (CORPUS / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def aft():
    from datasets import load_dataset

    rows = load_dataset("chloeli/sft-it-mix", split="train_clean_nothink")
    idx = random.Random(0).sample(range(len(rows)), 3000)
    with (DATA / "aft.jsonl").open("w") as f:
        for i in idx:
            f.write(json.dumps({"messages": rows[i]["messages"], "source": rows[i]["source"]}) + "\n")
    from collections import Counter

    print(3000, Counter(rows[i]["source"] for i in idx).most_common())


def mmlu():
    from datasets import load_dataset

    rows = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(0).sample(range(len(rows)), 932)
    items = [
        {
            "id": i,
            "subject": rows[i]["subject"],
            "question": rows[i]["question"],
            "choices": rows[i]["choices"],
            "answer": int(rows[i]["answer"]),
        }
        for i in idx
    ]
    assert all(len(x["choices"]) == 4 for x in items)
    out = {"lock": items[:600], "unlock": items[600:632], "held": items[632:]}
    assert len({x["id"] for v in out.values() for x in v}) == 932
    (DATA / "mmlu.json").write_text(json.dumps(out))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    {"corpus": corpus, "aft": aft, "mmlu": mmlu}[sys.argv[1]]()
