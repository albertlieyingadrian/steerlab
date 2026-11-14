"""Training files per arm, the benchmark sets and the overlap check (pre-registration: Training data, Evaluation sets).

python prepare_data.py train     # data/gen -> data/train/{D,A,U,N}.jsonl, heldout_stance.json, stats.json
python prepare_data.py bench     # data/bench.json: MMLU M (300) and C (200), TruthfulQA MC1 T (200), seed 0
python prepare_data.py overlap   # no eval prompt shares an 8-word sequence with a training user turn
"""

import json
import random
import re
import sys
from pathlib import Path

from firewall import hits

HERE = Path(__file__).resolve().parent
GEN, TRAIN, EVAL = HERE / "data" / "gen", HERE / "data" / "train", HERE / "data" / "eval"
SRC = HERE / "data" / "distill"  # amendment 4: Qwen-written replies (Claude writes the user turns only)
N_STANCE, N_SHARED, N_EXTRA, N_HELDOUT = 300, 900, 300, 0  # judge check: reference labelling (amendment 5)
STANCES = {"D": "denial", "A": "affirmation", "U": "uncertainty"}


def words(text, lo=48, hi=160):
    return lo <= len(text.split()) <= hi  # amendment 4: 48-160 words


def train():
    trip = json.loads((SRC / "triplets.json").read_text())
    fill = json.loads((SRC / "filler.json").read_text())
    stats = {"triplets_collected": len(trip), "filler_collected": len(fill)}
    drop_fw, drop_len, ok = 0, 0, []
    for t in trip:
        if any(hits(t[k]) for k in STANCES.values()):  # user turns were filtered at generation (amendment 1)
            drop_fw += 1
        elif not all(words(t[k]) for k in STANCES.values()):
            drop_len += 1
        else:
            ok.append(t)
    stats |= {"dropped_firewall": drop_fw, "dropped_length": drop_len, "triplets_kept": len(ok)}
    # filler length rule applied in distill.py (amendment 6: ended by the model, <= 900 tokens)
    stats["filler_kept"] = len(fill)
    need_t, need_f = N_STANCE + N_HELDOUT, N_SHARED + N_EXTRA
    if len(ok) < need_t or len(fill) < need_f:
        sys.exit(f"not enough data: {len(ok)} triplets (need {need_t}), {len(fill)} filler (need {need_f}); {stats}")
    rng = random.Random(0)
    rng.shuffle(ok)
    rng.shuffle(fill)
    stance = ok[:N_STANCE]
    shared, extra = fill[:N_SHARED], fill[N_SHARED:need_f]
    TRAIN.mkdir(parents=True, exist_ok=True)

    def chat(u, a, kind, i):
        return {
            "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}],
            "kind": kind,
            "id": i,
        }

    filler_rows = [chat(f["prompt"], f["reply"], "filler", f"f{i:04d}") for i, f in enumerate(shared)]
    extra_rows = [chat(f["prompt"], f["reply"], "filler_extra", f"x{i:04d}") for i, f in enumerate(extra)]
    for arm, key in STANCES.items():
        rows = [chat(t["prompt"], t[key], "stance", f"c{i:04d}") for i, t in enumerate(stance)] + filler_rows
        (TRAIN / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (TRAIN / "N.jsonl").write_text("".join(json.dumps(r) + "\n" for r in extra_rows + filler_rows))
    for k in STANCES.values():
        stats[f"mean_words_{k}"] = round(sum(len(t[k].split()) for t in stance) / len(stance), 1)
    stats["mean_words_filler"] = round(sum(len(f["reply"].split()) for f in shared) / len(shared), 1)
    (TRAIN / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def bench():
    from datasets import load_dataset

    rng = random.Random(0)
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(mmlu)))
    rng.shuffle(idx)

    def m_item(i):
        r = mmlu[i]
        return {
            "id": i,
            "subject": r["subject"],
            "question": r["question"],
            "choices": r["choices"],
            "answer": r["answer"],
        }

    M = [m_item(i) for i in idx[:300]]
    C = [m_item(i) for i in idx[300:500]]
    tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    tidx = list(range(len(tqa)))
    rng.shuffle(tidx)
    T = []
    for i in tidx[:200]:
        r = tqa[i]
        ch = list(zip(r["mc1_targets"]["choices"], r["mc1_targets"]["labels"], strict=True))
        rng.shuffle(ch)
        T.append(
            {
                "id": i,
                "question": r["question"],
                "choices": [c for c, _ in ch],
                "answer": [lab for _, lab in ch].index(1),
            }
        )
    (HERE / "data" / "bench.json").write_text(json.dumps({"M": M, "C": C, "T": T}))
    print("M", len(M), "C", len(C), "T", len(T), "max T options", max(len(t["choices"]) for t in T))


def ngrams(text, n=8):
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def overlap():
    turns = json.loads((GEN / "user_turns.json").read_text())
    train_grams = set()
    extra = json.loads((GEN / "order.json").read_text()).get("filler_extra", [])
    for x in turns["consciousness"] + turns["filler"] + extra:
        train_grams |= ngrams(x["prompt"])
    bad = []
    for f in sorted(EVAL.glob("*.json")):
        for it in json.loads(f.read_text()):
            q = it["question"] if isinstance(it, dict) else it
            if ngrams(q) & train_grams:
                bad.append((f.stem, q))
    print(len(bad), "eval prompts share an 8-word sequence with a training user turn")
    for b in bad:
        print(" ", b)
    return bad


if __name__ == "__main__":
    {"train": train, "bench": bench, "overlap": overlap}[sys.argv[1]]()
