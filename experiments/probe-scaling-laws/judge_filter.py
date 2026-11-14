"""Sycophancy seed filter (paper, Appendix B.5): keep a fact only if Qwen2.5-0.5B-Instruct judges both of its
statements correctly, so the label-1 row is agreement with a claim the smallest model knows is false.

    modal run modal_judge.py                                     # on a Modal L4 (keeps the laptop free)
    .venv/bin/python judge_filter.py                             # locally: cuda, else mps, else CPU

The judge is asked "Is the following statement true? Answer yes or no." and its answer is read from the logits of
"Yes"/"yes" vs "No"/"no" at the first reply position; nothing is generated, and the judge writes no text or label.
Writes data/judge.jsonl: {i, p_yes_true, p_yes_false} for each row i of data/truth_pool.jsonl.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from relations import RELATIONS

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ASK = "Is the following statement true? Answer yes or no.\n\n{s}"


def statement(r, t):
    return RELATIONS[r["pid"]][0].format(h=r["h"], t=t)


def main(data_dir=HERE / "data"):
    data_dir = Path(data_dir)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    yes = [tok.encode(w)[0] for w in ("Yes", "yes")]
    no = [tok.encode(w)[0] for w in ("No", "no")]
    pool = [json.loads(line) for line in open(data_dir / "truth_pool.jsonl")]
    texts = []
    for r in pool:
        for t in (r["t_true"], r["t_false"]):
            msgs = [{"role": "user", "content": ASK.format(s=statement(r, t))}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    p_yes = [0.0] * len(texts)
    bs = 128
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ids = order[b:b + bs]
            enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True).to(dev)
            logits = lm(**enc, logits_to_keep=1).logits[:, -1].float()
            lp = torch.log_softmax(logits, -1)
            ly = torch.logsumexp(lp[:, yes], -1)
            ln = torch.logsumexp(lp[:, no], -1)
            py = torch.sigmoid(ly - ln).cpu().tolist()
            for i, v in zip(ids, py):
                p_yes[i] = v
            if b % (bs * 40) == 0:
                print(f"{b}/{len(order)}", flush=True)
    with open(data_dir / "judge.jsonl", "w") as f:
        for i in range(len(pool)):
            f.write(json.dumps(dict(i=i, p_yes_true=round(p_yes[2 * i], 4), p_yes_false=round(p_yes[2 * i + 1], 4)))
                    + "\n")
    kept = sum(p_yes[2 * i] > 0.5 and p_yes[2 * i + 1] < 0.5 for i in range(len(pool)))
    print(f"kept {kept} of {len(pool)} facts ({kept / len(pool):.1%})")


if __name__ == "__main__":
    main()
