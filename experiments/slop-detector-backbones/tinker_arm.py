"""Round 3: a slop detector trained on Tinker as a label-token classifier (Inkling-2-Mini, Inkling-2-Small, and
Qwen3.5-9B on the same recipe as the control that separates method from model).

    python tinker_arm.py smoke --base thinkingmachines/Inkling-2-Mini      # 20 steps, 8 rows, 32 scored texts
    python tinker_arm.py train --arm inkling-2-mini                         # 1 epoch, then scores every split

Recipe. Each text (fenced code emptied, its first 512 tokens) becomes the prompt "Text:\\n<text>\\nLabel:", and the
model is trained (LoRA rank 16, cross-entropy on the one label token only, each row weighted by its band weight as in
04_train.py) to emit the digit of its bucket, "0" (human) to "3" (AI-generated). One epoch over data v2's 79,120 rows,
64 rows per step, Adam, the learning rate warmed up over 3% of steps and then cosine to zero (04_train.py's schedule).
A text's score is the expected bucket under the four label tokens' probabilities, renormalised and divided by 3, the
same scale as the bucket head, so the analysis and cut-offs are unchanged.

Cost is estimated from token counts at Tinker's list prices and checked against a hard cap ($130, BB_TINKER_CAP) before
every step and every scoring batch; a payment refusal stops the run. The key is read from the lab's .env with shlex
(the line carries text after the quoted value) and never printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import time
from pathlib import Path

import numpy as np

from bb import DATA, RESULTS, RUNS, SPLITS, sd

LEDGER = RESULTS / "tinker_spend.jsonl"
CAP = float(os.environ.get("BB_TINKER_CAP", "130"))
# USD per million tokens, Tinker models.json (fetched 2026-11-22)
PRICE = {
    "thinkingmachines/Inkling-2-Mini": {"train": 0.75, "prefill": 0.15, "sample": 0.50},
    "thinkingmachines/Inkling-2-Small": {"train": 1.50, "prefill": 0.30, "sample": 1.00},
    "Qwen/Qwen3.5-9B": {"train": 1.463, "prefill": 0.30, "sample": 0.807},
}
ARMS3 = {
    "inkling-2-mini": "thinkingmachines/Inkling-2-Mini",
    "inkling-2-small": "thinkingmachines/Inkling-2-Small",
    "qwen3.5-9b-tinker": "Qwen/Qwen3.5-9B",
}
RANK, BATCH, EPOCHS, LR, MAX_TEXT = 16, 64, 1.0, 2e-4, 512
LABELS = ("0", "1", "2", "3")
TOPK = 20


def load_key() -> None:
    """TINKER_API_KEY from the lab's .env, parsed as the shell would (never printed)."""
    if os.environ.get("TINKER_API_KEY"):
        return
    env = Path(os.environ.get("LAB_ENV", Path(__file__).resolve().parents[5] / ".env"))
    for line in env.read_text().splitlines():
        if line.strip().startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = shlex.split(line, comments=True)[0].split("=", 1)[1]
            return
    raise SystemExit("TINKER_API_KEY not found in the lab .env")


def spent() -> float:
    return sum(r.get("usd", 0.0) for r in sd.read_jsonl(LEDGER))


def charge(what: str, base: str, kind: str, tokens: int) -> None:
    usd = tokens * PRICE[base][kind] / 1e6
    if spent() + usd > CAP:
        raise SystemExit(f"Tinker cap ${CAP:.0f} reached before {what} (spent ${spent():.2f})")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "what": what, "base": base, "kind": kind, "tokens": tokens}
        f.write(json.dumps({**rec, "usd": round(usd, 4)}) + "\n")


class Prompter:
    """Text -> prompt tokens, with the label digits as single tokens (checked)."""

    def __init__(self, tok):
        self.tok = tok
        self.head = tok.encode("Text:\n", add_special_tokens=False)
        self.tail = tok.encode("\nLabel:", add_special_tokens=False)
        self.label_ids = [tok.encode(d, add_special_tokens=False) for d in LABELS]
        assert all(len(x) == 1 for x in self.label_ids), self.label_ids
        self.label_ids = [x[0] for x in self.label_ids]

    def prompt(self, text: str) -> list[int]:
        body = self.tok.encode(sd.empty_fences(text), add_special_tokens=False)[:MAX_TEXT]
        return self.head + body + self.tail


def datum(prompt: list[int], label_id: int, weight: float):
    from tinker import types

    tokens = [*prompt, label_id]
    w = np.zeros(len(tokens) - 1, dtype=np.float32)
    w[-1] = weight
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_numpy(np.array(tokens[1:], dtype=np.int64)),
            "weights": types.TensorData.from_numpy(w),
        },
    )


def lr_at(step: int, total: int) -> float:
    warm = max(1, int(0.03 * total))
    return LR * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))


def score_texts(sampler, pr: Prompter, base: str, texts: list[str], what: str, inflight: int = 128) -> dict:
    """Expected bucket / 3 from the four label tokens' top-k log-probabilities; also the mass they cover."""
    from tinker import types

    params = types.SamplingParams(max_tokens=1, temperature=1.0)
    scores, mass = [0.0] * len(texts), [0.0] * len(texts)
    t0 = time.time()
    for start in range(0, len(texts), inflight):
        chunk = list(range(start, min(len(texts), start + inflight)))
        prompts = [pr.prompt(texts[i]) for i in chunk]
        charge(f"{what} score {start}", base, "prefill", sum(len(p) for p in prompts))
        futs = [sampler.sample(types.ModelInput.from_ints(p), 1, params, topk_sample_logprobs=TOPK) for p in prompts]
        for i, f in zip(chunk, futs, strict=True):
            top = dict(f.result().sequences[0].topk_logprobs[0])
            p = np.array([math.exp(top[t]) if t in top else 0.0 for t in pr.label_ids])
            mass[i] = float(p.sum())
            scores[i] = float((p / p.sum()) @ np.arange(4) / 3) if p.sum() > 0 else 0.5
    return {
        "scores": [round(s, 6) for s in scores],
        "label_mass": [round(m, 4) for m in mass],
        "seconds": round(time.time() - t0, 1),
    }


def train(base: str, name: str, rows: list[dict], steps_cap: int | None, val_texts: list[str] | None) -> dict:
    import tinker
    from tinker import types

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=base, rank=RANK, seed=0)
    from transformers import AutoTokenizer

    # Tinker's get_tokenizer() needs tml_tokenizers for Inkling; the Hugging Face tokenizer of the same repo is used
    # for every arm, and the smoke run checks the label tokens carry the probability mass.
    pr = Prompter(AutoTokenizer.from_pretrained(base))
    rng = random.Random(0)
    order = list(range(len(rows)))
    rng.shuffle(order)
    batches = [order[i : i + BATCH] for i in range(0, int(len(order) * EPOCHS), BATCH)]
    if steps_cap:
        batches = batches[:steps_cap]
    total = len(batches)
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    t0, pending = time.time(), []
    for step, idx in enumerate(batches):
        data, ntok = [], 0
        for i in idx:
            p = pr.prompt(rows[i]["text"])
            data.append(datum(p, pr.label_ids[rows[i]["bucket"]], rows[i]["weight"] / len(idx)))
            ntok += len(p) + 1
        charge(f"{name} step {step}", base, "train", ntok)
        fb = tc.forward_backward(data, "cross_entropy")
        opt = tc.optim_step(types.AdamParams(learning_rate=lr_at(step, total)))
        pending.append((step, fb, opt))
        if len(pending) > 2:  # keep two steps in flight
            s, f, o = pending.pop(0)
            loss = float(f.result().metrics.get("loss:sum", float("nan")))
            o.result()
            if s % 10 == 0:
                log.write(
                    json.dumps(
                        {
                            "step": s,
                            "of": total,
                            "loss": round(loss, 4),
                            "lr": lr_at(s, total),
                            "elapsed": round(time.time() - t0),
                        }
                    )
                    + "\n"
                )
                log.flush()
    for _s, f, o in pending:
        f.result()
        o.result()
    sampler = tc.save_weights_and_get_sampling_client(name=f"{name}-final")
    cfg = {
        "base": base,
        "rank": RANK,
        "batch": BATCH,
        "epochs": EPOCHS,
        "lr": LR,
        "steps": total,
        "seconds": round(time.time() - t0),
        "label_ids": pr.label_ids,
        "max_text_tokens": MAX_TEXT,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return {"sampler": sampler, "prompter": pr, "cfg": cfg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["smoke", "train"])
    ap.add_argument("--base", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--arm", choices=list(ARMS3))
    a = ap.parse_args()
    load_key()
    train_rows = sd.read_jsonl(DATA / "train.jsonl")
    val_rows = sd.read_jsonl(DATA / "val.jsonl")
    if a.step == "smoke":
        rows = train_rows[:2000]
        global BATCH
        BATCH = 8
        r = train(a.base, f"smoke-tinker-{a.base.split('/')[-1]}", rows, 20, None)
        texts = [x["text"] for x in val_rows[:32]]
        s = score_texts(r["sampler"], r["prompter"], a.base, texts, "smoke")
        y = [x["bucket"] for x in val_rows[:32]]
        print(
            json.dumps(
                {
                    "label_mass_min": min(s["label_mass"]),
                    "label_mass_mean": float(np.mean(s["label_mass"])),
                    "score_vs_bucket_corr": float(np.corrcoef(s["scores"], y)[0, 1]),
                    "seconds_32": s["seconds"],
                    "spent_usd": round(spent(), 4),
                }
            )
        )
        return
    base = ARMS3[a.arm]
    name = f"bb-{a.arm}"
    r = train(base, name, train_rows, None, None)
    out = RUNS / name
    val = score_texts(r["sampler"], r["prompter"], base, [x["text"] for x in val_rows], f"{name} our_val")
    (out / "val_scores.json").write_text(json.dumps(val["scores"]) + "\n")
    (out / "scores").mkdir(exist_ok=True)
    timing = {"our_val": val["seconds"]}
    ev = __import__("bb").eval_texts()
    mass = {"our_val": float(np.mean(val["label_mass"]))}
    for split in SPLITS:
        s = score_texts(r["sampler"], r["prompter"], base, ev[split], f"{name} {split}")
        (out / "scores" / f"{split}.json").write_text(json.dumps(s["scores"]) + "\n")
        timing[split] = s["seconds"]
        mass[split] = float(np.mean(s["label_mass"]))
    (out / "timing.json").write_text(json.dumps({"seconds": timing, "label_mass_mean": mass}, indent=1) + "\n")
    print(name, "done", json.dumps(mass))


if __name__ == "__main__":
    main()
