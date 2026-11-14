"""The Glimmer recipe on Qwen3-8B, with activations saved, for the confidence-probe twist.

COSTS MONEY (Modal H100, roughly $4/hour; about 1.5 hours for all three phases). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/glimmer-kalshi/modal_qwen.py --phase smoke
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/glimmer-kalshi/modal_qwen.py --phase all

Same data, prompt layout and loss as Fireworks' decision recipe: the cookbook's system prompt and
"State: ... Question: ... A) ... Reply with only the letter" text, options in sorted-key order, yes/no as
A) yes B) no, K-way cross-entropy over the option-letter logits at the answer position, LoRA rank 64 /
alpha 32, Adam lr 1e-4, batch 64, one epoch, seed 0. Qwen3's thinking is off (empty think block), the
analogue of Glimmer's "reasoning low" prefill.

Phases (each writes to the Modal Volume `glimmer-kalshi-qwen` under qwen3-8b/):
  readout base  -> option logits + residual activations (last position, several layers) for dev and test
  train         -> LoRA adapter
  readout tuned -> same as base, with the adapter
Probes are fitted next to the activations on a Modal CPU container: --phase probe (probe.py).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-8B"
VOL = "/vol"
OUT = f"{VOL}/qwen3-8b"
LAYERS = (
    9,
    18,
    27,
    36,
)  # hidden_states index: 0 = embeddings, 36 = final layer output (before norm)
SYSTEM_PROMPT = "You are a decision engine. You answer questions about a state by replying with a single letter."
CODES = [chr(65 + i) for i in range(26)]

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.20.2",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "numpy",
    )
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(str(HERE / "data"), "/root/data")
)
probe_image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install("numpy", "scikit-learn==1.7.2")
    .env({"QWEN_DIR": f"{VOL}/qwen3-8b", "RESULTS_DIR": f"{VOL}/qwen3-8b/results"})
    .add_local_file(str(HERE / "evaluate.py"), "/root/evaluate.py")
    .add_local_file(str(HERE / "probe.py"), "/root/probe.py")
)
app = modal.App("glimmer-kalshi-qwen", image=image)
volume = modal.Volume.from_name("glimmer-kalshi-qwen", create_if_missing=True)


def render_text(v, fallback=""):
    if v is None:
        return fallback
    return (
        v
        if isinstance(v, str)
        else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    )


def questions(split: str, max_rows: int | None = None):
    """(row index, qid, type, prompt user text, K, gold) per labeled question, cookbook layout."""
    out = []
    with open(f"/root/data/{split}.jsonl") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            row = json.loads(line)
            state = "State:\n" + render_text(row["state"]) + "\n\nQuestion:"
            for qid in sorted(row["questions"]):
                q = row["questions"][qid]
                if q.get("label") is None:
                    continue
                if q["type"] == "noul":
                    keys, labels = ["true", "false"], ["yes", "no"]
                    gold = 0 if q["label"] in (True, "true") else 1
                else:
                    keys = sorted(q["criteria"])
                    labels = [render_text(q["criteria"][k], k) for k in keys]
                    gold = keys.index(str(q["label"]))
                lines = "".join(f"{CODES[j]}) {lab}\n" for j, lab in enumerate(labels))
                branch = f" {render_text(q.get('instructions'))}\n{lines}\nReply with only the letter of the correct option."
                out.append(
                    {
                        "row": i,
                        "qid": qid,
                        "type": q["type"],
                        "text": state + branch,
                        "k": len(keys),
                        "gold": gold,
                    }
                )
    return out


def encode(tok, items):
    code_ids = [tok.encode(c, add_special_tokens=False)[0] for c in CODES]
    for it in items:
        s = tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": it["text"]},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        it["ids"] = tok.encode(s, add_special_tokens=False)
    return code_ids


def batches(items, max_tokens):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    cur, longest = [], 0
    for i in order:
        n = len(items[i]["ids"])
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            yield cur
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        yield cur


def pad_left(tok, seqs, device):
    import torch

    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, n - len(s) :] = torch.tensor(s)
        mask[j, n - len(s) :] = 1
    return ids.to(device), mask.to(device)


def load_model(adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def do_readout(tag: str, max_rows: int | None):
    import numpy as np
    import torch

    adapter = f"{OUT}/adapter" if tag == "tuned" else None
    tok, model = load_model(adapter)
    for split in ("dev", "test"):
        items = questions(split, max_rows)
        code_ids = encode(tok, items)
        logits_out = [None] * len(items)
        acts = {
            L: np.zeros((len(items), model.config.hidden_size), dtype=np.float16)
            for L in LAYERS
        }
        t0 = time.time()
        with torch.no_grad():
            for b in batches(items, 24000):
                ids, mask = pad_left(tok, [items[i]["ids"] for i in b], "cuda")
                out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                last = out.logits[:, -1, :].float()
                for j, i in enumerate(b):
                    k = items[i]["k"]
                    logits_out[i] = last[j, code_ids[:k]].tolist()
                for L in LAYERS:
                    h = (
                        out.hidden_states[L][:, -1, :]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                    acts[L][b] = h
        print(
            f"{tag}/{split}: {len(items)} questions in {time.time() - t0:.0f}s",
            flush=True,
        )
        Path(OUT).mkdir(parents=True, exist_ok=True)
        with open(f"{OUT}/{tag}_{split}.jsonl", "w") as fh:
            fh.writelines(
                json.dumps(
                    {
                        "row": it["row"],
                        "qid": it["qid"],
                        "type": it["type"],
                        "gold": it["gold"],
                        "n_tokens": len(it["ids"]),
                        "logprobs": lg,
                    }
                )
                + "\n"
                for it, lg in zip(items, logits_out)
            )
        for L in LAYERS:
            np.save(f"{OUT}/{tag}_{split}_L{L}.npy", acts[L])
        volume.commit()


def do_train(
    max_rows: int | None,
    epochs: int = 1,
    batch: int = 64,
    lr: float = 1e-4,
    seed: int = 0,
):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    random.seed(seed)
    torch.manual_seed(seed)
    tok, model = load_model(None)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=64,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.train()
    items = questions("train", max_rows)
    code_ids = encode(tok, items)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    # The recipe batches rows (events); a row contributes all its questions. Batch 64 events, as it does.
    by_row: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        by_row.setdefault(it["row"], []).append(i)
    rows = list(by_row)
    log = []
    step = 0
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(rows)
        total_steps = math.ceil(len(rows) / batch) * epochs
        for s in range(0, len(rows), batch):
            idx = [i for r in rows[s : s + batch] for i in by_row[r]]
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for mb in batches([items[i] for i in idx], 12000):
                sub = [idx[j] for j in mb]
                ids, mask = pad_left(tok, [items[i]["ids"] for i in sub], "cuda")
                out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
                last = out.logits[:, -1, :].float()
                loss = 0.0
                for j, i in enumerate(sub):
                    k = items[i]["k"]
                    lg = last[j, code_ids[:k]]
                    loss = loss + F.cross_entropy(
                        lg[None], torch.tensor([items[i]["gold"]], device="cuda")
                    )
                (loss / len(idx)).backward()
                loss_sum += float(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            log.append({"step": step, "loss": loss_sum / len(idx), "n": len(idx)})
            if step % 10 == 0 or step == 1:
                print(
                    f"step {step}/{total_steps} loss {loss_sum / len(idx):.4f}",
                    flush=True,
                )
    model.save_pretrained(f"{OUT}/adapter")
    Path(f"{OUT}/train_log.json").write_text(json.dumps(log))
    volume.commit()


@app.function(gpu="H100", volumes={VOL: volume}, timeout=4 * 3600)
def run(phase: str, max_rows: int | None = None, sub: str = "qwen3-8b"):
    global OUT
    OUT = f"{VOL}/{sub}"
    if phase in ("readout_base", "all"):
        do_readout("base", max_rows)
    if phase in ("train", "all"):
        do_train(max_rows)
    if phase in ("readout_tuned", "all"):
        do_readout("tuned", max_rows)


@app.function(
    image=probe_image, cpu=16, memory=65536, volumes={VOL: volume}, timeout=3 * 3600
)
def probe():
    import sys

    sys.path.insert(0, "/root")
    import probe as p

    p.main()
    volume.commit()


@app.local_entrypoint()
def main(phase: str = "smoke"):
    if phase == "smoke":
        run.remote("all", max_rows=40, sub="smoke")
    elif phase == "probe":
        probe.remote()
    else:
        run.remote(phase)
