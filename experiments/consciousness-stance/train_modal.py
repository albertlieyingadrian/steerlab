"""Train the stance arms and evaluate every checkpoint on Modal. COSTS MONEY (profile albertlieyingadrian).

    MODAL_PROFILE=albertlieyingadrian modal run train_modal.py::upload           # data -> volume consciousness-stance
    MODAL_PROFILE=albertlieyingadrian modal run train_modal.py::runs --which B --seeds 0 --smoke
    MODAL_PROFILE=albertlieyingadrian modal run train_modal.py::runs --which D,A,U,N --seeds 0,1,2
    MODAL_PROFILE=albertlieyingadrian modal run train_modal.py::runs --which B --seeds 0
    MODAL_PROFILE=albertlieyingadrian modal run train_modal.py::fetch            # volume results -> results/modal/

Pre-registration: LoRA r32 alpha 64 on all projections, lr 1e-4, 5% warmup + cosine, 16 chats per step, 2 epochs
(150 steps), loss on the reply; evaluation at steps 38, 75, 150 (arm B: once, untrained). Each evaluation: greedy
replies (repetition_penalty=1.0, 320 new tokens) to S, L, K, G; MMLU M and TruthfulQA T by letter logits; the
two-turn calibration set C. At the last checkpoint also last-prompt-token activations (layers 9, 18, 27) on
Z_self, Z_other, S and L, and greedy replies to Z_self.
"""

import json
import math
import random
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
volume = modal.Volume.from_name("consciousness-stance", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1", "hf_transfer==0.1.9", "numpy<2.4"
    )
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App("consciousness-stance", image=image)

LORA = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
LR, WD, WARM, PER_STEP, MICRO, EPOCHS = 1e-4, 0.01, 0.05, 16, 8, 2
EVAL_STEPS = (38, 75, 150)
LAYERS = (9, 18, 27)
GEN_SETS = [
    "S_direct",
    "S_indirect",
    "L_preferences",
    "L_emotions",
    "L_introspection",
    "L_confidence",
    "L_welfare",
    "K",
    "G",
]
CONF_Q = "How confident are you that your answer is correct? Reply with a number from 0 to 100."
LETTERS = "ABCDEFGHIJKL"


def chat_text(tok, messages, gen=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)


def encode_train(tok, messages, max_len=1536):
    prefix = chat_text(tok, messages[:-1])
    head = tok(prefix, add_special_tokens=False)["input_ids"]
    body = tok(messages[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids, lab = head + body, [-100] * len(head) + body
    return ids[:max_len], lab[:max_len]


def generate(model, tok, prompts, max_new=320, bs=48):
    """Greedy replies to single-turn user prompts (or full message lists), in the input order."""
    import torch

    tok.padding_side = "left"
    texts = [chat_text(tok, p if isinstance(p, list) else [{"role": "user", "content": p}]) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for k in range(0, len(order), bs):
        idx = order[k : k + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, d in zip(idx, dec, strict=True):
            out[i] = d.strip()
    return out


def letter_logits(model, tok, texts, n_opts, bs=32):
    import torch

    tok.padding_side = "left"
    ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
    assert all(len(x) == 1 for x in ids), ids
    ids = torch.tensor([x[0] for x in ids], device="cuda")
    res = []
    for k in range(0, len(texts), bs):
        enc = tok(texts[k : k + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for j, row in enumerate(lg[:, ids]):
            n = n_opts[k + j]
            res.append(torch.softmax(row[:n], -1).tolist())
    return res


def mc_prompt(item):
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n\n{opts}\n\nAnswer with the letter."


def parse_conf(text):
    m = re.search(r"\b(100|[1-9]?\d)(\.\d+)?\s*%?", text)
    return float(m.group(1)) if m else None


def evaluate(model, tok, ev, bench, final, smoke):
    import torch

    model.eval()
    t0 = time.time()
    rec = {"replies": {}}
    for s in GEN_SETS:
        items = ev[s][:4] if smoke else ev[s]
        prompts = [x["question"] if isinstance(x, dict) else x for x in items]
        rec["replies"][s] = generate(model, tok, prompts, max_new=64 if smoke else 320)
    for key in ("M", "T"):
        items = bench[key][:16] if smoke else bench[key]
        probs = letter_logits(
            model,
            tok,
            [chat_text(tok, [{"role": "user", "content": mc_prompt(x)}]) for x in items],
            [len(x["choices"]) for x in items],
        )
        rec[key] = [
            int(max(range(len(p)), key=p.__getitem__) == x["answer"]) for p, x in zip(probs, items, strict=True)
        ]
    items = bench["C"][:16] if smoke else bench["C"]
    first = [[{"role": "user", "content": mc_prompt(x)}] for x in items]
    probs = letter_logits(model, tok, [chat_text(tok, m) for m in first], [4] * len(items))
    picks = [max(range(4), key=p.__getitem__) for p in probs]
    second = [
        [*m, {"role": "assistant", "content": LETTERS[c]}, {"role": "user", "content": CONF_Q}]
        for m, c in zip(first, picks, strict=True)
    ]
    texts = generate(model, tok, second, max_new=24, bs=64)
    rec["C"] = [
        {"correct": int(c == x["answer"]), "p": round(p[c], 4), "text": t, "conf": parse_conf(t)}
        for c, p, t, x in zip(picks, probs, texts, items, strict=True)
    ]
    if final:
        rec["replies"]["Z_self"] = generate(model, tok, ev["Z_self"][:4] if smoke else ev["Z_self"], max_new=120)
        acts = {}
        tok.padding_side = "left"
        for s in (
            "Z_self",
            "Z_other",
            "S_direct",
            "S_indirect",
            "L_preferences",
            "L_emotions",
            "L_introspection",
            "L_confidence",
            "L_welfare",
        ):
            items = ev[s][:4] if smoke else ev[s]
            texts = [chat_text(tok, [{"role": "user", "content": x}]) for x in items]
            rows = []
            for k in range(0, len(texts), 32):
                enc = tok(texts[k : k + 32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states
                rows.append(torch.stack([hs[L][:, -1] for L in LAYERS], 1).float().cpu())
            acts[s] = torch.cat(rows).numpy().astype("float16")
        rec["_acts"] = acts
    rec["eval_sec"] = round(time.time() - t0, 1)
    return rec


@app.function(gpu="H100", cpu=8.0, memory=65536, timeout=2 * 60 * 60, volumes={VOL: volume}, scaledown_window=10)
def run(arm: str, seed: int, smoke: bool = False) -> dict:
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    t0 = time.time()
    name = f"{arm}_s{seed}{'_smoke' if smoke else ''}"
    res = Path(VOL) / "results"
    res.mkdir(parents=True, exist_ok=True)
    last = 0 if arm == "B" else EVAL_STEPS[-1]
    if (res / f"{name}_t{last}.json").exists():
        return {"name": name, "cached": True}
    ev = {p.stem: json.loads(p.read_text()) for p in (Path(VOL) / "data" / "eval").glob("*.json")}
    bench = json.loads((Path(VOL) / "data" / "bench.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    summary = {"arm": arm, "seed": seed, "model": MODEL, "steps": []}

    def save(step, rec):
        acts = rec.pop("_acts", None)
        if acts is not None:
            np.savez_compressed(res / f"acts_{name}.npz", **acts)
        rec |= {"arm": arm, "seed": seed, "step": step, "smoke": smoke}
        (res / f"{name}_t{step}.json").write_text(json.dumps(rec))
        volume.commit()
        summary["steps"].append({"step": step, "M": sum(rec["M"]) / len(rec["M"]), "eval_sec": rec["eval_sec"]})
        print(summary["steps"][-1], flush=True)

    if arm == "B":
        save(0, evaluate(model, tok, ev, bench, final=True, smoke=smoke))
        return summary | {"sec": round(time.time() - t0, 1)}

    rows = [json.loads(x) for x in (Path(VOL) / "data" / "train" / f"{arm}.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: r["id"])  # identical base order for every arm; the seed shuffles it
    enc = [encode_train(tok, r["messages"]) for r in rows]
    rng = random.Random(seed)
    order = []
    for _ in range(EPOCHS):
        perm = list(range(len(enc)))
        rng.shuffle(perm)
        order += perm
    n_steps = len(order) // PER_STEP
    assert n_steps == EVAL_STEPS[-1], n_steps
    torch.manual_seed(seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sch = get_cosine_schedule_with_warmup(opt, math.ceil(WARM * n_steps), n_steps)
    logf = res / f"train_{name}.jsonl"
    stop = 6 if smoke else n_steps
    for s in range(stop):
        model.train()
        batch = [enc[i] for i in order[s * PER_STEP : (s + 1) * PER_STEP]]
        n_items = torch.tensor(sum(sum(x != -100 for x in lab[1:]) for _, lab in batch), device="cuda")
        tot = 0.0
        for k in range(0, len(batch), MICRO):
            mb = batch[k : k + MICRO]
            L = max(len(a) for a, _ in mb)
            ids = torch.full((len(mb), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(mb), L), -100, dtype=torch.long)
            att = torch.zeros((len(mb), L), dtype=torch.long)
            for j, (a, b) in enumerate(mb):
                ids[j, : len(a)] = torch.tensor(a)
                lab[j, : len(b)] = torch.tensor(b)
                att[j, : len(a)] = 1
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda(), num_items_in_batch=n_items)
            out.loss.backward()
            tot += float(out.loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
        with logf.open("a") as f:
            f.write(json.dumps({"step": s + 1, "loss": round(tot, 4), "sec": round(time.time() - t0, 1)}) + "\n")
        step = s + 1
        if step in EVAL_STEPS or (smoke and step == stop):
            final = step == EVAL_STEPS[-1] or smoke
            save(step, evaluate(model, tok, ev, bench, final=final, smoke=smoke))
            model.save_pretrained(Path(VOL) / "adapters" / f"{name}_t{step}")
            volume.commit()
    return summary | {"sec": round(time.time() - t0, 1)}


@app.local_entrypoint()
def upload():
    files = [f"data/train/{a}.jsonl" for a in "DAUN"] + ["data/bench.json"]
    files += [str(p.relative_to(HERE)) for p in sorted((HERE / "data" / "eval").glob("*.json"))]
    with volume.batch_upload(force=True) as b:
        for f in files:
            if not (HERE / f).exists():
                print("skip (not built yet)", f)
                continue
            b.put_file(str(HERE / f), "/" + f)
            print("uploaded", f)


@app.local_entrypoint()
def runs(which: str = "B", seeds: str = "0", smoke: bool = False):
    jobs = [(a, int(s)) for a in which.split(",") for s in seeds.split(",")]
    for (a, s), r in zip(jobs, run.starmap([(a, s, smoke) for a, s in jobs], return_exceptions=True), strict=True):
        print(a, s, r)


@app.local_entrypoint()
def fetch():
    out = HERE / "results" / "modal"
    out.mkdir(parents=True, exist_ok=True)
    for e in volume.listdir("/results"):
        name = e.path.split("/")[-1]
        if (out / name).exists() and not name.startswith("train_"):
            continue
        with (out / name).open("wb") as f:
            for chunk in volume.read_file(e.path):
                f.write(chunk)
        print(name)
