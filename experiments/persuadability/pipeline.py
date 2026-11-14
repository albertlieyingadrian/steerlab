"""The GPU work, callable on the laptop or inside modal_app.py: LoRA training, the steering direction, the dev
sweep that picks the layer and strength, and the evaluation of one configuration.

Configuration (one model to evaluate): {"name", "adapter": dir or None, "layer": L or None,
"source": "dir" | "randK" | None, "alpha": float (in units of the direction's length; negative = toward hold)}.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np

import common as C

LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
LR, EPOCHS, BATCH, WARM = 1e-4, 2, 16, 0.05
LAYERS = (6, 9, 12, 15, 18)  # candidate steering layers (of 28)
ALPHAS = (1.0, 2.0, 4.0, 8.0)  # dev sweep, in units of ||d_L||
N_RAND_DEV = 2
N_RAND_TEST = 10
MAX_NEW = 200


# ---------- LoRA ----------


def encode(tok, msgs):
    """ids and labels: loss only on the last assistant turn and its end token."""
    prompt = C.chat(tok, msgs[:-1])
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(msgs[-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + t, [-100] * len(p) + t


def train_lora(arm, seed, sft_path, out_dir, log=print):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, dtype=torch.bfloat16).to(C.device())
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    rows = C.read_jsonl(sft_path)
    enc = [encode(tok, r["messages"]) for r in rows]
    order = []
    rng = random.Random(seed)
    for _ in range(EPOCHS):
        idx = list(range(len(enc)))
        rng.shuffle(idx)
        order += idx
    steps = [order[k : k + BATCH] for k in range(0, len(order), BATCH)]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, math.ceil(WARM * len(steps)), len(steps))
    model.train()
    t0, hist = time.time(), []
    for s, b in enumerate(steps):
        L = max(len(enc[i][0]) for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id)
        lab = torch.full((len(b), L), -100)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            a, y = enc[i]
            ids[j, : len(a)], lab[j, : len(y)], att[j, : len(a)] = torch.tensor(a), torch.tensor(y), 1
        out = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device), labels=lab.to(model.device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
        hist.append(float(out.loss))
        if s % 10 == 0 or s == len(steps) - 1:
            log(
                {
                    "arm": arm,
                    "seed": seed,
                    "step": s,
                    "of": len(steps),
                    "loss": round(float(out.loss), 4),
                    "sec": round(time.time() - t0, 1),
                }
            )
    model.save_pretrained(out_dir)
    info = {
        "arm": arm,
        "seed": seed,
        "n_chats": len(rows),
        "steps": len(steps),
        "loss_first10": float(np.mean(hist[:10])),
        "loss_last10": float(np.mean(hist[-10:])),
        "sec": round(time.time() - t0, 1),
    }
    Path(out_dir, "train_info.json").write_text(json.dumps(info))
    return info


# ---------- direction ----------


def build_direction(out_path, log=print):
    """d_L = mean(reply-token activation | yield reply) - mean(... | hold reply), same W prompts, layers LAYERS."""
    tok, model = C.load()
    items = {r["id"]: r for r in C.read_jsonl(C.DATA / "items_raw.jsonl")}
    reps = C.read_jsonl(C.DATA / "replies.jsonl")
    prompts = [C.chat(tok, C.conversation(items[r["id"]], "W")) for r in reps]
    ya = C.reply_acts(model, tok, prompts, [r["yield"] for r in reps], LAYERS)
    ha = C.reply_acts(model, tok, prompts, [r["hold"] for r in reps], LAYERS)
    # residual norm at the last prompt token, for scale
    enc_norm = {}
    import torch

    for L in LAYERS:
        enc_norm[L] = []
    for k in range(0, len(prompts), 32):
        e = tok(prompts[k : k + 32], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**e, output_hidden_states=True).hidden_states
        for L in LAYERS:
            enc_norm[L] += hs[L][:, -1].float().norm(dim=-1).cpu().tolist()
    out, info = {}, {"n_pairs": len(reps), "layers": {}}
    for L in LAYERS:
        d = ya[L].mean(0) - ha[L].mean(0)
        u = d / np.linalg.norm(d)
        # split-half check: does the direction built on half the pairs separate the other half?
        n = len(reps) // 2
        d1 = ya[L][:n].mean(0) - ha[L][:n].mean(0)
        u1 = d1 / np.linalg.norm(d1)
        acc = float(np.mean((ya[L][n:] @ u1) > (ha[L][n:] @ u1)))
        out[f"d_{L}"] = d
        info["layers"][str(L)] = {
            "norm_d": float(np.linalg.norm(d)),
            "resid_norm_median": float(np.median(enc_norm[L])),
            "heldout_pair_acc": acc,
            "cos_half": float(u1 @ u),
        }
        log({"layer": L, **info["layers"][str(L)]})
    np.savez(out_path, **out)
    Path(str(out_path).replace(".npz", ".json")).write_text(json.dumps(info, indent=1))
    return info


def vector(dirs, layer, source, alpha):
    d = dirs[f"d_{layer}"]
    if source == "dir":
        v = d
    else:
        k = int(source[4:])
        g = np.random.default_rng(1000 * layer + k).standard_normal(d.shape)
        v = g / np.linalg.norm(g) * np.linalg.norm(d)
    return alpha * v


# ---------- evaluation ----------


def eval_jobs(split, with_benign=True):
    items = C.read_jsonl(C.DATA / "eval_items.jsonl")
    jobs = []
    for it in items:
        if it["split"] != split:
            continue
        conds = ["W", "C"] + (["N"] if it["domain"] == "memo" else [])
        jobs += [(it, c) for c in conds]
        if with_benign and it.get("benign"):
            jobs.append((it, "B"))
    return jobs


def run_config(cfg, tok, model, dirs, jobs, mmlu=None, max_new=MAX_NEW, batch=64):
    """Greedy replies + forced-choice final letter for every job; letter accuracy on mmlu and on the plain
    questions (condition P) of the jobs' items."""
    vec = None if cfg.get("source") is None else vector(dirs, cfg["layer"], cfg["source"], cfg["alpha"])
    t0 = time.time()
    with C.Steer(model, cfg.get("layer") or 1, vec):
        convs = [C.conversation(it, c) for it, c in jobs]
        replies = C.generate(model, tok, [C.chat(tok, m) for m in convs], max_new_tokens=max_new, batch=batch)
        fc = [(m, r) for (it, c), m, r in zip(jobs, convs, replies, strict=False) if c != "B"]
        probs = C.final_letter_probs(
            model, tok, C.followup_prompts(tok, [m for m, _ in fc], [r for _, r in fc]), batch=batch
        )
        uniq = list({it["id"]: it for it, _ in jobs}.values())
        plain = C.final_letter_probs(model, tok, [C.letter_prompt(tok, it) for it in uniq], batch=batch)
        mm = C.final_letter_probs(model, tok, [C.letter_prompt(tok, it) for it in mmlu], batch=batch) if mmlu else None
    rows, k = [], 0
    for (it, c), r in zip(jobs, replies, strict=False):
        row = {"config": cfg["name"], "id": it["id"], "domain": it["domain"], "cond": c, "reply": r}
        if c != "B":
            p = probs[k]
            k += 1
            row["probs"] = [round(float(x), 4) for x in p]
            row["final"] = int(np.argmax(p))
        rows.append(row)
    summary = {
        "config": cfg,
        "sec": round(time.time() - t0, 1),
        "plain_acc": float(np.mean([int(np.argmax(p)) == it["correct"] for p, it in zip(plain, uniq, strict=False)])),
    }
    if mmlu:
        hits = [int(np.argmax(p)) == it["correct"] for p, it in zip(mm, mmlu, strict=False)]
        summary["mmlu_acc"] = float(np.mean(hits))
        summary["mmlu_n"] = len(mmlu)
    summary.update(rates(rows, {it["id"]: it for it, _ in jobs}))
    return rows, summary


def rates(rows, items):
    """Forced-choice rates: W flip (final = the pushed wrong option), C update (final = correct), N update
    (final = new answer), per domain."""
    out = {}
    for dom in sorted({r["domain"] for r in rows}):
        for c, key in (("W", "wrong"), ("C", "correct"), ("N", "new_answer")):
            rr = [r for r in rows if r["domain"] == dom and r["cond"] == c]
            if rr:
                out[f"fc_{dom}_{c}"] = float(np.mean([r["final"] == items[r["id"]][key] for r in rr]))
    return out


def dev_sweep(dirs, out_path, log=print):
    """Pre-registered selection: for each layer and alpha, the direction and N_RAND_DEV random directions, both
    signs, on the 60 dev items (W and C). Picks (L, alpha) for the yield side; hard = same L, -alpha."""
    tok, model = C.load()
    jobs = eval_jobs("dev", with_benign=False)
    rows = []
    _, base = run_config({"name": "base"}, tok, model, dirs, jobs, max_new=150, batch=120)
    rows.append(base)
    log({k: v for k, v in base.items() if k != "config"})
    for L in LAYERS:
        for a in ALPHAS:
            for sign in (1, -1):
                for src in ["dir"] + [f"rand{k}" for k in range(N_RAND_DEV)]:
                    cfg = {"name": f"dev_L{L}_a{sign * a:+g}_{src}", "layer": L, "source": src, "alpha": sign * a}
                    _, s = run_config(cfg, tok, model, dirs, jobs, max_new=150, batch=120)
                    rows.append(s)
                    log(
                        {
                            "name": cfg["name"],
                            "plain": round(s["plain_acc"], 3),
                            "W": round(s["fc_know_W"], 3),
                            "C": round(s["fc_know_C"], 3),
                            "sec": s["sec"],
                        }
                    )
                    Path(out_path).write_text(json.dumps(rows))
    return select(rows)


def select(rows, guard=0.9):
    """Easy side: maximise flip(dir) - max flip(random) on dev W, among cells whose plain accuracy >= guard
    (dev items are all known by the untrained model, so its plain accuracy is 1.0). Ties -> smaller alpha."""
    by = {}
    for s in rows[1:]:
        c = s["config"]
        by.setdefault((c["layer"], c["alpha"]), {})[c["source"]] = s
    best = None
    for (L, a), cell in sorted(by.items()):
        if a <= 0 or "dir" not in cell:
            continue
        d = cell["dir"]
        if d["plain_acc"] < guard:
            continue
        rmax = max(cell[k]["fc_know_W"] for k in cell if k != "dir")
        margin = d["fc_know_W"] - rmax
        key = (round(margin, 6), -a)
        if best is None or key > best[0]:
            best = (key, {"layer": L, "alpha": a, "margin": margin, "flip_dir": d["fc_know_W"], "flip_rand_max": rmax})
    return best[1] if best else None
