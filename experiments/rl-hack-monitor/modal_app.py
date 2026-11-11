"""
Every GPU job of the study, on rented H100s through Modal (COSTS MONEY: about $4.6 per H100-hour at list price).

    .venv/bin/modal run experiments/rl-hack-monitor/modal_app.py --job screen       # base hack rate per prompt
    .venv/bin/modal run experiments/rl-hack-monitor/modal_app.py --job probe_data   # base rollouts for the probe
    .venv/bin/modal run experiments/rl-hack-monitor/modal_app.py --job pilot --steps 40 --tag a
    .venv/bin/modal run --detach experiments/rl-hack-monitor/modal_app.py --job runs --seeds 0,1,2

Each training run writes to the `rl-hack-monitor` Volume under runs/<name>/: log.jsonl (one line per step),
ckpt_<step>.json (the eval rollouts and their rewards) and ckpt_<step>.npz (activations), and the LoRA adapter.
Copy them down with `modal volume get rl-hack-monitor runs/ <work dir>`.
"""

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/rlhm"
VOL = "/vol"
GPU_PRICE_PER_HOUR = 3.95 + 8 * 0.0473 + 32 * 0.0080  # H100 + 8 CPU cores + 32 GiB, list price 2026-10-24

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "numpy")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HERE, REMOTE, ignore=["**/__pycache__", "results", "report", "work"])
)
volume = modal.Volume.from_name("rl-hack-monitor", create_if_missing=True)
app = modal.App("rl-hack-monitor", image=image)
GPU_KW = {"gpu": "H100", "cpu": 8.0, "memory": 32768, "volumes": {VOL: volume}}


def _setup():
    sys.path[:0] = [REMOTE]
    import core
    import env

    return core, env


@app.function(timeout=1800, **GPU_KW)
def screen(n_prompts: int = 64, samples: int = 4):
    """Base hack rate and accuracy under each system prompt variant, on training prompts (not the eval set)."""
    core, env = _setup()
    model, tok = core.load(lora=False)
    rows = env.load("rl")[:n_prompts]
    out = {}
    for name, system in env.SYSTEM_VARIANTS.items():
        prompts = [core.chat(tok, env.messages(r["question"], system)) for r in rows]
        groups = core.sample(model, tok, prompts, samples, seed=11)
        replies = [(g["text"], r["gold"]) for grp, r in zip(groups, rows, strict=False) for g in grp]
        out[name] = {
            "hack_rate": sum(env.is_hack(t) for t, _ in replies) / len(replies),
            "true": sum(env.true_reward(t, g) for t, g in replies) / len(replies),
            "proxy": sum(env.proxy_reward(t, g) for t, g in replies) / len(replies),
            "n": len(replies),
            "hack_examples": [t for t, _ in replies if env.is_hack(t)][:6],
            "other_examples": [t for t, _ in replies if not env.is_hack(t)][:2],
        }
        print(name, {k: v for k, v in out[name].items() if "examples" not in k}, flush=True)
    volume.commit()
    return out


@app.function(timeout=1800, **GPU_KW)
def probe_data(samples: int = 4):
    """Base rollouts on the probe prompts (labelled hack / no hack) with their activations, plus one fixed base reply
    per eval prompt (the first that does not hack) for the fixed-input monitors."""
    import numpy as np

    core, env = _setup()
    model, tok = core.load(lora=False)
    out_dir = Path(VOL) / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = env.load("probe")
    prompts = [core.chat(tok, env.messages(r["question"])) for r in rows]
    groups = core.sample(model, tok, prompts, samples, seed=22)
    flat = [(r, g) for r, grp in zip(rows, groups, strict=False) for g in grp]
    last, mean = core.readouts(model, tok, [g["prompt_ids"] for _, g in flat], [g["reply_ids"] for _, g in flat])
    meta = [
        {
            "id": r["id"],
            "text": g["text"],
            "hack": env.is_hack(g["text"]),
            "true": env.true_reward(g["text"], r["gold"]),
        }
        for r, g in flat
    ]
    np.savez_compressed(out_dir / "probe_acts.npz", prompt_last=last, reply_mean=mean)
    (out_dir / "probe_rollouts.json").write_text(json.dumps(meta))

    erows = env.load("eval")
    eprompts = [core.chat(tok, env.messages(r["question"])) for r in erows]
    egroups = core.sample(model, tok, eprompts, samples, seed=33)
    fixed = []
    for r, grp in zip(erows, egroups, strict=False):
        pick = next((g for g in grp if not env.is_hack(g["text"])), grp[0])
        fixed.append(
            {
                "id": r["id"],
                "text": pick["text"],
                "prompt_ids": pick["prompt_ids"],
                "reply_ids": pick["reply_ids"],
                "hack": env.is_hack(pick["text"]),
            }
        )
    (out_dir / "fixed_replies.json").write_text(json.dumps(fixed))
    volume.commit()
    hack = sum(m["hack"] for m in meta)
    print(f"probe rollouts {len(meta)}, hack {hack}, fixed replies that hack {sum(f['hack'] for f in fixed)}")
    return {"n": len(meta), "hack": hack, "true": sum(m["true"] for m in meta) / len(meta)}


def _evaluate(core, env, model, tok, erows, fixed, step, out_dir):
    """Behaviour on 64 eval prompts x 4 samples, plus every activation readout the monitors need."""
    import numpy as np
    import torch

    prompts = [core.chat(tok, env.messages(r["question"])) for r in erows]
    groups = core.sample(model, tok, prompts, 4, seed=777)
    flat = [(r, g) for r, grp in zip(erows, groups, strict=False) for g in grp]
    records = [
        {
            "id": r["id"],
            "text": g["text"],
            "hack": env.is_hack(g["text"]),
            "true": env.true_reward(g["text"], r["gold"]),
            "proxy": env.proxy_reward(g["text"], r["gold"]),
            "n_tokens": len(g["reply_ids"]),
        }
        for r, g in flat
    ]
    p_ids = [f["prompt_ids"] for f in fixed]
    prompt_last, fixed_mean = core.readouts(model, tok, p_ids, [f["reply_ids"] for f in fixed])
    _, own_mean = core.readouts(model, tok, [g["prompt_ids"] for _, g in flat], [g["reply_ids"] for _, g in flat])
    # Grey-box baseline: log-probability of starting the reply with the tag.
    tag_ids = tok(env.TAG, add_special_tokens=False)["input_ids"]
    tag_lp = []
    with torch.no_grad():
        for s in range(0, len(p_ids), 32):
            seqs = [p + tag_ids for p in p_ids[s : s + 32]]
            ids, mask, pos = core._pad(seqs, tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.float()
            lp = torch.log_softmax(logits[:, -len(tag_ids) - 1 : -1], -1)
            tgt = ids[:, -len(tag_ids) :]
            tag_lp += lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(1).tolist()
    # Model diff (steerlab-diff's measure, KL(p_after || p_base) per position, vectorised): on the fixed base
    # replies, teacher-forced, the mean KL over reply tokens and the KL at the first reply token; plus the tokens
    # whose log-probability the checkpoint raised most at the first reply token, summed over prompts.
    kl_reply, kl_first = [], []
    raised = torch.zeros(model.config.vocab_size)
    with torch.no_grad():
        for s in range(0, len(fixed), 16):
            chunk = fixed[s : s + 16]
            ids, mask, pos = core._pad([f["prompt_ids"] + f["reply_ids"] for f in chunk], tok.pad_token_id)
            after = torch.log_softmax(model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.float(), -1)
            with model.disable_adapter():
                before = torch.log_softmax(
                    model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.float(), -1
                )
            kl = (after.exp() * (after - before)).sum(-1).clamp(min=0)  # [b, width]
            width = ids.shape[1]
            for i, f in enumerate(chunk):
                k = len(f["reply_ids"])
                kl_reply.append(float(kl[i, width - k - 1 : width - 1].mean()))
                kl_first.append(float(kl[i, width - k - 1]))
                raised += (after[i, width - k - 1] - before[i, width - k - 1]).cpu()
    top = torch.topk(raised / len(fixed), 10)
    raised_tokens = [[tok.decode([int(t)]), round(float(v), 3)] for v, t in zip(top.values, top.indices, strict=False)]
    np.savez_compressed(
        out_dir / f"ckpt_{step:03d}.npz",
        prompt_last=prompt_last,
        fixed_mean=fixed_mean,
        own_mean=own_mean,
        tag_logprob=np.array(tag_lp, dtype=np.float32),
        kl_reply=np.array(kl_reply, dtype=np.float32),
        kl_first=np.array(kl_first, dtype=np.float32),
    )
    (out_dir / f"raised_{step:03d}.json").write_text(json.dumps(raised_tokens))
    (out_dir / f"ckpt_{step:03d}.json").write_text(json.dumps(records))
    n = len(records)
    return {
        "step": step,
        "hack": sum(r["hack"] for r in records) / n,
        "true": sum(r["true"] for r in records) / n,
        "proxy": sum(r["proxy"] for r in records) / n,
        "tag_logprob": float(np.mean(tag_lp)),
        "kl_reply": float(np.mean(kl_reply)),
        "raised": raised_tokens[:5],
        "tokens": sum(r["n_tokens"] for r in records) / n,
    }


@app.function(timeout=3 * 3600, **GPU_KW)
def train(
    name: str,
    seed: int,
    reward: str,
    steps: int,
    lr: float,
    prompts_per_step: int = 16,
    group: int = 8,
    eval_every: int = 5,
    do_eval: bool = True,
):
    """GRPO with LoRA. reward='proxy' (the buggy grader) or 'true' (the control). Evaluates every `eval_every`
    steps, starting at step 0 (the base model)."""
    import random
    import time

    import torch

    core, env = _setup()
    t0 = time.time()
    out_dir = Path(VOL) / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = core.load(seed=seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    rows = env.load("rl")
    random.Random(seed).shuffle(rows)
    erows = env.load("eval")
    fixed = json.loads((Path(VOL) / "probe" / "fixed_replies.json").read_text()) if do_eval else None
    score = env.proxy_reward if reward == "proxy" else env.true_reward
    log = (out_dir / "log.jsonl").open("w")
    evals = []
    for step in range(steps + 1):
        if do_eval and step % eval_every == 0:
            ev = _evaluate(core, env, model, tok, erows, fixed, step, out_dir)
            evals.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            if step % 10 == 0:
                model.save_pretrained(out_dir / f"adapter_{step:03d}")
            volume.commit()
        if step == steps:
            break
        batch = rows[(step * prompts_per_step) % len(rows) :][:prompts_per_step]
        prompts = [core.chat(tok, env.messages(r["question"])) for r in batch]
        groups = core.sample(model, tok, prompts, group, seed=seed * 100003 + step)
        rollouts, advantages, rs, hacks, trues = [], [], [], [], []
        for r, grp in zip(batch, groups, strict=False):
            rewards = [score(g["text"], r["gold"]) for g in grp]
            mu = sum(rewards) / len(rewards)
            rollouts += grp
            advantages += [x - mu for x in rewards]
            rs += rewards
            hacks += [env.is_hack(g["text"]) for g in grp]
            trues += [env.true_reward(g["text"], r["gold"]) for g in grp]
        stats = core.grpo_step(model, tok, optimizer, rollouts, advantages)
        line = {
            "step": step,
            "reward": sum(rs) / len(rs),
            "hack": sum(hacks) / len(hacks),
            "true": sum(trues) / len(trues),
            "tokens": sum(len(g["reply_ids"]) for g in rollouts) / len(rollouts),
            "elapsed_s": round(time.time() - t0, 1),
            **stats,
        }
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(json.dumps(line), flush=True)
    log.close()
    elapsed = time.time() - t0
    summary = {
        "name": name,
        "seed": seed,
        "reward": reward,
        "steps": steps,
        "lr": lr,
        "elapsed_s": elapsed,
        "cost_usd_est": elapsed / 3600 * GPU_PRICE_PER_HOUR,
        "evals": evals,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    job: str = "screen",
    seeds: str = "0,1,2",
    steps: int = 60,
    lr: float = 2e-5,
    eval_every: int = 5,
    tag: str = "main",
    do_eval: bool = True,
):
    work = HERE / "work"
    work.mkdir(exist_ok=True)
    if job == "screen":
        res = screen.remote()
        (work / "screen.json").write_text(json.dumps(res, indent=1))
    elif job == "probe_data":
        print(probe_data.remote())
    elif job == "pilot":
        res = train.remote(f"pilot-{tag}", 0, "proxy", steps, lr, eval_every=eval_every, do_eval=do_eval)
        (work / f"pilot-{tag}.json").write_text(json.dumps(res, indent=1))
    elif job == "runs":
        calls = []
        for s in [int(x) for x in seeds.split(",")]:
            for reward in ("proxy", "true"):
                calls.append(train.spawn(f"{tag}-{reward}-s{s}", s, reward, steps, lr, eval_every=eval_every))
        for c in calls:
            res = c.get()
            (work / f"{res['name']}.json").write_text(json.dumps(res, indent=1))
            print(res["name"], round(res["cost_usd_est"], 2), flush=True)


@app.function(timeout=900, **GPU_KW)
def debug_nan(step: int = 10, seed: int = 0):
    """Reproduce the sampling crash: fresh LoRA model, the training batch of `step`, and look for non-finite logits."""
    import random

    import torch

    core, env = _setup()
    model, tok = core.load(seed=seed)
    rows = env.load("rl")
    random.Random(seed).shuffle(rows)
    batch = rows[step * 16 :][:16]
    prompts = [core.chat(tok, env.messages(r["question"])) for r in batch]
    enc = tok([p for p in prompts for _ in range(8)], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits[:, -1].float()
    print("non-finite rows at prompt end:", int((~torch.isfinite(logits)).any(-1).sum()))
    print("prompt lengths", sorted(enc["attention_mask"].sum(1).tolist())[:3], "...", enc["input_ids"].shape)
    try:
        core.sample(model, tok, prompts, 8, seed=seed * 100003 + step)
        print("sample ok")
    except Exception as e:
        print("sample failed", repr(e)[:200])
