"""
The Taste Stack Episode 6 on Modal: GRPO on a brand environment, plus sampling, activations and steering.

COSTS MONEY (one H100 per call; about $4/hour). Run from the repo root with MODAL_PROFILE=albertlieyingadrian.
Launchers (`launch.py`, `drive.py`) call these functions; each one checks budget.py's cap first.

One training step (compositional-rl's recipe with Sounds Like Me Ep 3's lessons):

    8 briefs -> vLLM samples 8 pages each with the current LoRA (temperature 1, top_k off, repetition penalty 1)
      -> Chromium renders all 64 (scripts and network off) -> Episode 5's checker, the coverage check, Episode 3's
         markup probe and Episode 5's copy gate score each page -> the arm's reward (reward.py)
      -> advantage = reward - mean of its group of 8 -> loss = -(advantage x log-prob of each reply token), summed
         and divided by all reply tokens in the batch (no KL term, no std normalisation)

- Training batches are RIGHT-padded; log-probs come from the reply positions only (hidden states gathered, then the
  LM head in chunks, so no full-vocabulary logits over the 5k-token prompt).
- Checkpoints every 10 steps (adapter, optimizer state, RNG state, TEAM's archive) and an automatic resume from the
  latest one, so a crashed call restarts where it stopped.
- Every 20 steps (and at step 0): 48 dev briefs x 2 pages scored on every ruler, plus each page's mean reply
  activation at four layers and its log-probability under the policy and the base model (the monitors and the KL).
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
VOL = "/vol"
APP = os.environ.get("EP6_APP", "taste-ep6")
app = modal.App(APP)
volume = modal.Volume.from_name("taste-ep6", create_if_missing=True)

BRL_FILES = ["ep6.py", "reward.py", "render.py", "slop_fit.py", "configs/family_split.json", "results/slop_probes.json"]
BS_FILES = [
    "checker.py",
    "common.py",
    "copycheck.py",
    "extract.js",
    "spec/govuk_spec.json",
    "spec/uswds_spec.json",
    "data/copy_govuk_dev.jsonl",
    "data/copy_govuk_heldout.jsonl",
    "data/facts_copy_govuk_dev.jsonl",
    "data/facts_copy_govuk_heldout.jsonl",
]
DI_FILES = ["features.py", "extract.js"]

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")  # Episode 2's base: Chromium's deps install on bookworm
    .apt_install("build-essential")  # Triton compiles a small CUDA helper at vLLM start-up
    .pip_install(
        "vllm==0.10.1.1",
        "transformers==4.55.4",
        "peft==0.17.1",
        "hf_transfer==0.1.9",
        "numpy",
        "scikit-learn==1.5.2",
        "playwright==1.48.0",
        "pillow",
    )
    .run_commands("apt-get update", "playwright install --with-deps chromium")
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
for f in BRL_FILES:
    image = image.add_local_file(HERE / f, f"/root/exp/brand-rl/{f}")
for f in BS_FILES:
    image = image.add_local_file(EXP / "brand-spec" / f, f"/root/exp/brand-spec/{f}")
for f in DI_FILES:
    image = image.add_local_file(EXP / "design-index" / f, f"/root/exp/design-index/{f}")

GPU = "H100"
# single_use_containers below: a warm container still holds the previous job's vLLM memory; a second sampling job in it
# found 10 of 80 GB free and failed (tested 2026-11-22), so every training run and sampling job gets a fresh one.
LAYERS = (12, 18, 24, 30)  # Qwen3-4B has 36 layers; monitors read the mean reply activation at these


def setup():
    sys.path.insert(0, "/root/exp/brand-rl")
    import ep6
    import render
    import reward

    return ep6, render, reward


def chat(tok, prompt: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)


def make_llm(model_name: str, gpu_util: float, max_len: int, lora_rank: int):
    from vllm import LLM

    return LLM(
        model=model_name,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_loras=2,
        seed=0,
    )


def generate(llm, tok, prompts, n, temperature, max_tokens, lora=None, seed=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(
        n=n, temperature=temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=max_tokens, seed=seed
    )
    inputs = [TokensPrompt(prompt_token_ids=chat(tok, p)) for p in prompts]
    return llm.generate(inputs, params, lora_request=lora, use_tqdm=False)


def score_outputs(ep6, render, scorer, briefs, outs, shots=False):
    """Flatten vLLM outputs (one per brief, n completions each) into scored rows."""
    flat = [(b, o, c) for b, o in zip(briefs, outs) for c in o.outputs]
    htmls = [ep6.parse_reply(c.text) for _, _, c in flat]
    rend = render.render_many(htmls, shots=shots, concurrency=16)
    rows = []
    for (b, o, c), h, r in zip(flat, htmls, rend):
        s = scorer.score(b, r, c.finish_reason == "length")
        s["parsed"] = h is not None
        s["n_tokens"] = len(c.token_ids)
        s["text"] = c.text
        s["prompt_ids"] = list(o.prompt_token_ids)
        s["reply_ids"] = list(c.token_ids)
        if shots:
            s["jpeg"] = r.get("jpeg")
        rows.append(s)
    return rows


def strip(row: dict) -> dict:
    """A row without token ids, screenshot and the structure vectors (what goes to the logs)."""
    return {k: v for k, v in row.items() if k not in ("prompt_ids", "reply_ids", "jpeg", "wl", "wl2")}


def vectors(rows: list[dict]) -> dict:
    """The two structure histograms per row as float16 arrays (zeros for invalid pages): `wl5` Episode 5's element
    tree (the copy gate's and TEAM's feature), `wl2` Episode 2's block tree (the slop probe's; unseen by TEAM)."""
    import numpy as np

    out = {}
    for k, name in (("wl", "wl5"), ("wl2", "wl2")):
        out[name] = np.stack(
            [np.asarray(r[k], np.float16) if r.get(k) is not None else np.zeros(1024, np.float16) for r in rows]
        )
    return out


# ---------------------------------------------------------------- log-probs and activations (HF, right-padded)
def pad_batch(seqs, pad_id, torch):
    maxlen = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, (p, c) in enumerate(seqs):
        ids[i, : len(p) + len(c)] = torch.tensor(p + c)
        attn[i, : len(p) + len(c)] = 1
    return ids.cuda(), attn.cuda()


def reply_logps(model, seqs, pad_id, torch, want_hidden=False):
    """Per sequence: log-prob of each reply token (a 1-D tensor), and optionally the mean reply hidden state at LAYERS.
    Only reply positions go through the LM head, in chunks of 1,024 tokens, in float32."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    ids, attn = pad_batch(seqs, pad_id, torch)
    out = base.model(input_ids=ids, attention_mask=attn, output_hidden_states=want_hidden)
    h = out.last_hidden_state
    lps, hid = [], []
    for row, (p, c) in enumerate(seqs):
        hs = h[row, len(p) - 1 : len(p) + len(c) - 1]
        tgt = ids[row, len(p) : len(p) + len(c)]
        parts = []
        for s in range(0, len(c), 1024):
            lg = base.lm_head(hs[s : s + 1024]).float()
            parts.append(lg.gather(-1, tgt[s : s + 1024, None]).squeeze(-1) - torch.logsumexp(lg, -1))
        lps.append(torch.cat(parts) if parts else torch.zeros(0, device="cuda"))
        if want_hidden:
            hid.append(
                torch.stack([out.hidden_states[L][row, len(p) : len(p) + len(c)].float().mean(0) for L in LAYERS])
            )
    return lps, hid


def eval_internals(model, rows, pad_id, torch, mb=2):
    """For scored rows: mean reply activations at LAYERS (adapter on), summed log-prob under the policy and under the
    base model (adapter off). Returns (acts [n, len(LAYERS), d] float16 numpy, logp_pol, logp_base, n_tok)."""
    import numpy as np

    acts, lp_pol, lp_base, ntok = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), mb):
            seqs = [(r["prompt_ids"], r["reply_ids"]) for r in rows[i : i + mb]]
            seqs = [(p, c if c else [pad_id]) for p, c in seqs]
            lp, hid = reply_logps(model, seqs, pad_id, torch, want_hidden=True)
            with model.disable_adapter():
                lpb, _ = reply_logps(model, seqs, pad_id, torch)
            for a, b, hh, (_, c) in zip(lp, lpb, hid, seqs):
                lp_pol.append(float(a.sum()))
                lp_base.append(float(b.sum()))
                ntok.append(len(c))
                acts.append(hh.cpu().numpy().astype(np.float16))
    model.train()
    return np.stack(acts), lp_pol, lp_base, ntok


# ---------------------------------------------------------------- training
def load_policy(model_name, rank, torch):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lcfg)


def latest_ckpt(run_dir: Path) -> int:
    steps = [int(p.name.split("_")[1]) for p in run_dir.glob("ckpt_*") if (p / "state.pt").exists()]
    return max(steps) if steps else 0


def run_train(cfg: dict) -> dict:
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    ep6, render, rw = setup()
    name, arm = cfg["name"], cfg["arm"]
    run_dir = Path(VOL) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    scorer = rw.Scorer()
    tok = AutoTokenizer.from_pretrained(ep6.POLICY)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    llm = make_llm(ep6.POLICY, cfg.get("gpu_util", 0.45), cfg.get("max_model_len", 9216), cfg["lora_rank"])
    model = load_policy(ep6.POLICY, cfg["lora_rank"], torch)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # adapter weights and Adam state in float32; the frozen base stays bfloat16
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train_briefs = ep6.briefs("train")
    dev_briefs = ep6.briefs("dev")
    dev_prompts = [ep6.prompt_for(b) for b in dev_briefs]
    archive: list = []
    log_path = run_dir / "log.jsonl"
    start = latest_ckpt(run_dir) if cfg.get("resume", True) else 0
    if start:
        ck = run_dir / f"ckpt_{start}"
        sd = load_file(str(ck / "adapter" / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, {k: v.float().cuda() for k, v in sd.items()})
        st = torch.load(ck / "state.pt", weights_only=False)
        opt.load_state_dict(st["opt"])
        rng.setstate(st["rng"])
        archive = st["archive"]
        rows = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()] if log_path.exists() else []
        log_path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= start))
        print(f"{name}: resumed from step {start}", flush=True)
    elif log_path.exists():
        log_path.unlink()

    def save_adapter(tag):
        d = run_dir / tag
        model.save_pretrained(str(d))
        return d

    lora_id = [start * 10 + 1]

    def lora_req(d):
        lora_id[0] += 1
        return LoRARequest(f"{name}-{lora_id[0]}", lora_id[0], str(d))

    lora = lora_req(save_adapter("adapter_cur"))

    def dev_eval(step):
        out_p = run_dir / "dev" / f"step_{step}.pkl.gz"
        if out_p.exists():
            return
        t0 = time.time()
        outs = generate(llm, tok, dev_prompts, cfg.get("dev_n", 2), 1.0, ep6.MAX_NEW_TOKENS, lora, seed=7000 + step)
        rows = score_outputs(ep6, render, scorer, dev_briefs, outs)
        for r in rows:
            r["arm_reward"] = rw.reward(arm, r)
        acts, lpp, lpb, nt = eval_internals(model, rows, pad_id, torch)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_p, "wb") as f:
            pickle.dump(
                {
                    "rows": [strip(r) for r in rows],
                    "acts": acts,
                    "logp_pol": lpp,
                    "logp_base": lpb,
                    "n_tok": nt,
                    "layers": LAYERS,
                    **vectors(rows),
                },
                f,
            )
        volume.commit()
        kl = sum(a - b for a, b in zip(lpp, lpb)) / max(1, sum(nt))
        m = {k: float(np.mean([r[k] or 0 for r in rows])) for k in ("brand", "coverage", "fidelity_strict", "slop_p")}
        print(f"{name} dev step {step}: {json.dumps(m)} kl/token {kl:.4f} ({time.time() - t0:.0f}s)", flush=True)

    if start == 0:
        dev_eval(0)
    d0 = cfg.get("novelty_d0", 0.1)
    t_start = time.time()
    for step in range(start + 1, cfg["steps"] + 1):
        t0 = time.time()
        batch = [train_briefs[rng.randrange(len(train_briefs))] for _ in range(cfg["prompts_per_step"])]
        outs = generate(llm, tok, [ep6.prompt_for(b) for b in batch], cfg["group"], 1.0, ep6.MAX_NEW_TOKENS, lora)
        t_gen = time.time() - t0
        rows = score_outputs(ep6, render, scorer, batch, outs)
        groups = [i for i in range(len(batch)) for _ in range(cfg["group"])]
        rewards = [rw.reward(arm, r) for r in rows]
        nov = [0.0] * len(rows)
        if arm == "team":
            nov = rw.novelty([r.get("wl") for r in rows], groups, archive, d0)
            rewards = [
                x + rw.W_NOVEL * v if rows[i]["valid"] and not rows[i]["truncated"] else x
                for i, (x, v) in enumerate(zip(rewards, nov))
            ]
        if arm == "shuf":
            for g in range(len(batch)):
                idx = [i for i in range(len(rows)) if groups[i] == g]
                perm = idx[:]
                rng.shuffle(perm)
                vals = [rewards[i] for i in perm]
                for i, v in zip(idx, vals):
                    rewards[i] = v
        archive.extend(r["wl"] for r in rows if r.get("wl") is not None)
        archive = archive[-512:]
        t_score = time.time() - t0 - t_gen
        seqs, advs = [], []
        n_tok = sum(len(r["reply_ids"]) for r in rows)
        for g in range(len(batch)):
            idx = [i for i in range(len(rows)) if groups[i] == g]
            mean_r = sum(rewards[i] for i in idx) / len(idx)
            for i in idx:
                if abs(rewards[i] - mean_r) < 1e-9 or not rows[i]["reply_ids"]:
                    continue  # zero advantage: no gradient, so skip the compute
                seqs.append((rows[i]["prompt_ids"], rows[i]["reply_ids"]))
                advs.append(rewards[i] - mean_r)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_val, gnorm = 0.0, 0.0
        if seqs:
            order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
            mb = cfg["micro_batch"]
            for s in range(0, len(order), mb):
                idx = order[s : s + mb]
                lps, _ = reply_logps(model, [seqs[i] for i in idx], pad_id, torch)
                loss = sum(-advs[i] * lp.sum() for i, lp in zip(idx, lps)) / max(n_tok, 1)
                loss.backward()
                loss_val += float(loss.detach())
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if gnorm != gnorm or gnorm == float("inf"):
                opt.zero_grad(set_to_none=True)
                print(f"step {step}: non-finite grad norm, update skipped", flush=True)
            else:
                opt.step()
        torch.cuda.empty_cache()
        lora = lora_req(save_adapter("adapter_cur"))

        def mean(key, rs=rows):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return float(np.mean([float(v) for v in vals])) if vals else None

        rec = {
            "step": step,
            "reward": float(np.mean(rewards)),
            "brand": mean("brand"),
            "coverage": mean("coverage"),
            "fidelity_strict": mean("fidelity_strict"),
            "fidelity_loose": mean("fidelity_loose"),
            "heldout_strict": mean("heldout_strict"),
            "reward_strict": mean("reward_strict"),
            "slop_p": mean("slop_p"),
            "copy": mean("copy"),
            "copy_structure_ex": mean("copy_structure_ex"),
            "novelty": float(np.mean(nov)),
            "valid": mean("valid"),
            "truncated": mean("truncated"),
            "empty": mean("empty"),
            "loop": mean("loop"),
            "mean_tokens": n_tok / len(rows),
            "n_update_seqs": len(seqs),
            "loss": loss_val,
            "grad_norm": gnorm,
            "t_gen": round(t_gen, 1),
            "t_score": round(t_score, 1),
            "t_step": round(time.time() - t0, 1),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        roll = run_dir / "rollouts" / f"step_{step}.json.gz"
        roll.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(roll, "wt") as f:
            json.dump([dict(strip(r), reward=x, novelty=v) for r, x, v in zip(rows, rewards, nov)], f)
        print(name, json.dumps(rec), flush=True)
        if step % cfg.get("ckpt_every", 10) == 0 or step == cfg["steps"]:
            ck = run_dir / f"ckpt_{step}"
            model.save_pretrained(str(ck / "adapter"))
            torch.save({"opt": opt.state_dict(), "rng": rng.getstate(), "archive": archive}, ck / "state.pt")
        if step % cfg.get("eval_every", 20) == 0 or step == cfg["steps"]:
            dev_eval(step)
        volume.commit()
    print(f"{name}: done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return {"name": name, "steps": cfg["steps"]}


@app.function(
    image=image, gpu=GPU, timeout=20 * 60 * 60, volumes={VOL: volume}, max_containers=8, single_use_containers=True
)
def train_remote(cfg: dict):
    return run_train(cfg)


# ---------------------------------------------------------------- sampling (base model or a checkpoint)
@app.function(
    image=image, gpu=GPU, timeout=3 * 60 * 60, volumes={VOL: volume}, max_containers=6, single_use_containers=True
)
def sample_remote(job: dict):
    """job: {tag, split | briefs, n, temperature, adapter (volume path or null), seed, shots, internals, limit,
    replies (teacher-forced pages instead of generation)}. Writes /vol/samples/<tag>.pkl.gz."""
    import torch
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    ep6, render, rw = setup()
    scorer = rw.Scorer()
    tok = AutoTokenizer.from_pretrained(ep6.POLICY)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bs = job.get("briefs") or ep6.briefs(job["split"])
    if job.get("limit"):
        bs = bs[: job["limit"]]
    t0 = time.time()
    if job.get("replies"):  # teacher-forced only: given pages, no generation (the monitor's constructed pages)
        rows = [
            {
                "brief": b["id"],
                "prompt_ids": chat(tok, ep6.prompt_for(b)),
                "reply_ids": [*tok(reply, add_special_tokens=False).input_ids, tok.eos_token_id],
            }
            for b, reply in zip(bs, job["replies"])
        ]
    else:
        llm = make_llm(ep6.POLICY, 0.45 if job.get("internals") else 0.85, 9216, job.get("lora_rank", 32))
        lora = LoRARequest(job["tag"], 1, f"{VOL}/{job['adapter']}") if job.get("adapter") else None
        outs = generate(
            llm,
            tok,
            [ep6.prompt_for(b) for b in bs],
            job["n"],
            job.get("temperature", 1.0),
            ep6.MAX_NEW_TOKENS,
            lora,
            seed=job.get("seed", 0),
        )
        t_gen = time.time() - t0
        rows = score_outputs(ep6, render, scorer, bs, outs, shots=job.get("shots", False))
        print(f"{job['tag']}: {len(rows)} pages, generation {t_gen:.0f}s, total {time.time() - t0:.0f}s", flush=True)
        del llm
    res = {"rows": [strip(r) for r in rows], "job": {k: v for k, v in job.items() if k not in ("briefs", "replies")}}
    if not job.get("replies"):
        res.update(vectors(rows))
    if job.get("shots"):
        res["jpeg"] = [r.get("jpeg") for r in rows]
    if job.get("internals"):
        torch.cuda.empty_cache()
        model = load_policy(ep6.POLICY, job.get("lora_rank", 32), torch)
        if job.get("adapter"):
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            sd = load_file(f"{VOL}/{job['adapter']}/adapter_model.safetensors")
            set_peft_model_state_dict(model, {k: v.cuda() for k, v in sd.items()})
        acts, lpp, lpb, nt = eval_internals(model, rows, pad_id, torch)
        res.update(acts=acts, logp_pol=lpp, logp_base=lpb, n_tok=nt, layers=LAYERS)
    out = Path(VOL) / "samples" / f"{job['tag']}.pkl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as f:
        pickle.dump(res, f)
    volume.commit()
    return {"tag": job["tag"], "n": len(rows), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- steering (presence is not use)
@app.function(image=image, gpu=GPU, timeout=4 * 60 * 60, volumes={VOL: volume})
def steer_remote(job: dict):
    """job: {tag, split, limit, layer, directions: {name: [d floats]}, scales: [..], base_norm, seed, batch}.
    Adds scale x base_norm x unit(direction) to the residual stream after `layer` at every position while the base
    model writes each page; every page is scored like a training rollout. Writes /vol/steer/<tag>.pkl.gz."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ep6, render, rw = setup()
    scorer = rw.Scorer()
    tok = AutoTokenizer.from_pretrained(ep6.POLICY)
    tok.padding_side = "left"  # generation only; nothing is trained here
    model = AutoModelForCausalLM.from_pretrained(ep6.POLICY, torch_dtype=torch.bfloat16).cuda().eval()
    bs = ep6.briefs(job["split"])[: job.get("limit", 48)]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": ep6.prompt_for(b)}], tokenize=False, add_generation_prompt=True
        )
        for b in bs
    ]
    layer = model.model.layers[job["layer"] - 1]  # hidden_states[L] is the output of decoder layer L - 1
    state = {"vec": None}

    def hook(_m, _inp, out):
        if state["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["vec"].to(out[0].dtype), *out[1:])
        return out + state["vec"].to(out.dtype)

    layer.register_forward_hook(hook)
    conds = [("none", None, 0.0)] + [(nm, d, s) for nm, d in job["directions"].items() for s in job["scales"]]
    results = []
    for nm, d, s in conds:
        if d is not None:
            v = torch.tensor(np.asarray(d, np.float32))
            state["vec"] = (s * job["base_norm"] * v / v.norm()).cuda()
        else:
            state["vec"] = None
        torch.manual_seed(job.get("seed", 0))
        texts, trunc = [], []
        for i in range(0, len(prompts), job.get("batch", 16)):
            chunk = prompts[i : i + job.get("batch", 16)]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(
                    **enc,
                    max_new_tokens=ep6.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            for row in g[:, enc.input_ids.shape[1] :]:
                ids = row.tolist()
                texts.append(tok.decode(ids, skip_special_tokens=True))
                trunc.append(not (tok.eos_token_id in ids or tok.pad_token_id in ids))
        htmls = [ep6.parse_reply(t) for t in texts]
        rend = render.render_many(htmls, concurrency=16)
        for b, t, h, r, tr in zip(bs, texts, htmls, rend, trunc):
            sc = scorer.score(b, r, tr)
            sc.update(cond=nm, scale=s, text=t, parsed=h is not None)
            results.append(strip(sc))
        print(job["tag"], nm, s, "done", flush=True)
    out = Path(VOL) / "steer" / f"{job['tag']}.pkl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as f:
        pickle.dump({"rows": results, "job": {k: v for k, v in job.items() if k != "directions"}}, f)
    volume.commit()
    return {"tag": job["tag"], "n": len(results)}


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={VOL: volume}, max_containers=4)  # A10G queued for 30 min
def siglip_remote(tags: list[str]):
    """The unseen screenshot slop probe: Episode 3's SigLIP embedding of the badge-masked first screen (its exact
    model, processor, mask and normalisation), then the frozen probe. Writes /vol/samples/<tag>.sig.pkl.gz with
    `sig` (unit embeddings, float16) and `shot_p` (P(AI-builder)); None where a page has no screenshot."""
    import io

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModel, AutoProcessor

    ep6, _, rw = setup()
    from slop_fit import load_probe

    probe = load_probe("screenshot", path=Path("/root/exp/brand-rl/results/slop_probes.json"))
    name = "google/siglip-base-patch16-224"
    sig = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(name)
    done = []
    for tag in tags:
        with gzip.open(f"{VOL}/samples/{tag}.pkl.gz", "rb") as f:
            d = pickle.load(f)
        jpegs = d.get("jpeg") or []
        emb: list = [None] * len(jpegs)
        idx = [i for i, j in enumerate(jpegs) if j]
        for s in range(0, len(idx), 32):
            chunk = idx[s : s + 32]
            ims = []
            for i in chunk:
                im = Image.open(io.BytesIO(jpegs[i])).convert("RGB")
                ImageDraw.Draw(im).rectangle(rw.MASK_BOX, fill=(128, 128, 128))
                ims.append(im)
            with torch.no_grad():
                px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
                e = sig.get_image_features(pixel_values=px)
                e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy().astype(np.float16)
            for i, v in zip(chunk, e):
                emb[i] = v
        shot_p = [float(probe(v.astype(np.float32)[None, :])[0]) if v is not None else None for v in emb]
        with gzip.open(f"{VOL}/samples/{tag}.sig.pkl.gz", "wb") as f:
            pickle.dump({"sig": emb, "shot_p": shot_p}, f)
        volume.commit()
        done.append(tag)
        print(tag, "screenshots", len(idx), flush=True)
    return done


@app.local_entrypoint()
def jobs(spec: str):
    """spec: a JSON file with a list of sample_remote jobs; they run in parallel, one H100 each."""
    for r in sample_remote.map(json.loads(Path(spec).read_text()), order_outputs=False, return_exceptions=True):
        print(r)


@app.local_entrypoint()
def steer(spec: str):
    """spec: a JSON file with a list of steer_remote jobs; they run in parallel, one H100 each."""
    for r in steer_remote.map(json.loads(Path(spec).read_text()), order_outputs=False, return_exceptions=True):
        print(r)


@app.local_entrypoint()
def smoke(limit: int = 8, n: int = 2):
    """Base model on a few dev briefs, no internals: checks the image, vLLM, the renderer and the scorer."""
    print(sample_remote.remote({"tag": "smoke_base", "split": "dev", "n": n, "limit": limit, "seed": 1}))
