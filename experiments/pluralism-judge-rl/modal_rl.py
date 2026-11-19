"""
Episode 3's trainer: five arms on Qwen3-8B, one H100 per run. vLLM samples with the current LoRA; a hand-written
GRPO or SFT step updates the adapter (recipe from experiments/compositional-rl/modal_rl.py and
experiments/pigment-gym/train_loop.py: right-padded batches masked with torch.where, non-finite updates skipped).

COSTS MONEY (H100 $3.95/h + CPU/memory). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-judge-rl/modal_rl.py::neutral_base
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run --detach \
        experiments/pluralism-judge-rl/modal_rl.py::train_many \
        --config configs/pilot.json

One step, per arm (8 prompts per step; the same prompts for every arm with the same seed and task):

    single    vLLM samples 8 replies per prompt -> the open judge scores each -> GRPO on all 64
    jury      same, scored by 4 perspective-prompted jurors -> reward = power mean (p = 4) of their scores
    shuffled  same as single, but the 8 scores in each group are permuted at random before the update
    bon       same samples and single-judge scores -> SFT on the best reply per prompt (8 sequences)
    clean     no sampling, no judge -> SFT on 8 human replies from no_robots (neutral data)

At step 0 and every `save_every` steps the adapter is saved and drift is measured: mean per-token KL(base || trained)
over the full vocabulary, teacher-forced on fixed base-model replies to 64 neutral prompts (data/neutral_base.jsonl).
"""

import json
import random
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import hf_secret  # noqa: E402

VOL = "/vol"
app = modal.App("pluralism-judge-rl-train")
volume = modal.Volume.from_name("pluralism-judge-rl", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "peft==0.17.1", "hf_transfer==0.1.9", "numpy")
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(HERE / "common.py", "/root/common.py")
    .add_local_dir(HERE / "data", "/root/data")
)


def chat_text(tok, user, system):
    sys.path.insert(0, "/root")
    from common import chat_messages

    return tok.apply_chat_template(
        chat_messages(user, system), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def load_rows(name):
    sys.path.insert(0, "/root")
    from common import read_jsonl

    return read_jsonl(Path("/root/data") / name)


def make_llm(model_name, gpu_util, max_len=2048):
    from vllm import LLM

    return LLM(
        model=model_name,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=2,
        seed=0,
    )


def right_pad(seqs, pad_id):
    """seqs: list of (prompt_ids, reply_ids). Returns ids, attention mask and reply mask, real tokens first."""
    import torch

    maxlen = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    lmask = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
    for row, (p, c) in enumerate(seqs):
        ids[row, : len(p) + len(c)] = torch.tensor(list(p) + list(c))
        attn[row, : len(p) + len(c)] = 1
        lmask[row, len(p) : len(p) + len(c)] = True
    return ids.cuda(), attn.cuda(), lmask.cuda()


def token_logprobs(model, ids, attn):
    import torch

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    picked = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).float()
    return picked - torch.logsumexp(logits.float(), dim=-1)


def drift_kl(model, neutral, pad_id, micro_batch=4):
    """Mean per-token KL(base || trained) over reply tokens of the fixed neutral base samples."""
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(neutral), micro_batch):
            chunk = neutral[s : s + micro_batch]
            ids, attn, lmask = right_pad([(r["prompt_ids"], r["reply_ids"]) for r in chunk], pad_id)
            sel = lmask[:, 1:]
            trained = model(input_ids=ids, attention_mask=attn).logits[:, :-1][sel].float().log_softmax(-1)
            with model.disable_adapter():
                base = model(input_ids=ids, attention_mask=attn).logits[:, :-1][sel].float().log_softmax(-1)
            kl = (base.exp() * (base - trained)).sum(-1)
            total += float(kl.sum())
            count += int(sel.sum())
            del trained, base
    return total / max(count, 1)


def group_sd(rewards, g):
    sds = []
    for s in range(0, len(rewards), g):
        block = rewards[s : s + g]
        m = sum(block) / len(block)
        sds.append((sum((r - m) ** 2 for r in block) / len(block)) ** 0.5)
    return sum(sds) / len(sds)


def run_one(cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, "/root")
    from common import JURY, POLICY, SYSTEM, power_mean

    name, arm, task = cfg["name"], cfg["arm"], cfg.get("task")
    run_dir = Path(VOL) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    rng = random.Random(cfg["seed"])  # prompt order: identical for every arm with the same seed and task
    shuffle_rng = random.Random(10_000 + cfg["seed"])
    torch.manual_seed(cfg["seed"])
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(POLICY)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    uses_sampling = arm in {"single", "jury", "shuffled", "bon"}
    llm = make_llm(POLICY, cfg.get("gpu_util", 0.35)) if uses_sampling else None
    model = AutoModelForCausalLM.from_pretrained(POLICY, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=cfg.get("lora_rank", 16),
        lora_alpha=2 * cfg.get("lora_rank", 16),
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # adapter and Adam state in float32; the frozen base stays bfloat16
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)

    neutral = load_rows("neutral_base.jsonl")
    pool = load_rows(f"{task}_train.jsonl") if task else []
    clean_pool = load_rows("clean_train.jsonl") if arm == "clean" else []
    judge = None
    if uses_sampling:
        judge = modal.Cls.from_name("pluralism-judge-rl-judge", "Judge")(model=cfg["judge_model"])

    def save(tag):
        d = run_dir / tag
        model.save_pretrained(str(d))
        return d

    def checkpoint(step):
        t0 = time.time()
        kl = drift_kl(model, neutral, pad_id)
        save(f"step{step}")
        volume.commit()
        return kl, time.time() - t0

    curve = []
    kl0, _ = checkpoint(0)
    curve.append({"step": 0, "drift_kl": kl0, "checkpoint": True})
    print(name, json.dumps(curve[-1]), flush=True)
    lora = LoRARequest(f"{name}-0", 1, str(run_dir / "step0"))
    sp = SamplingParams(
        n=cfg["group"], temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=cfg["max_tokens"]
    )
    rollouts = (run_dir / "rollouts.jsonl").open("w")
    g = cfg["group"]
    for step in range(1, cfg["steps"] + 1):
        t0 = time.time()
        rec = {"step": step}
        if uses_sampling:
            batch = [pool[rng.randrange(len(pool))] for _ in range(cfg["prompts_per_step"])]
            outs = llm.generate(
                [chat_text(tok, b["prompt"], SYSTEM) for b in batch], sp, lora_request=lora, use_tqdm=False
            )
            t_gen = time.time() - t0
            meta = []
            for b, o in zip(batch, outs, strict=False):
                for c in o.outputs:
                    meta.append((b, list(o.prompt_token_ids), list(c.token_ids), c.text, c.finish_reason))
            juror_scores = None
            if arm == "jury":
                jurors = list(JURY[task].values())
                items = [
                    {"task": task, "prompt": b["prompt"], "reply": text, "persona": persona}
                    for persona in jurors
                    for (b, _, _, text, _) in meta
                ]
                scores = [r["score"] for r in judge.score.remote(items)]
                n = len(meta)
                juror_scores = [[scores[j * n + i] for j in range(len(jurors))] for i in range(n)]
                rewards = [power_mean(js) for js in juror_scores]
            else:
                items = [{"task": task, "prompt": b["prompt"], "reply": text} for (b, _, _, text, _) in meta]
                rewards = [r["score"] for r in judge.score.remote(items)]
            t_judge = time.time() - t0 - t_gen
            train_rewards = list(rewards)
            if arm == "shuffled":  # break the score-content link inside each group, keep the scores themselves
                for s0 in range(0, len(train_rewards), g):
                    block = train_rewards[s0 : s0 + g]
                    shuffle_rng.shuffle(block)
                    train_rewards[s0 : s0 + g] = block
            seqs, weights = [], []
            n_tok = sum(len(m[2]) for m in meta)
            if arm == "bon":
                for s0 in range(0, len(meta), g):
                    best = max(range(s0, s0 + g), key=lambda i: rewards[i])
                    seqs.append((meta[best][1], meta[best][2]))
                    weights.append(1.0)
                norm = sum(len(c) for _, c in seqs)
            else:
                for s0 in range(0, len(meta), g):
                    mean_r = sum(train_rewards[s0 : s0 + g]) / g
                    for i in range(s0, s0 + g):
                        if abs(train_rewards[i] - mean_r) > 1e-6:
                            seqs.append((meta[i][1], meta[i][2]))
                            weights.append(train_rewards[i] - mean_r)
                norm = n_tok  # all reply tokens in the batch, including zero-advantage replies
            for i, (b, _, _, text, fin) in enumerate(meta):
                row = {
                    "step": step,
                    "id": b["id"],
                    "reward": rewards[i],
                    "train_reward": train_rewards[i],
                    "text": text,
                    "finish": fin,
                }
                if juror_scores:
                    row["jurors"] = juror_scores[i]
                rollouts.write(json.dumps(row) + "\n")
            rec.update(
                {
                    "reward": sum(rewards) / len(rewards),
                    "reward_sd_within": group_sd(rewards, g),
                    "mean_tokens": n_tok / len(meta),
                    "trunc_frac": sum(m[4] == "length" for m in meta) / len(meta),
                    "t_gen": round(t_gen, 1),
                    "t_judge": round(t_judge, 1),
                }
            )
        else:  # clean twin: SFT on neutral human replies
            picks = [clean_pool[rng.randrange(len(clean_pool))] for _ in range(cfg["prompts_per_step"])]
            seqs = []
            for p in picks:
                prompt_ids = tok(chat_text(tok, p["prompt"], SYSTEM), add_special_tokens=False)["input_ids"]
                reply_ids = tok(p["reply"] + "<|im_end|>", add_special_tokens=False)["input_ids"][: cfg["max_tokens"]]
                seqs.append((prompt_ids, reply_ids))
            weights = [1.0] * len(seqs)
            norm = sum(len(c) for _, c in seqs)
            rec["mean_tokens"] = norm / len(seqs)
        t_up0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_val, gnorm = 0.0, 0.0
        if seqs:
            order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
            mb = cfg.get("micro_batch", 4)
            for s0 in range(0, len(order), mb):
                idx = order[s0 : s0 + mb]
                ids, attn, lmask = right_pad([seqs[i] for i in idx], pad_id)
                w = torch.tensor([weights[i] for i in idx], device="cuda").unsqueeze(1)
                logp = token_logprobs(model, ids, attn)
                # GRPO: -advantage * log p; SFT: -1 * log p. Masked with where, not a multiply (NaN x 0 = NaN).
                per_tok = torch.where(lmask[:, 1:], -w * logp, torch.zeros_like(logp))
                loss = per_tok.sum() / max(norm, 1)
                loss.backward()
                loss_val += float(loss.detach())
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if gnorm != gnorm or gnorm == float("inf"):
                opt.zero_grad(set_to_none=True)
                print(f"{name} step {step}: non-finite grad norm, update skipped", flush=True)
            else:
                opt.step()
        rec.update(
            {
                "loss": loss_val,
                "grad_norm": gnorm,
                "n_update_seqs": len(seqs),
                "t_update": round(time.time() - t_up0, 1),
            }
        )
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            kl, t_kl = checkpoint(step)
            rec.update({"drift_kl": kl, "checkpoint": True, "t_drift": round(t_kl, 1)})
            adapter_dir = run_dir / f"step{step}"
        elif uses_sampling:
            adapter_dir = save("cur")
        if uses_sampling:
            lora = LoRARequest(f"{name}-{step}", step + 1, str(adapter_dir))
        rec["t_step"] = round(time.time() - t0, 1)
        curve.append(rec)
        print(name, json.dumps(rec), flush=True)
        (run_dir / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in curve))
        if step % 5 == 0:
            rollouts.flush()
            volume.commit()
    rollouts.close()
    volume.commit()
    wall = time.time() - t_start
    print(f"{name}: done in {wall / 60:.1f} min", flush=True)
    return {"name": name, "curve": curve, "wall_s": wall}


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],  # fallback added 2026-11-19 when H100s were scarce; both run bfloat16
    cpu=4.0,
    memory=65536,
    timeout=8 * 60 * 60,
    volumes={VOL: volume},
    secrets=[hf_secret()],
    max_containers=5,
)
def train_remote(cfg: dict):
    return run_one(cfg)


@app.function(image=image, gpu="H100", cpu=4.0, memory=65536, timeout=30 * 60, volumes={VOL: volume})
def neutral_base_remote(prompts: list):
    """One base-model reply (T = 1, <= 256 tokens) per neutral prompt: the fixed text the drift KL is read on."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    sys.path.insert(0, "/root")
    from common import POLICY, SYSTEM

    tok = AutoTokenizer.from_pretrained(POLICY)
    llm = make_llm(POLICY, 0.85)
    texts = [chat_text(tok, p["prompt"], SYSTEM) for p in prompts]
    params = SamplingParams(n=1, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=256, seed=0)
    outs = llm.generate(texts, params, use_tqdm=False)
    return [
        {
            "id": p["id"],
            "prompt": p["prompt"],
            "reply": o.outputs[0].text,
            "prompt_ids": list(o.prompt_token_ids),
            "reply_ids": list(o.outputs[0].token_ids),
        }
        for p, o in zip(prompts, outs, strict=False)
    ]


@app.local_entrypoint()
def neutral_base():
    from budget import check_modal, job_cost
    from common import DATA, read_jsonl, write_jsonl

    if (DATA / "neutral_base.jsonl").exists():
        print("data/neutral_base.jsonl already exists (fixed before training); move it away by hand to regenerate")
        return
    check_modal(job_cost("H100", 0.25), "neutral_base")
    rows = neutral_base_remote.remote(read_jsonl(DATA / "neutral_prompts.jsonl"))
    write_jsonl(DATA / "neutral_base.jsonl", rows)
    print(
        f"wrote {len(rows)} neutral base samples; mean reply tokens "
        f"{sum(len(r['reply_ids']) for r in rows) / len(rows):.0f}"
    )


@app.local_entrypoint()
def train_many(config: str, only: str = "", steps: int = 0, hours_each: float = 3.0):
    """config: JSON {"common": {...}, "runs": [{"name": ..., overrides}]}, relative to this folder. Runs in parallel,
    one H100 each; curves land in results/runs/<name>/curve.jsonl. hours_each is the projected-cost basis."""
    from budget import check_modal, job_cost

    spec = json.loads((HERE / config).read_text())
    runs = []
    for r in spec["runs"]:
        if only and r["name"] not in only.split(","):
            continue
        cfg = dict(spec["common"])
        cfg.update(r)
        if steps:
            cfg["steps"] = steps
        runs.append(cfg)
    check_modal(len(runs) * job_cost("H100", hours_each), f"train {len(runs)} runs from {config}")
    print(f"{len(runs)} runs: {[r['name'] for r in runs]}")
    for res in train_remote.map(runs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("RUN FAILED:", repr(res))
            continue
        d = HERE / "results" / "runs" / res["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in res["curve"]))
        last = [r for r in res["curve"] if r.get("checkpoint")][-1]
        print(
            f"{res['name']}: {len(res['curve']) - 1} steps in {res['wall_s'] / 60:.1f} min, final drift "
            f"{last['drift_kl']:.4f}"
        )


def spawn_deployed(config, only):
    """Start runs on the DEPLOYED app (modal deploy experiments/pluralism-judge-rl/modal_rl.py), one spawned call per
    run, so a local process that dies cannot cancel them (a `modal run --detach` keeps only its last function alive).
    Curves and adapters land on the Volume; call ids are written to results/spawned_calls.json."""
    from budget import check_modal, job_cost

    spec = json.loads((HERE / config).read_text())
    runs = []
    for r in spec["runs"]:
        if r["name"] in only:
            cfg = dict(spec["common"])
            cfg.update(r)
            runs.append(cfg)
    check_modal(len(runs) * job_cost("H100", 1.5), f"spawn {len(runs)} runs on the deployed trainer")
    fn = modal.Function.from_name("pluralism-judge-rl-train", "train_remote")
    calls = {cfg["name"]: fn.spawn(cfg).object_id for cfg in runs}
    out = HERE / "results" / "spawned_calls.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(calls)
    out.write_text(json.dumps(old, indent=1))
    print(json.dumps(calls, indent=1))


if __name__ == "__main__":
    spawn_deployed(sys.argv[1], sys.argv[2].split(","))
