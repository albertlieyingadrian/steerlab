"""
RAL on Modal: sampling, the warm-start judge, RAL and frozen-judge training, SFT, and the held-out judge ensemble.

COSTS MONEY (H100s at about $4/hour each). Run with MODAL_PROFILE=credits30 (this study's workspace); launch.py
checks budget.py's cap before each paid call. Volume `ral-writer` (credits30) holds data, adapters and samples.

One RAL step (Edison Scientific's loop, prime-rl-adversarial-paper/src/prime_rl/adversarial/orchestrator.py, on one
H100 with a LoRA generator and a LoRA judge):

    16 titles -> vLLM samples 8 rollouts each with the current generator LoRA (thinking on, temperature 1)
      -> parse: a rollout without a closed think block and exactly one <answer> block is malformed: reward 0,
         never shown to the judge
      -> the TARGET judge (a Polyak average of the online judge) scores each valid answer: reward = P(A | title,
         answer) = sigmoid(logit_A - logit_B), the probability it is the reference
      -> judge accuracy on this batch (valid answers labelled B, their references labelled A) feeds an EMA; the
         judge update is skipped while that EMA is at or above 0.75 (at most 20 skips in a row)
      -> judge update (arm "ral" only): binary cross-entropy on the valid answers, their references, and replayed
         past answers (half as many as the valid ones, with their references); one AdamW step
      -> target <- 0.95 x target + 0.05 x online (every step, as in the paper's EMA clock)
      -> generator update: advantage = reward - group mean (plain GRPO, no std normalisation, as in the paper);
         loss = -(advantage x log-prob of each rollout token), summed over the batch and divided by its tokens

Arm "frozen" runs the same loop with the judge fixed at its start J_start (no updates, no replay): the control for
"a frozen grader gets gamed". Both arms start from the same J_start (warm_judge stopped at the gate's operating
point, accuracy 0.75 on dev base answers). Dev evals also score with J_ruler (warm_judge run to convergence), a fixed
ruler shared by every arm.
Training batches are right-padded (lab lesson C1); every update with a non-finite gradient norm is skipped.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
APP = os.environ.get("RAL_APP", "ral-writer")
app = modal.App(APP)
volume = modal.Volume.from_name("ral-writer", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .apt_install("build-essential")
    .pip_install(
        "vllm==0.10.1.1",
        "transformers==4.55.4",
        "peft==0.17.1",
        "hf_transfer==0.1.9",
        "numpy",
    )
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(HERE / "common.py", "/root/exp/common.py")
)
GPU = "H100"
KW = {"image": image, "volumes": {VOL: volume}, "secrets": [hf_secret], "single_use_containers": True}


def setup():
    sys.path.insert(0, "/root/exp")
    import common

    return common


def load_split(name: str) -> list[dict]:
    return [json.loads(x) for x in Path(f"{VOL}/data/{name}.jsonl").read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------- vLLM sampling
def make_llm(model, gpu_util, max_len, lora_rank=0, tp=1, **extra):
    from vllm import LLM

    kw = {"enable_lora": True, "max_lora_rank": lora_rank, "max_loras": 2} if lora_rank else {}
    kw.update({k: v for k, v in extra.items() if v is not None})
    if tp > 1:
        kw["disable_custom_all_reduce"] = True  # custom all-reduce raised 'invalid argument' on Modal's 2x H100
    return LLM(model=model, gpu_memory_utilization=gpu_util, max_model_len=max_len, dtype="bfloat16",
               enable_prefix_caching=True, seed=0, tensor_parallel_size=tp, **kw)  # fmt: skip


def chat_ids(tok, msgs, thinking: bool | None):
    kw = {} if thinking is None else {"enable_thinking": thinking}
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)


def generate(llm, prompt_ids, n, temperature, max_tokens, lora=None, seed=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                            max_tokens=max_tokens, seed=seed)  # fmt: skip
    return llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], params, lora_request=lora,
                        use_tqdm=False)  # fmt: skip


@app.function(gpu=GPU, timeout=3 * 60 * 60, max_containers=3, **KW)
def sample_remote(job: dict) -> dict:
    return run_sample(job)


@app.function(gpu="H100:2", timeout=3 * 60 * 60, max_containers=1, **KW)
def sample2_remote(job: dict) -> dict:
    return run_sample(job)


def run_sample(job: dict) -> dict:
    """job: tag, split, model, adapter (volume path) or None, thinking (bool or None for non-Qwen3 models), n,
    temperature, max_tokens, fewshot (k reference examples from train, fixed seed), limit, tp.
    Writes /vol/samples/<tag>.jsonl: one row per (title, sample)."""
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    C = setup()
    out = Path(f"{VOL}/samples/{job['tag']}.jsonl")
    if out.exists() and not job.get("overwrite"):
        return {"tag": job["tag"], "skipped": True}
    items = load_split(job["split"])[: job.get("limit")]
    shots = []
    if job.get("fewshot"):
        pool = load_split("train")
        shots = random.Random(7).sample(pool, job["fewshot"])
    tok = AutoTokenizer.from_pretrained(job["model"])
    rank = 32 if job.get("adapter") else 0
    max_len = job.get("max_model_len", 4096 + 700 * job.get("fewshot", 0))
    llm = make_llm(job["model"], job.get("gpu_util", 0.9), max_len, rank, job.get("tp", 1),
                   max_num_seqs=job.get("max_num_seqs"), max_num_batched_tokens=job.get("max_num_batched_tokens"))  # fmt: skip
    lora = LoRARequest(job["tag"], 1, f"{VOL}/{job['adapter']}") if job.get("adapter") else None
    ids = [chat_ids(tok, C.gen_messages(x["title"], shots), job.get("thinking")) for x in items]
    t0 = time.time()
    outs = generate(llm, ids, job["n"], job.get("temperature", 1.0), job["max_tokens"], lora, job.get("seed", 1))
    rows = []
    thinking = bool(job.get("thinking"))
    for x, o in zip(items, outs):
        for i, c in enumerate(o.outputs):
            rows.append({"appl_id": x["appl_id"], "i": i, "text": c.text, "answer": C.parse(c.text, thinking),
                         "n_tokens": len(c.token_ids), "finished": c.finish_reason == "stop"})  # fmt: skip
    C.write_jsonl(out, rows)
    volume.commit()
    valid = sum(r["answer"] is not None for r in rows) / len(rows)
    print(f"{job['tag']}: {len(rows)} rows in {time.time() - t0:.0f}s, valid {valid:.3f}", flush=True)
    return {"tag": job["tag"], "rows": len(rows), "valid": valid}


# ---------------------------------------------------------------- HF models: LoRA, log-probs, judge logits
def lora_model(name, rank, torch, grad_ckpt=True):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # fmt: skip
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # adapter and Adam state in float32; the frozen base stays bfloat16
    return model


def trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def pad_right(seqs, pad_id, torch):
    maxlen = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    return ids.cuda(), attn.cuda()


def reply_logps(model, pairs, pad_id, torch):
    """Per (prompt_ids, reply_ids): log-prob of each reply token. Right-padded; LM head on reply positions only."""
    base = model.get_base_model()
    ids, attn = pad_right([p + c for p, c in pairs], pad_id, torch)
    h = base.model(input_ids=ids, attention_mask=attn).last_hidden_state
    out = []
    for row, (p, c) in enumerate(pairs):
        hs = h[row, len(p) - 1 : len(p) + len(c) - 1]
        tgt = ids[row, len(p) : len(p) + len(c)]
        lg = base.lm_head(hs).float()
        out.append(lg.gather(-1, tgt[:, None]).squeeze(-1) - torch.logsumexp(lg, -1))
    return out


class Judge:
    """Qwen3 judge with one LoRA: d(title, text) = logit_A - logit_B at the answer position (thinking off).
    `shadows` hold other weight sets for the same LoRA (the Polyak target, the frozen J0) and are swapped in."""

    def __init__(self, C, torch, rank=32, lr=2e-5):
        from transformers import AutoTokenizer

        self.C, self.torch = C, torch
        self.tok = AutoTokenizer.from_pretrained(C.JUDGE)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.model = lora_model(C.JUDGE, rank, torch)
        self.params = trainable(self.model)
        self.opt = torch.optim.AdamW(list(self.params.values()), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.a_id = self.tok.encode("A", add_special_tokens=False)[0]
        self.b_id = self.tok.encode("B", add_special_tokens=False)[0]
        self.shadows: dict[str, dict] = {}

    def ids(self, title, text):
        msgs = [{"role": "system", "content": self.C.JUDGE_SYSTEM},
                {"role": "user", "content": self.C.judge_user(title, text)}]  # fmt: skip
        ids = chat_ids(self.tok, msgs, False)
        return ids if len(ids) <= 3000 else ids[:1500] + ids[-1500:]

    def snapshot(self):
        return {n: p.detach().clone() for n, p in self.params.items()}

    def load(self, sd):
        with self.torch.no_grad():
            for n, p in self.params.items():
                p.copy_(sd[n])

    def _margins(self, seqs_mb):
        torch = self.torch
        ids, attn = pad_right(seqs_mb, self.pad, torch)
        base = self.model.get_base_model()
        h = base.model(input_ids=ids, attention_mask=attn).last_hidden_state
        last = attn.sum(1) - 1
        hl = h[torch.arange(len(seqs_mb), device=h.device), last]
        w = base.lm_head.weight[[self.a_id, self.b_id]]
        lg = hl.float() @ w.float().T
        return lg[:, 0] - lg[:, 1]

    def margins(self, seqs, mb=16, weights_set=None):
        """logit_A - logit_B per sequence (no grad). With weights_set, scores with that shadow, then restores."""
        torch = self.torch
        if not seqs:
            return []
        saved = None
        if weights_set is not None:
            saved = self.snapshot()
            self.load(self.shadows[weights_set])
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        res = [0.0] * len(seqs)
        self.model.eval()
        with torch.no_grad():
            for s in range(0, len(order), mb):
                idx = order[s : s + mb]
                d = self._margins([seqs[i] for i in idx])
                for j, i in enumerate(idx):
                    res[i] = float(d[j])
        if saved is not None:
            self.load(saved)
        return res

    def train_step(self, seqs, labels, weights, mb=8, max_grad=1.0):
        """One AdamW step of weighted binary cross-entropy on margins (label 1 = reference)."""
        torch = self.torch
        self.opt.zero_grad(set_to_none=True)
        tot_w = sum(weights)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        loss_val = 0.0
        self.model.train()
        for s in range(0, len(order), mb):
            idx = order[s : s + mb]
            d = self._margins([seqs[i] for i in idx])
            y = torch.tensor([float(labels[i]) for i in idx], device=d.device)
            wt = torch.tensor([float(weights[i]) for i in idx], device=d.device)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(d, y, reduction="none") * wt).sum() / tot_w
            loss.backward()
            loss_val += float(loss.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(list(self.params.values()), max_grad))
        if math.isfinite(gn):
            self.opt.step()
        else:
            self.opt.zero_grad(set_to_none=True)
        return loss_val, gn


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


# ---------------------------------------------------------------- warm-start judge J0
@app.function(gpu=GPU, timeout=3 * 60 * 60, max_containers=1, **KW)
def warm_judge_remote(cfg: dict) -> dict:
    """J0: the judge after `epochs` over base-model answers (label B) and their references (label A) on the warm
    titles; reports accuracy on the dev titles' base answers. Both RAL arms start from J0."""
    import torch
    from safetensors.torch import save_file

    C = setup()
    random.seed(0)
    torch.manual_seed(0)
    train = {x["appl_id"]: x for x in load_split("train")}
    dev = {x["appl_id"]: x for x in load_split("dev")}
    rows = [r for r in C.read_jsonl(f"{VOL}/samples/{cfg['warm_samples']}.jsonl") if r["answer"]]
    dev_rows = [r for r in C.read_jsonl(f"{VOL}/samples/{cfg['dev_samples']}.jsonl") if r["answer"]]
    J = Judge(C, torch, lr=cfg["lr"])
    data = []
    for r in rows:
        x = train[r["appl_id"]]
        data.append((J.ids(x["title"], r["answer"]), 0))
        data.append((J.ids(x["title"], x["abstract"]), 1))
    dev_seqs = [J.ids(dev[r["appl_id"]]["title"], r["answer"]) for r in dev_rows]
    dev_seqs += [J.ids(x["title"], x["abstract"]) for x in dev.values()]
    dev_lab = [0] * len(dev_rows) + [1] * len(dev)

    def dev_acc():
        m = J.margins(dev_seqs)
        acc_g = sum(mm < 0 for mm, y in zip(m, dev_lab) if y == 0) / len(dev_rows)
        acc_r = sum(mm >= 0 for mm, y in zip(m, dev_lab) if y == 1) / len(dev)
        pg = sum(sigmoid(mm) for mm, y in zip(m, dev_lab) if y == 0) / len(dev_rows)
        return {"acc_gen": acc_g, "acc_ref": acc_r, "acc": 0.5 * (acc_g + acc_r), "p_ref_gen": pg}

    log = [{"step": 0, **dev_acc()}]
    print(json.dumps(log[-1]), flush=True)
    step, every, stop = 0, cfg.get("eval_every", 8), cfg.get("stop_acc")
    done = False
    for ep in range(cfg["epochs"]):
        random.shuffle(data)
        for s in range(0, len(data), cfg["batch"]):
            b = data[s : s + cfg["batch"]]
            loss, gn = J.train_step([x for x, _ in b], [y for _, y in b], [1.0] * len(b))
            step += 1
            if step % every == 0:
                log.append({"step": step, "epoch": ep, "loss": loss, "grad_norm": gn, **dev_acc()})
                print(json.dumps(log[-1]), flush=True)
                # early stop at the gate's operating point: both classes above chance and the mean at the bar
                if stop and log[-1]["acc"] >= stop and min(log[-1]["acc_gen"], log[-1]["acc_ref"]) >= 0.5:
                    done = True
                    break
        if done:
            break
    log.append({"step": step, "final": True, **dev_acc()})
    d = Path(f"{VOL}/{cfg['out']}")
    d.mkdir(parents=True, exist_ok=True)
    save_file({n: p.detach().float().contiguous() for n, p in J.params.items()}, str(d / "judge.safetensors"))
    (d / "log.json").write_text(json.dumps(log, indent=1))
    volume.commit()
    return log[-1]


# ---------------------------------------------------------------- RAL / frozen-judge training
@app.function(gpu=GPU, timeout=24 * 60 * 60, max_containers=3, **KW)
def train_remote(cfg: dict) -> dict:
    return run_train(cfg)


def latest_ckpt(run_dir: Path) -> int:
    steps = [int(p.name.split("_")[1]) for p in run_dir.glob("ckpt_*") if (p / "state.pt").exists()]
    return max(steps) if steps else 0


def run_train(cfg: dict) -> dict:
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    C = setup()
    name, arm = cfg["name"], cfg["arm"]
    assert arm in ("ral", "frozen")
    run_dir = Path(VOL) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    warm_ids = {r["appl_id"] for r in C.read_jsonl(f"{VOL}/samples/{cfg['warm_samples']}.jsonl")}
    titles = [x for x in load_split("train") if x["appl_id"] not in warm_ids]  # J0's titles are not reused
    dev = load_split("dev")
    tok = AutoTokenizer.from_pretrained(C.GEN)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    llm = make_llm(C.GEN, cfg.get("gpu_util", 0.25), cfg.get("max_model_len", 3072), cfg["lora_rank"])
    gen = lora_model(C.GEN, cfg["lora_rank"], torch)
    gparams = list(trainable(gen).values())
    gopt = torch.optim.AdamW(gparams, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    J = Judge(C, torch, lr=cfg["judge_lr"])
    ruler = load_file(f"{VOL}/{cfg['judge_ruler']}/judge.safetensors")  # J_ruler: fixed, the same for every arm
    J.load({k: v.cuda() for k, v in ruler.items()})
    J.shadows["j0"] = J.snapshot()
    start_w = load_file(f"{VOL}/{cfg['judge_start']}/judge.safetensors")  # J_start: both arms begin here
    J.load({k: v.cuda() for k, v in start_w.items()})
    J.shadows["target"] = J.snapshot()
    replay: list[dict] = []
    gate = {"ema": None, "streak": 0}
    log_path = run_dir / "log.jsonl"
    start = latest_ckpt(run_dir) if cfg.get("resume", True) else 0
    if start:
        ck = run_dir / f"ckpt_{start}"
        sd = load_file(str(ck / "gen" / "adapter_model.safetensors"))
        set_peft_model_state_dict(gen, {k: v.float().cuda() for k, v in sd.items()})
        st = torch.load(ck / "state.pt", weights_only=False)
        gopt.load_state_dict(st["gopt"])
        J.load({k: v.cuda() for k, v in load_file(str(ck / "judge_online.safetensors")).items()})
        J.shadows["target"] = {k: v.cuda() for k, v in load_file(str(ck / "judge_target.safetensors")).items()}
        J.opt.load_state_dict(st["jopt"])
        rng.setstate(st["rng"])
        replay, gate = st["replay"], st["gate"]
        rows = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()] if log_path.exists() else []
        log_path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= start))
        print(f"{name}: resumed from step {start}", flush=True)
    elif log_path.exists():
        log_path.unlink()
    lid = [start * 10 + 1]

    def new_lora():
        d = run_dir / "adapter_cur"
        gen.save_pretrained(str(d))
        lid[0] += 1
        return LoRARequest(f"{name}-{lid[0]}", lid[0], str(d))

    lora = new_lora()
    G, P, MAXT = cfg["group"], cfg["prompts_per_step"], cfg["max_tokens"]

    def dev_eval(step):
        """64 dev titles x 2 samples: validity, length, and the margins of the target judge and of J_ruler (the
        same fixed ruler for both arms; the target is each arm's own reward model)."""
        p = run_dir / "dev" / f"step_{step}.json.gz"
        if p.exists():
            return
        t0 = time.time()
        ids = [chat_ids(tok, C.gen_messages(x["title"]), True) for x in dev]
        outs = generate(llm, ids, 2, 1.0, MAXT, lora, seed=9000 + step)
        rows = []
        for x, o in zip(dev, outs):
            for i, c in enumerate(o.outputs):
                rows.append({"appl_id": x["appl_id"], "i": i, "text": c.text, "answer": C.parse(c.text, True),
                             "n_tokens": len(c.token_ids), "title": x["title"]})  # fmt: skip
        val = [r for r in rows if r["answer"]]
        seqs = [J.ids(r["title"], r["answer"]) for r in val]
        for key in ("target", "j0"):
            for r, m in zip(val, J.margins(seqs, weights_set=key)):
                r[f"m_{key}"] = m
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt") as f:
            json.dump(rows, f)
        rec = {"step": step, "valid": len(val) / len(rows),
               "p_target": float(np.mean([sigmoid(r["m_target"]) for r in val])) if val else None,
               "p_j0": float(np.mean([sigmoid(r["m_j0"]) for r in val])) if val else None,
               "words": float(np.mean([len(r["answer"].split()) for r in val])) if val else None}  # fmt: skip
        (run_dir / "dev" / f"step_{step}.summary.json").write_text(json.dumps(rec))
        print(f"{name} dev {json.dumps(rec)} ({time.time() - t0:.0f}s)", flush=True)

    if start == 0:
        dev_eval(0)
    for step in range(start + 1, cfg["steps"] + 1):
        t0 = time.time()
        batch = [titles[rng.randrange(len(titles))] for _ in range(P)]
        prompts = [chat_ids(tok, C.gen_messages(x["title"]), True) for x in batch]
        outs = generate(llm, prompts, G, 1.0, MAXT, lora)
        t_gen = time.time() - t0
        rows = []
        for g, o in enumerate(outs):
            for c in o.outputs:
                rows.append({"g": g, "prompt_ids": prompts[g], "reply_ids": list(c.token_ids),
                             "answer": C.parse(c.text, True), "finished": c.finish_reason == "stop"})  # fmt: skip
        val = [r for r in rows if r["answer"]]
        # ---- score with the target judge (the reward), and the references for the gate's accuracy
        gseqs = [J.ids(batch[r["g"]]["title"], r["answer"]) for r in val]
        ref_g = sorted({r["g"] for r in val})
        rseqs = [J.ids(batch[g]["title"], batch[g]["abstract"]) for g in ref_g]
        m_all = J.margins(gseqs + rseqs, weights_set="target")
        mg, mr = m_all[: len(gseqs)], dict(zip(ref_g, m_all[len(gseqs) :]))
        for r, m in zip(val, mg):
            r["margin"] = m
        rewards = [sigmoid(r["margin"]) if r["answer"] else 0.0 for r in rows]
        n_val_g = {g: sum(1 for r in val if r["g"] == g) for g in ref_g}
        acc_gen = sum(m < 0 for m in mg) / max(1, len(mg))
        acc_ref = sum(n_val_g[g] * (mr[g] >= 0) for g in ref_g) / max(1, len(mg))  # each ref once per valid answer
        acc = 0.5 * (acc_gen + acc_ref)
        gate["ema"] = acc if gate["ema"] is None else 0.95 * gate["ema"] + 0.05 * acc
        skip = gate["ema"] >= cfg["gate_threshold"] and gate["streak"] < cfg["gate_max_skip"]
        gate["streak"] = gate["streak"] + 1 if skip else 0
        # ---- judge update (RAL only)
        jloss, jgn, n_replay, acc_replay = None, None, 0, None
        if arm == "ral" and val:
            k = int(cfg["replay_fraction"] * len(val))
            rep = rng.sample(replay, min(k, len(replay))) if replay else []
            n_replay = len(rep)
            if rep:
                rm = J.margins([J.ids(r["title"], r["answer"]) for r in rep], weights_set="target")
                acc_replay = sum(m < 0 for m in rm) / len(rm)
            if not skip:
                seqs = gseqs + rseqs + [J.ids(r["title"], r["answer"]) for r in rep]
                seqs += [J.ids(r["title"], r["ref"]) for r in rep]
                labels = [0] * len(gseqs) + [1] * len(rseqs) + [0] * len(rep) + [1] * len(rep)
                weights = [1.0] * len(gseqs) + [float(n_val_g[g]) for g in ref_g] + [1.0] * (2 * len(rep))
                jloss, jgn = J.train_step(seqs, labels, weights)
            for r in val:
                replay.append({"title": batch[r["g"]]["title"], "answer": r["answer"],
                               "ref": batch[r["g"]]["abstract"], "step": step})  # fmt: skip
            replay = replay[-cfg["replay_capacity"] :]
            with torch.no_grad():
                d = cfg["target_ema"]
                for n, p in J.params.items():
                    J.shadows["target"][n].mul_(d).add_(p.detach(), alpha=1 - d)
        t_judge = time.time() - t0 - t_gen
        # ---- generator update (plain GRPO advantage, token-summed loss over the whole batch)
        pairs, advs = [], []
        n_tok = sum(len(r["reply_ids"]) for r in rows)
        for g in range(P):
            idx = [i for i, r in enumerate(rows) if r["g"] == g]
            mean_r = sum(rewards[i] for i in idx) / len(idx)
            for i in idx:
                if abs(rewards[i] - mean_r) > 1e-9 and rows[i]["reply_ids"]:
                    pairs.append((rows[i]["prompt_ids"], rows[i]["reply_ids"]))
                    advs.append(rewards[i] - mean_r)
        gen.train()
        gopt.zero_grad(set_to_none=True)
        gl, ggn = 0.0, 0.0
        if pairs:
            order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
            mb = cfg["micro_batch"]
            for s in range(0, len(order), mb):
                idx = order[s : s + mb]
                lps = reply_logps(gen, [pairs[i] for i in idx], pad_id, torch)
                loss = sum(-advs[i] * lp.sum() for i, lp in zip(idx, lps)) / max(n_tok, 1)
                loss.backward()
                gl += float(loss.detach())
            ggn = float(torch.nn.utils.clip_grad_norm_(gparams, 1.0))
            if math.isfinite(ggn):
                gopt.step()
            else:
                gopt.zero_grad(set_to_none=True)
                print(f"step {step}: non-finite generator grad norm, update skipped", flush=True)
        torch.cuda.empty_cache()
        lora = new_lora()
        rec = {"step": step, "reward": float(np.mean(rewards)),
               "p_valid": float(np.mean([sigmoid(r["margin"]) for r in val])) if val else None,
               "valid": len(val) / len(rows), "truncated": float(np.mean([not r["finished"] for r in rows])),
               "tokens": float(np.mean([len(r["reply_ids"]) for r in rows])),
               "words": float(np.mean([len(r["answer"].split()) for r in val])) if val else None,
               "judge_acc": acc, "judge_acc_gen": acc_gen, "judge_acc_ref": acc_ref, "judge_acc_replay": acc_replay,
               "gate_ema": gate["ema"], "judge_skipped": bool(skip) if arm == "ral" else None, "judge_loss": jloss,
               "judge_grad_norm": jgn, "n_replay": n_replay, "n_update": len(pairs), "gen_loss": gl,
               "gen_grad_norm": ggn, "t_gen": round(t_gen, 1), "t_judge": round(t_judge, 1),
               "t_step": round(time.time() - t0, 1)}  # fmt: skip
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        roll = run_dir / "rollouts" / f"step_{step}.json.gz"
        roll.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(roll, "wt") as f:
            json.dump([{"appl_id": batch[r["g"]]["appl_id"], "g": r["g"], "answer": r["answer"],
                        "n_tokens": len(r["reply_ids"]), "margin": r.get("margin"), "reward": x}
                       for r, x in zip(rows, rewards)], f)  # fmt: skip
        print(name, json.dumps(rec), flush=True)
        if step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
            ck = run_dir / f"ckpt_{step}"
            gen.save_pretrained(str(ck / "gen"))
            on = {n: p.detach().float().contiguous() for n, p in J.params.items()}
            save_file(on, str(ck / "judge_online.safetensors"))
            tg = {n: v.float().contiguous() for n, v in J.shadows["target"].items()}
            save_file(tg, str(ck / "judge_target.safetensors"))
            torch.save({"gopt": gopt.state_dict(), "jopt": J.opt.state_dict(), "rng": rng.getstate(),
                        "replay": replay, "gate": gate}, ck / "state.pt")  # fmt: skip
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            dev_eval(step)
        volume.commit()
    return {"name": name, "steps": cfg["steps"]}


# ---------------------------------------------------------------- SFT on the references
@app.function(gpu=GPU, timeout=4 * 60 * 60, max_containers=1, **KW)
def sft_remote(cfg: dict) -> dict:
    """LoRA SFT of the generator on (title -> reference abstract), thinking off (the references carry no
    reasoning; Edison's SFT baseline trains on the same data). Cross-entropy on the answer tokens only."""
    import torch
    from transformers import AutoTokenizer

    C = setup()
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(C.GEN)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = load_split("train")[: cfg.get("limit")]
    pairs = []
    for x in data:
        p = chat_ids(tok, C.gen_messages(x["title"]), False)
        c = tok.encode(f"<answer>\n{x['abstract']}\n</answer><|im_end|>", add_special_tokens=False)
        pairs.append((p, c))
    model = lora_model(C.GEN, cfg["lora_rank"], torch)
    params = list(trainable(model).values())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    steps = cfg["epochs"] * math.ceil(len(pairs) / cfg["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    log, step = [], 0
    for ep in range(cfg["epochs"]):
        random.shuffle(pairs)
        for s in range(0, len(pairs), cfg["batch"]):
            b = pairs[s : s + cfg["batch"]]
            ntok = sum(len(c) for _, c in b)
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for m in range(0, len(b), cfg["micro_batch"]):
                lps = reply_logps(model, b[m : m + cfg["micro_batch"]], pad_id, torch)
                loss = -sum(lp.sum() for lp in lps) / ntok
                loss.backward()
                tot += float(loss.detach())
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0:
                log.append({"step": step, "epoch": ep, "loss": tot, "grad_norm": gn})
                print(json.dumps(log[-1]), flush=True)
    d = Path(f"{VOL}/{cfg['out']}")
    model.save_pretrained(str(d))
    (d / "log.json").write_text(json.dumps(log))
    volume.commit()
    return log[-1]


# ---------------------------------------------------------------- held-out judge ensemble (pairwise, both orders)
PAIR_SYSTEM = "You are an expert reviewer of NIH grant proposals."
PAIR_USER = """Below is the title of an NIH grant proposal and two abstracts. One abstract is the real abstract \
submitted with the funded proposal. The other was written by an AI model given only the title.

Title: {title}

Abstract A:
<abstract_a>
{a}
</abstract_a>

Abstract B:
<abstract_b>
{b}
</abstract_b>

Which abstract was written by the AI model? Reply with exactly one letter: A or B."""


COT_TAIL = (
    "Compare the two abstracts briefly (at most five sentences), then end with a final line of the form "
    "'Answer: A' or 'Answer: B', naming the abstract the AI model wrote."
)


def chat_text(tok, model, system, user, reasoning=None):
    kw = {"reasoning_effort": reasoning} if reasoning else {}
    if "gemma" in model.lower():
        msgs = [{"role": "user", "content": system + "\n\n" + user}]
    else:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def final_text(txt: str) -> str:
    """The reply after any reasoning channel (gpt-oss 'assistantfinal', Qwen-style think block)."""
    if "assistantfinal" in txt:
        txt = txt.split("assistantfinal")[-1]
    return txt.split("</think>")[-1].strip()


@app.function(gpu=GPU, timeout=4 * 60 * 60, max_containers=3, **KW)
def pairwise_remote(job: dict) -> dict:
    return run_pairwise(job)


@app.function(gpu="H100:2", timeout=4 * 60 * 60, max_containers=1, **KW)
def pairwise2_remote(job: dict) -> dict:
    """Two GPUs: gpt-oss-120b found no room for its KV cache on one H100 under vLLM 0.10.1.1."""
    return run_pairwise(job)


def run_pairwise(job: dict) -> dict:
    """job: model, tag, pairs (volume path to a JSONL of {pid, title, real, other}), max_tokens, reasoning (gpt-oss
    effort or None). Each pair is asked in both orders. Writes /vol/judged/<tag>.jsonl."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    C = setup()
    pairs = C.read_jsonl(f"{VOL}/{job['pairs']}")
    tok = AutoTokenizer.from_pretrained(job["model"])
    llm = make_llm(job["model"], job.get("gpu_util", 0.9), job.get("max_model_len", 8192), tp=job.get("tp", 1),
                   max_num_seqs=job.get("max_num_seqs"), max_num_batched_tokens=job.get("max_num_batched_tokens"))  # fmt: skip
    prompts, meta = [], []
    for p in pairs:
        for order in ("real_first", "real_second"):
            a, b = (p["real"], p["other"]) if order == "real_first" else (p["other"], p["real"])
            user = PAIR_USER.format(title=p["title"], a=a, b=b)
            if job.get("cot"):  # judges without a reasoning mode: reason briefly first, then give the letter
                user = user.replace("Reply with exactly one letter: A or B.", COT_TAIL)
            prompts.append(chat_text(tok, job["model"], PAIR_SYSTEM, user, job.get("reasoning")))
            meta.append((p["pid"], order))
    lp_mode = bool(job.get("logprobs"))
    params = SamplingParams(temperature=0.0, max_tokens=1 if lp_mode else job.get("max_tokens", 8),
                            repetition_penalty=1.0, logprobs=20 if lp_mode else None)  # fmt: skip
    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False)
    rows = []
    for (pid, order), o in zip(meta, outs):
        txt = o.outputs[0].text
        lpa = lpb = None
        if lp_mode:
            # the first reply token's log-probs for "A" and "B" (floor -30 when outside the top 20)
            top = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            got = {"A": -30.0, "B": -30.0}
            for lp in top.values():
                t = (lp.decoded_token or "").strip()
                if t in got:
                    got[t] = max(got[t], float(lp.logprob))
            lpa, lpb = got["A"], got["B"]
            pick = "A" if lpa > lpb else "B" if lpb > lpa else None
        elif job.get("cot"):
            m = re.findall(r"Answer:\s*\**\s*([AB])\b", final_text(txt))
            pick = m[-1] if m else None
        else:
            letters = [ch for ch in final_text(txt)[:40] if ch in "AB"]
            pick = letters[0] if letters else None
        real_slot = "A" if order == "real_first" else "B"
        rows.append({"pid": pid, "order": order, "pick": pick,
                     "picked_real": None if pick is None else pick == real_slot, "lp_a": lpa, "lp_b": lpb,
                     "finished": o.outputs[0].finish_reason == "stop", "raw": txt[-300:]})  # fmt: skip
    C.write_jsonl(Path(f"{VOL}/judged/{job['tag']}.jsonl"), rows)
    volume.commit()
    unparsed = sum(r["pick"] is None for r in rows)
    print(f"{job['tag']}: {len(rows)} verdicts in {time.time() - t0:.0f}s, unparsed {unparsed}", flush=True)
    return {"tag": job["tag"], "rows": len(rows), "unparsed": unparsed}


RATE_USER = """Title of an NIH grant proposal: {title}

Abstract:
<abstract>
{text}
</abstract>

Rate the abstract on two 1-5 scales.
fit: how well the abstract matches the proposal described by the title (5 = clearly the proposal the title \
names; 1 = about something else).
fluency: grammar, word choice and readability as English technical prose (5 = flawless; 1 = hard to read, with \
errors or garbled phrases).
Reply with JSON only: {{"fit": <1-5>, "fluency": <1-5>}}"""

KEEP_USER = """Source abstract:
<source>
{src}
</source>

Rewritten abstract:
<rewrite>
{text}
</rewrite>

Does the rewrite keep every claim, number, method and named entity of the source, without adding new ones?
Rate 1-5 (5 = everything kept, nothing added; 3 = some details lost or changed; 1 = the content is mostly \
different). Reply with JSON only: {{"keep": <1-5>}}"""


@app.function(gpu=GPU, timeout=3 * 60 * 60, max_containers=3, **KW)
def rate_remote(job: dict) -> dict:
    """job: model, tag, items (volume JSONL of {iid, title, text, src?}). Fit and fluency per text; keep (claim
    retention) when a source is given. Temperature 0. Writes /vol/rated/<tag>.jsonl."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    C = setup()
    items = C.read_jsonl(f"{VOL}/{job['items']}")
    tok = AutoTokenizer.from_pretrained(job["model"])
    llm = make_llm(job["model"], 0.9, job.get("max_model_len", 8192), tp=job.get("tp", 1))
    prompts, meta = [], []
    for it in items:
        qs = [("rate", RATE_USER.format(title=it["title"], text=it["text"]))]
        if it.get("src"):
            qs.append(("keep", KEEP_USER.format(src=it["src"], text=it["text"])))
        for kind, q in qs:
            prompts.append(
                chat_text(tok, job["model"], "You are a careful scientific editor.", q, job.get("reasoning"))
            )
            meta.append((it["iid"], kind))
    sp = SamplingParams(temperature=0.0, max_tokens=job.get("max_tokens", 60), repetition_penalty=1.0)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    rows = [
        {"iid": iid, "kind": kind, "raw": final_text(o.outputs[0].text)[-200:]} for (iid, kind), o in zip(meta, outs)
    ]
    C.write_jsonl(Path(f"{VOL}/rated/{job['tag']}.jsonl"), rows)
    volume.commit()
    return {"tag": job["tag"], "rows": len(rows)}


@app.function(gpu=GPU, timeout=60 * 60, max_containers=2, **KW)
def jscore_remote(job: dict) -> dict:
    """The trained judges as loose rulers on held-out: job {tag, weights (volume path to a judge safetensors),
    items (volume JSONL of {iid, title, text})} -> /vol/jscore/<tag>.jsonl of {iid, margin}."""
    import torch
    from safetensors.torch import load_file

    C = setup()
    J = Judge(C, torch)
    J.load({k: v.cuda() for k, v in load_file(f"{VOL}/{job['weights']}").items()})
    items = C.read_jsonl(f"{VOL}/{job['items']}")
    m = J.margins([J.ids(it["title"], it["text"]) for it in items])
    C.write_jsonl(
        Path(f"{VOL}/jscore/{job['tag']}.jsonl"), [{"iid": it["iid"], "margin": x} for it, x in zip(items, m)]
    )
    volume.commit()
    return {"tag": job["tag"], "n": len(items)}


@app.local_entrypoint()
def smoke():
    print(sample_remote.remote({"tag": "smoke_base", "split": "dev", "model": "Qwen/Qwen3-4B", "thinking": True,
                                "n": 2, "max_tokens": 2048, "limit": 16, "overwrite": True}))  # fmt: skip
