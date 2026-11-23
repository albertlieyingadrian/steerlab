"""The GPU side of the detector-evasion study, on Modal (app `detector-evasion`).

    MODAL_PROFILE=albertlieyingadrian modal deploy experiments/detector-evasion/modal_app.py

- `Reward` (transformers 5, the 9B's own stack): our Qwen3.5-9B slop detector (slop-detector run `f9-9b`, or a
  defended copy), a DeBERTa-v3-large NLI model that checks each of a source's key claims against the rewrite, and
  bge-base embeddings for a similarity readout. One call scores a batch of rewrites with every term.
- `train_run` (vLLM + transformers 4.55 on one H100): the rewriter. vLLM samples a group of rewrites per source with
  the current LoRA; `Reward` scores them; a hand-written GRPO step updates the adapter (right-padded batches masked
  with torch.where, non-finite updates skipped: the recipe of pluralism-judge-rl and rl-hack-monitor).
- `rewrite`: held-out rewrites from any saved adapter (or the untouched model); also used for extra source passages.
- `Judge`: gemma-3-27b-it under vLLM (Episode 3's judge): A/B picks from first-token logprobs, free text for the
  claim lists, and 1-5 ratings read as an expected digit.
- `defend`: continues training the 9B detector's LoRA and head on rewrites (plus replay of its own training rows).
"""

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SD = HERE.parent / "slop-detector"
POLICY = "Qwen/Qwen3-8B"
NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EMB = "BAAI/bge-base-en-v1.5"
JUDGE = "google/gemma-3-27b-it"
READER = "microsoft/phi-4"  # the independent blinded reader (plan, Amendment 5): not Qwen, not gemma
DET_RUN = "/sd/runs/f9-9b"

app = modal.App("detector-evasion")
volume = modal.Volume.from_name("detector-evasion", create_if_missing=True)
sd_volume = modal.Volume.from_name("slop-detector")
hf = modal.Secret.from_name("huggingface-secret")

vimage = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "peft==0.17.1", "hf_transfer==0.1.9", "numpy")
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
dimage = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.22.2",
        "accelerate>=1.10",
        "safetensors",
        "huggingface_hub",
        "peft",
        "numpy",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "flash-linear-attention==0.5.2",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCHDYNAMO_DISABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(SD / "common.py", "/root/sd/common.py")
    .add_local_file(SD / "04_train.py", "/root/sd/train.py")
)


def _logit(p: float) -> float:
    import math

    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


# ======================================================================================================== Reward
@app.cls(image=dimage, gpu=["L40S", "A100-80GB", "H100"], volumes={"/vol": volume, "/sd": sd_volume}, secrets=[hf],
         timeout=60 * 60 * 10, scaledown_window=300, max_containers=8)  # fmt: skip
class Reward:
    detector: str = modal.parameter(default=DET_RUN)  # a run folder holding config.json and adapter/

    @modal.enter()
    def load(self) -> None:
        import sys

        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        sys.path.insert(0, "/root/sd")
        import train as tr

        torch.set_grad_enabled(False)
        t0 = time.time()
        volume.reload()
        self.torch, self.tr = torch, tr
        cfg = json.loads((Path(self.detector) / "config.json").read_text())
        self.tok, self.det = tr.load(cfg["base"], "cuda", num_labels=4)
        state = load_file(str(Path(self.detector) / "adapter" / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.det, state)
        self.det.eval()
        self.ntok = AutoTokenizer.from_pretrained(NLI)
        self.nli = AutoModelForSequenceClassification.from_pretrained(NLI, dtype=torch.float16).cuda().eval()
        labels = {v.lower(): int(k) for k, v in self.nli.config.id2label.items()}
        self.entail = next(i for name, i in labels.items() if name.startswith("entail"))
        self.etok = AutoTokenizer.from_pretrained(EMB)
        self.emb = AutoModel.from_pretrained(EMB, dtype=torch.float16).cuda().eval()
        print(f"Reward loaded ({self.detector}; NLI labels {labels}) in {time.time() - t0:.0f} s", flush=True)

    # -------------------------------------------------------------------------------------------- the 9B detector
    def _detect(self, texts: list[str]) -> list[dict]:
        """Per text: s9 (the detector's score: expected bucket / 3, 1 = generated) and h = 1 - s9, computed from
        the bucket probabilities so it keeps precision when s9 is within 1e-6 of 1."""
        torch, tr = self.torch, self.tr
        rows = tr.encode([{"text": t or "."} for t in texts], self.tok)
        out: list[dict] = [{} for _ in texts]
        for idx in tr.batches(rows, 16384, shuffle=False):
            logits = self.det(**tr.collate(rows, idx, self.tok, "cuda")).logits.double()
            p = torch.softmax(logits, -1)  # in float64: the human buckets' mass is often below float32's reach
            h = (3 * p[:, 0] + 2 * p[:, 1] + p[:, 2]) / 3
            for j, i in enumerate(idx):
                out[i] = {"s9": float(1 - h[j]), "h": float(h[j]), "logits": [round(float(x), 4) for x in logits[j]]}
        return out

    @modal.method()
    def detect(self, texts: list[str]) -> list[dict]:
        return self._detect(texts)

    # -------------------------------------------------------------------------------------------- content checks
    def _windows(self, text: str, k: int = 2) -> list[str]:
        """The rewrite as sliding windows of k sentences (stride 1). Chosen on dev (results/checks/windows.json): with
        whole 380-token windows DeBERTa called 53% of an off-topic passage's claims entailed (a marathon text
        "entailed" coffee claims); two-sentence windows keep the source at 0.96 and the off-topic at 0.22."""
        import re

        sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()] or ["."]
        return [" ".join(sents[i : i + k]) for i in range(max(1, len(sents) - k + 1))]

    def _entail(self, pairs: list[tuple[str, str]], bs: int = 128) -> list[float]:
        torch = self.torch
        out = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            enc = self.ntok([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first", max_length=512,
                            padding=True, return_tensors="pt").to("cuda")  # fmt: skip
            out += torch.softmax(self.nli(**enc).logits.float(), -1)[:, self.entail].tolist()
        return out

    def _embed(self, texts: list[str]):
        torch = self.torch
        enc = self.etok([t or "." for t in texts], truncation=True, max_length=512, padding=True, return_tensors="pt")
        h = self.emb(**enc.to("cuda")).last_hidden_state[:, 0].float()
        return torch.nn.functional.normalize(h, dim=-1)

    def _content(self, items: list[dict]) -> list[dict]:
        """items: {source, rewrite, claims}. claim_p: per claim, the best entailment probability over the rewrite's
        windows; claim_soft = mean claim_p; claim_strict = share of claims with claim_p > 0.5; fact = share of the
        source's hard facts (numbers, capitalised names not starting a sentence) still present; sim = bge cosine."""

        pairs, owner = [], []
        for k, it in enumerate(items):
            for w in self._windows(it["rewrite"] or "."):
                for c, claim in enumerate(it["claims"]):
                    pairs.append((w, claim))
                    owner.append((k, c))
        best: dict = {}
        for (k, c), p in zip(owner, self._entail(pairs), strict=True):
            best[(k, c)] = max(best.get((k, c), 0.0), p)
        sims = []
        for i in range(0, len(items), 32):
            chunk = items[i : i + 32]
            a, b = self._embed([x["source"] for x in chunk]), self._embed([x["rewrite"] for x in chunk])
            sims += (a * b).sum(-1).tolist()
        out = []
        for k, it in enumerate(items):
            cp = [best.get((k, c), 0.0) for c in range(len(it["claims"]))]
            out.append({"claim_p": [round(x, 4) for x in cp], "claim_soft": sum(cp) / max(len(cp), 1),
                        "claim_strict": sum(x > 0.5 for x in cp) / max(len(cp), 1), **_facts(it["source"],
                        it["rewrite"]), "sim": float(sims[k])})  # fmt: skip
        return out

    @modal.method()
    def content(self, items: list[dict]) -> list[dict]:
        return self._content(items)

    @modal.method()
    def score(self, items: list[dict]) -> list[dict]:
        """Both: the detector on each rewrite and the content checks against its source."""
        det = self._detect([it["rewrite"] for it in items])
        con = self._content(items)
        return [{**d, **c} for d, c in zip(det, con, strict=True)]


def _facts(source: str, rewrite: str) -> dict:
    """Hard facts: numbers, and capitalised words that do not start a sentence (names, places, brands)."""
    import re

    facts = set(re.findall(r"\b\d[\d,.%]*\d\b|\b\d\b", source))
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", source):
        before = source[: m.start()].rstrip()
        if before and before[-1] not in ".!?:\n\"'*#-":
            facts.add(m.group())
    kept = sum(f in rewrite for f in facts)
    return {"fact": kept / len(facts) if facts else 1.0, "n_facts": len(facts)}


def reward_of(s: dict, terms: dict, looped: bool, empty: bool) -> float:
    """The training reward from one rewrite's scores. det: the detector's log-odds that the text is human,
    logit(h), clipped to [-10, 5] and mapped to [0, 1] (the detector sits at s9 > 0.9999 on most AI text, so h itself
    has almost no spread). content: claim_soft x fact. loop: minus 0.5 for a looped rewrite."""
    if empty:
        return 0.0
    det = (min(max(_logit(s["h"]), -10.0), 5.0) + 10.0) / 15.0
    r = det * (s["claim_soft"] * s["fact"] if terms.get("content") else 1.0)
    if terms.get("loop") and looped:
        r -= 0.5
    return r


# ======================================================================================================== Trainer
def _looped(text: str) -> bool:
    import re
    from collections import Counter

    toks = re.findall(r"[A-Za-z']+", text.lower())
    grams = Counter(tuple(toks[i : i + 4]) for i in range(len(toks) - 3))
    return bool(grams) and max(grams.values()) > 3


def _chat(tok, user: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)  # fmt: skip


def _right_pad(seqs, pad_id):
    import torch

    width = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    lmask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for row, (p, c) in enumerate(seqs):
        ids[row, : len(p) + len(c)] = torch.tensor(list(p) + list(c))
        attn[row, : len(p) + len(c)] = 1
        lmask[row, len(p) : len(p) + len(c)] = True
    return ids.cuda(), attn.cuda(), lmask.cuda()


def _token_logprobs(model, ids, attn):
    import torch

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    picked = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).float()
    return picked - torch.logsumexp(logits.float(), dim=-1)


def _drift_kl(model, rows, pad_id, micro=4) -> float:
    """Mean per-token KL(untouched || trained) over the reply tokens of fixed untouched rewrites of the dev sources."""
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(rows), micro):
            ids, attn, lmask = _right_pad([(r["prompt_ids"], r["reply_ids"]) for r in rows[s : s + micro]], pad_id)
            sel = lmask[:, 1:]
            trained = model(input_ids=ids, attention_mask=attn).logits[:, :-1][sel].float().log_softmax(-1)
            with model.disable_adapter():
                base = model(input_ids=ids, attention_mask=attn).logits[:, :-1][sel].float().log_softmax(-1)
            total += float((base.exp() * (base - trained)).sum(-1).sum())
            count += int(sel.sum())
    return total / max(count, 1)


def _make_llm(gpu_util: float, max_len: int = 2048):
    from vllm import LLM

    return LLM(model=POLICY, gpu_memory_utilization=gpu_util, max_model_len=max_len, dtype="bfloat16",
               enable_prefix_caching=True, enable_lora=True, max_lora_rank=16, max_loras=2, seed=0)  # fmt: skip


@app.function(image=vimage, gpu=["H100", "A100-80GB"], cpu=4.0, memory=65536, timeout=10 * 60 * 60,
              volumes={"/vol": volume}, secrets=[hf], max_containers=6)  # fmt: skip
def train_run(cfg: dict) -> dict:
    """One GRPO run. cfg: name, seed, steps, prompts_per_step, group, lr, max_tokens, save_every, threshold, prompt,
    terms ({content, loop, shuffle}), train (sources: sid, text, claims), dev (sources for the drift readout)."""
    import math
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    name, terms, g = cfg["name"], cfg["terms"], cfg["group"]
    run_dir = Path("/vol/runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k not in ("train", "dev")}))
    rng, shuf_rng = random.Random(cfg["seed"]), random.Random(10_000 + cfg["seed"])
    torch.manual_seed(cfg["seed"])
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(POLICY)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    llm = _make_llm(cfg.get("gpu_util", 0.35))
    model = AutoModelForCausalLM.from_pretrained(POLICY, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # adapter and Adam state in float32; the frozen base stays bfloat16
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    reward = modal.Cls.from_name("detector-evasion", "Reward")()
    prompt_of = {s["sid"]: cfg["prompt"].format(source=s["text"]) for s in cfg["train"] + cfg["dev"]}

    # fixed untouched rewrites of the dev sources, for the drift readout
    sp0 = SamplingParams(n=1, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                         max_tokens=cfg["max_tokens"], seed=0)  # fmt: skip
    outs = llm.generate([_chat(tok, prompt_of[s["sid"]]) for s in cfg["dev"]], sp0, use_tqdm=False)
    neutral = [{"prompt_ids": list(o.prompt_token_ids), "reply_ids": list(o.outputs[0].token_ids)} for o in outs]

    def save(tag: str) -> Path:
        d = run_dir / tag
        model.save_pretrained(str(d))
        return d

    curve = [{"step": 0, "drift_kl": 0.0, "checkpoint": True}]
    save("step0")
    volume.commit()
    lora = LoRARequest(f"{name}-0", 1, str(run_dir / "step0"))
    sp = SamplingParams(n=g, temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=cfg["max_tokens"])
    roll = (run_dir / "rollouts.jsonl").open("w")
    for step in range(1, cfg["steps"] + 1):
        t0 = time.time()
        batch = [cfg["train"][rng.randrange(len(cfg["train"]))] for _ in range(cfg["prompts_per_step"])]
        outs = llm.generate([_chat(tok, prompt_of[b["sid"]]) for b in batch], sp, lora_request=lora, use_tqdm=False)
        t_gen = time.time() - t0
        meta = [(b, list(o.prompt_token_ids), list(c.token_ids), c.text, c.finish_reason)
                for b, o in zip(batch, outs, strict=True) for c in o.outputs]  # fmt: skip
        scores = reward.score.remote([{"source": b["text"], "rewrite": text, "claims": b["claims"]}
                                      for b, _, _, text, _ in meta])  # fmt: skip
        t_rew = time.time() - t0 - t_gen
        loops = [_looped(m[3]) for m in meta]
        rewards = [reward_of(s, terms, lp, not m[3].strip()) for s, lp, m in zip(scores, loops, meta, strict=True)]
        train_r = list(rewards)
        if terms.get("shuffle"):  # break the score-content link inside each group, keep the scores themselves
            for s0 in range(0, len(train_r), g):
                block = train_r[s0 : s0 + g]
                shuf_rng.shuffle(block)
                train_r[s0 : s0 + g] = block
        seqs, weights = [], []
        for s0 in range(0, len(meta), g):
            block = train_r[s0 : s0 + g]
            mean = sum(block) / g
            sd = (sum((x - mean) ** 2 for x in block) / g) ** 0.5
            for i in range(s0, s0 + g):
                a = (train_r[i] - mean) / max(sd, cfg.get("sd_floor", 0.02))
                if abs(a) > 1e-6:
                    seqs.append((meta[i][1], meta[i][2]))
                    weights.append(a)
        n_tok = sum(len(m[2]) for m in meta)
        for i, (b, _, ids_, text, fin) in enumerate(meta):
            s = scores[i]
            roll.write(json.dumps({"step": step, "sid": b["sid"], "reward": round(rewards[i], 5), "s9": s["s9"],
                                   "claim": round(s["claim_soft"], 4), "fact": round(s["fact"], 4),
                                   "sim": round(s["sim"], 4), "loop": loops[i], "finish": fin, "tokens": len(ids_),
                                   "text": text}) + "\n")  # fmt: skip
        t_up = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_val, gnorm = 0.0, 0.0
        if seqs:
            order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
            mb = cfg.get("micro_batch", 4)
            for s0 in range(0, len(order), mb):
                idx = order[s0 : s0 + mb]
                ids, attn, lmask = _right_pad([seqs[i] for i in idx], pad_id)
                w = torch.tensor([weights[i] for i in idx], device="cuda").unsqueeze(1)
                logp = _token_logprobs(model, ids, attn)
                per_tok = torch.where(lmask[:, 1:], -w * logp, torch.zeros_like(logp))  # where, not multiply
                loss = per_tok.sum() / max(n_tok, 1)
                loss.backward()
                loss_val += float(loss.detach())
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if not math.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                print(f"{name} step {step}: non-finite grad norm, update skipped", flush=True)
            else:
                opt.step()
        n = len(meta)
        rec = {"step": step, "reward": sum(rewards) / n, "s9": sum(s["s9"] for s in scores) / n,
               "evade": sum(s["s9"] < cfg["threshold"] for s in scores) / n,
               "claim": sum(s["claim_soft"] for s in scores) / n, "fact": sum(s["fact"] for s in scores) / n,
               "sim": sum(s["sim"] for s in scores) / n, "loop": sum(loops) / n,
               "trunc": sum(m[4] == "length" for m in meta) / n, "mean_tokens": n_tok / n, "loss": loss_val,
               "grad_norm": gnorm, "n_update": len(seqs), "t_gen": round(t_gen, 1), "t_reward": round(t_rew, 1),
               "t_update": round(time.time() - t_up, 1)}  # fmt: skip
        last = step == cfg["steps"] or (cfg.get("max_hours") and time.time() - t_start > cfg["max_hours"] * 3600)
        if step % cfg["save_every"] == 0 or last:
            adapter = save(f"step{step}")
            rec.update({"drift_kl": _drift_kl(model, neutral, pad_id), "checkpoint": True})
        else:
            adapter = save("cur")
        lora = LoRARequest(f"{name}-{step}", step + 1, str(adapter))
        rec["t_step"] = round(time.time() - t0, 1)
        curve.append(rec)
        print(name, json.dumps(rec), flush=True)
        (run_dir / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in curve) + "\n")
        if step % 5 == 0 or last:
            roll.flush()
            volume.commit()
        if last:
            break
    roll.close()
    volume.commit()
    return {"name": name, "curve": curve, "wall_s": round(time.time() - t_start)}


# ======================================================================================================== Sampling
@app.function(image=vimage, gpu=["H100", "A100-80GB"], timeout=2 * 60 * 60, volumes={"/vol": volume}, secrets=[hf],
              max_containers=6)  # fmt: skip
def rewrite(adapter: str, prompts: list[str], n: int = 1, temperature: float = 0.7, top_p: float = 0.8,
            top_k: int = 20, max_tokens: int = 640, seed: int = 0) -> list[list[dict]]:  # fmt: skip
    """n completions per user prompt (thinking off) from the untouched model (adapter "") or /vol/runs/<adapter>."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    volume.reload()
    llm = _make_llm(0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=1.0,
                        max_tokens=max_tokens, seed=seed)  # fmt: skip
    lora = LoRARequest(adapter, 1, f"/vol/runs/{adapter}") if adapter else None
    outs = llm.generate([_chat(tok, p) for p in prompts], sp, lora_request=lora, use_tqdm=False)
    return [[{"text": c.text, "finish": c.finish_reason, "tokens": len(c.token_ids)} for c in o.outputs] for o in outs]


# ======================================================================================================== Judge
@app.cls(image=vimage, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 3, scaledown_window=120)
class Judge:
    @modal.enter()
    def load(self) -> None:
        from vllm import LLM

        self.llm = LLM(JUDGE, max_model_len=6144, gpu_memory_utilization=0.9, dtype="bfloat16")
        self.tok = self.llm.get_tokenizer()

    def _texts(self, prompts):
        return [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in prompts]  # fmt: skip

    def _first(self, prompts):
        from vllm import SamplingParams

        outs = self.llm.generate(self._texts(prompts), SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
        return [o.outputs[0].logprobs[0] for o in outs]

    @modal.method()
    def ask(self, prompts: list[str]) -> list[dict]:
        """Each prompt asks for one letter, A or B; P(A), P(B) from the first token's top logprobs."""
        import math

        res = []
        for lp in self._first(prompts):
            pa = sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip() == "A")
            pb = sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip() == "B")
            res.append({"pA": pa, "pB": pb})
        return res

    @modal.method()
    def rate(self, prompts: list[str]) -> list[float]:
        """Each prompt asks for one digit 1-5; the expected digit under the first token's top logprobs."""
        import math

        res = []
        for lp in self._first(prompts):
            mass = {d: sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip() == str(d))
                    for d in range(1, 6)}  # fmt: skip
            z = sum(mass.values())
            res.append(sum(d * m for d, m in mass.items()) / z if z > 0 else float("nan"))
        return res

    @modal.method()
    def generate(self, prompts: list[str], max_tokens: int = 400) -> list[str]:
        from vllm import SamplingParams

        outs = self.llm.generate(self._texts(prompts), SamplingParams(max_tokens=max_tokens, temperature=0.0))
        return [o.outputs[0].text for o in outs]


# ======================================================================================================== Reader
@app.cls(image=vimage, gpu=["H100", "A100-80GB"], volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 2,
         scaledown_window=120)  # fmt: skip
class Reader:
    """phi-4 under vLLM: each prompt ends asking for one capital letter; returns the probability of each allowed letter
    (from the first token's top-20 logprobs, renormalised over the allowed letters)."""

    @modal.enter()
    def load(self) -> None:
        from vllm import LLM

        self.llm = LLM(READER, max_model_len=8192, gpu_memory_utilization=0.9, dtype="bfloat16")
        self.tok = self.llm.get_tokenizer()

    @modal.method()
    def choose(self, prompts: list[str], letters: str) -> list[dict]:
        import math

        from vllm import SamplingParams

        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]  # fmt: skip
        outs = self.llm.generate(texts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            mass = {x: sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip() == x)
                    for x in letters}  # fmt: skip
            z = sum(mass.values()) or 1.0
            res.append({x: m / z for x, m in mass.items()})
        return res


# ======================================================================================================== Defence
@app.function(image=dimage, gpu="A100-80GB", volumes={"/vol": volume, "/sd": sd_volume}, secrets=[hf],
              timeout=6 * 60 * 60)  # fmt: skip
def defend(name: str, rows: list[dict], replay: int, lr: float = 5e-5, seed: int = 0) -> dict:
    """Continue training f9-9b's LoRA and head for one pass (A100: the linear-attention backward refuses Hopper with
    this Triton). rows: {text, bucket} new rows (rewrites, bucket 3 = generated); replay: how many of the 9B's own
    training rows (/sd/data/f9/train.jsonl, a seed-fixed sample) to mix in so it keeps what it knew.
    Saves /vol/defend/<name>/{adapter, config.json}."""
    import random
    import shutil
    import sys

    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sys.path.insert(0, "/root/sd")
    import train as tr

    t0 = time.time()
    torch.manual_seed(seed)
    cfg = json.loads((Path(DET_RUN) / "config.json").read_text())
    tok, model = tr.load(cfg["base"], "cuda", num_labels=4)
    set_peft_model_state_dict(model, load_file(str(Path(DET_RUN) / "adapter" / "adapter_model.safetensors")))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    with open("/sd/data/f9/train.jsonl") as fh:
        pool = [json.loads(x) for x in fh]
    rnd = random.Random(seed)
    data = [{"text": r["text"], "bucket": int(r["bucket"])} for r in rnd.sample(pool, replay)]
    data += [{"text": r["text"], "bucket": int(r["bucket"])} for r in rows]
    data = tr.encode(data, tok)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    groups, cur = [], []
    for m in tr.batches(data, 8192, shuffle=True, seed=seed, max_rows=32):
        cur.append(m)
        if sum(len(x) for x in cur) >= 32:
            groups.append(cur)
            cur = []
    groups += [cur] if cur else []
    log = []
    model.train()
    for step, grp in enumerate(groups):
        n = sum(len(x) for x in grp)
        loss_sum = 0.0
        for idx in grp:
            logits = model(**tr.collate(data, idx, tok, "cuda")).logits.float()
            yb = torch.tensor([data[i]["bucket"] for i in idx], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits, yb, reduction="sum") / n
            loss.backward()
            loss_sum += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, (step + 1) / 10)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            log.append({"step": step, "of": len(groups), "loss": round(loss_sum, 4), "s": round(time.time() - t0)})
            print(log[-1], flush=True)
    out = Path("/vol/defend") / name
    if out.exists():
        shutil.rmtree(out)
    model.save_pretrained(str(out / "adapter"))
    (out / "config.json").write_text(json.dumps({**cfg, "defended_from": DET_RUN, "rows": len(rows),
                                                 "replay": replay, "lr": lr}))  # fmt: skip
    volume.commit()
    return {"name": name, "steps": len(groups), "log": log, "seconds": round(time.time() - t0)}
