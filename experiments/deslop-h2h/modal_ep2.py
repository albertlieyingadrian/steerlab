"""The GPU side of De-slopping, Episode 2, on Modal (app `deslop-ep2`).

    MODAL_PROFILE=albertlieyingadrian modal deploy experiments/deslop-h2h/modal_ep2.py

- `Runner` (Qwen3-8B, thinking off; optional LoRA adapter): generation with an optional system prompt, the deslop
  toolkit's backtracking sampler, and one residual intervention at a layer (`add`: a fixed vector; `sae`: subtract
  c x the decoded part of listed Qwen-Scope SAE features). Also KL drift on human passages, the quiz, and steerlab's
  difference-of-means direction. The hooks, sampler and direction builder are the packages' own code.
- `train_sft`, `train_dpo`: LoRA training (deslop's recipe; DPO with an NLL anchor on the chosen reply).
- `Judge`: gemma-3-27b-it under vLLM, a blind pairwise judge read from the logprobs of "A" and "B".
Sampling uses temperature 0.7, top-p 0.8, top-k 20 and repetition_penalty 1.0 everywhere.
"""

import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parents[1] / "packages" if modal.is_local() else Path("/root")
BASE = "Qwen/Qwen3-8B"
SAE_REPO, SAE_FILE, SAE_K = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", "layer18.sae.pt", 50
JUDGE = "google/gemma-3-27b-it"
EOS_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>

app = modal.App("deslop-ep2")
volume = modal.Volume.from_name("deslop-ep2", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.20.2",
        "accelerate>=1.10",
        "peft==0.21.2",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "scipy",
        "scikit-learn",
        "pydantic>=2.7,<3",
        "typer",
    )
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)
image = (
    base_image.add_local_dir(PACKAGES / "deslop" / "src" / "deslop", "/root/deslop")
    .add_local_dir(PACKAGES / "steerlab" / "src" / "steerlab", "/root/steerlab")
    .add_local_dir(PACKAGES / "sandbagkit" / "src" / "sandbagkit", "/root/sandbagkit")
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
)


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def _render(tok, messages) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)


@contextmanager
def _using(ctx, model):
    with ctx:
        yield model


@app.cls(image=image, gpu=["H100", "A100-80GB"], volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 3,
         scaledown_window=120, max_containers=20)  # fmt: skip
class Runner:
    adapter: str = modal.parameter(default="")

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        torch.set_grad_enabled(False)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
        if self.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, f"/vol/adapters/{self.adapter}")
        self.model.eval()
        self._sae = None
        print(f"loaded {BASE} {self.adapter} in {time.time() - t0:.0f} s")

    # ------------------------------------------------------------------ interventions
    def _layers(self):
        from deslop.steer import decoder_layers

        return decoder_layers(self.model)

    def _sae_parts(self):
        if self._sae is None:
            from huggingface_hub import hf_hub_download

            st = self.torch.load(hf_hub_download(SAE_REPO, SAE_FILE), map_location="cuda", weights_only=True)
            self._sae = {k: st[k].float().cuda() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
        return self._sae

    def _hooks(self, job: dict) -> ExitStack:
        """One residual intervention at the output of decoder layer `layer`, inside the block."""
        torch = self.torch
        stack = ExitStack()
        iv = job.get("intervention")
        if not iv:
            return stack
        layer = self._layers()[int(iv["layer"])]
        if iv["kind"] == "add":
            v = torch.tensor(iv["vector"], device="cuda", dtype=torch.float32)
            delta = (float(iv["norm"]) * v / v.norm()).to(torch.bfloat16)

            def fn(h):
                return h + delta.to(h.dtype)
        elif iv["kind"] == "sae":
            S = self._sae_parts()
            fi = torch.tensor(iv["features"], device="cuda", dtype=torch.long)
            Wd, c = S["W_dec"][:, fi], float(iv["coef"])

            def fn(h):
                hf = h.float()
                flat = hf.reshape(-1, hf.shape[-1])
                pre = flat @ S["W_enc"].T + S["b_enc"]
                val, idx = pre.topk(SAE_K, dim=-1)
                acts = pre.new_zeros(pre.shape).scatter_(-1, idx, val)[:, fi]
                return (hf - c * (acts @ Wd.T).reshape(hf.shape)).to(h.dtype)
        else:
            raise ValueError(iv["kind"])

        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (fn(out[0]), *out[1:])
            return fn(out)

        handle = layer.register_forward_hook(hook)
        stack.callback(handle.remove)
        return stack

    # ------------------------------------------------------------------ generation
    def _generate(self, job: dict) -> list[dict]:
        torch = self.torch
        rows, n = job["rows"], int(job.get("n", 1))
        system, temp = job.get("system"), float(job.get("temperature", 0.7))
        max_new, seed = int(job.get("max_new_tokens", 512)), int(job.get("seed", 0))
        out: list[dict] = []
        t0 = time.time()
        with self._hooks(job):
            if job.get("sampler"):
                from deslop.sampler import antislop_generate

                for row in rows:
                    ids = self.tok(_render(self.tok, _messages(row["prompt"], system)), add_special_tokens=False)
                    for k in range(n):
                        res = antislop_generate(self.model, self.tok, prompt_ids=ids["input_ids"],
                                                max_new_tokens=max_new, temperature=temp, seed=seed + k,
                                                max_backtracks=int(job.get("max_backtracks", 300)))  # fmt: skip
                        out.append({"id": row["id"], "sample": k, "text": res.text.strip(),
                                    "n_tokens": len(res.token_ids), "finished": res.finished,
                                    "backtracks": res.backtracks, "accepted": len(res.accepted)})  # fmt: skip
            else:
                todo = [(row, k) for k in range(n) for row in rows]
                batch = int(job.get("batch", 48))
                for b in range(0, len(todo), batch):
                    chunk = todo[b : b + batch]
                    texts = [_render(self.tok, _messages(r["prompt"], system)) for r, _ in chunk]
                    enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                    torch.manual_seed(seed + 1000 * chunk[0][1] + b)
                    kw = {"do_sample": True, "temperature": temp, "top_p": 0.8, "top_k": 20} if temp > 0 else \
                        {"do_sample": False}  # fmt: skip
                    gen = self.model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.0,
                                              pad_token_id=self.tok.pad_token_id, **kw)  # fmt: skip
                    for (row, k), ids in zip(chunk, gen[:, enc["input_ids"].shape[1] :], strict=True):
                        ids = ids.tolist()
                        ends = [i for i, t in enumerate(ids) if t in EOS_IDS]
                        ids = ids[: ends[0]] if ends else ids
                        text = self.tok.decode(ids, skip_special_tokens=True).strip()
                        out.append({"id": row["id"], "sample": k, "text": text, "n_tokens": len(ids),
                                    "finished": bool(ends)})  # fmt: skip
        print(f"generate: {len(out)} replies in {time.time() - t0:.0f} s")
        return out

    @modal.method()
    def generate(self, job: dict) -> list[dict]:
        """rows [{id, prompt}], system, n, temperature, max_new_tokens, seed, sampler, intervention, batch."""
        return self._generate(job)

    @modal.method()
    def quiz(self, job: dict) -> list[dict]:
        rows = [{"id": f"q{i:02d}", "prompt": q + " Answer in one sentence."} for i, q in enumerate(job["questions"])]
        return self._generate({**job, "rows": rows, "n": 1, "temperature": 0.0, "max_new_tokens": 48})

    @modal.method()
    def kl(self, job: dict) -> list[dict]:
        """Mean per-token KL(this setup || untouched model) on human passages given as the reply to a fixed request."""
        F = self.torch.nn.functional
        request = "Write a short passage of prose."
        out = []
        for p in job["passages"]:
            base_ids, base_start = self._encode_reply(request, p["text"], None)
            ids, start = self._encode_reply(request, p["text"], job.get("system"))
            with self._base_context() as base_model:
                base_logits = base_model(base_ids).logits[0, base_start - 1 : -1].float()
            with self._hooks(job):
                logits = self.model(ids).logits[0, start - 1 : -1].float()
            logp, logq = F.log_softmax(logits, -1), F.log_softmax(base_logits, -1)
            kl = (logp.exp() * (logp - logq)).sum(-1)
            out.append({"id": p["id"], "kl": float(kl.mean()), "n_tokens": int(logits.shape[0])})
        return out

    def _encode_reply(self, request: str, reply: str, system: str | None):
        prefix = _render(self.tok, _messages(request, system))
        start = len(self.tok(prefix, add_special_tokens=False)["input_ids"])
        ids = self.tok(prefix + reply, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids[:, : start + 400].to("cuda"), start

    def _base_context(self):
        if self.adapter:
            return _using(self.model.disable_adapter(), self.model)
        return _using(nullcontext(), self.model)

    @modal.method()
    def direction(self, job: dict) -> dict:
        """steerlab's difference-of-means direction at each layer: pairs [{prompt, positive, negative}], layers."""
        from steerlab.data import Concept, ContrastPair, Example
        from steerlab.directions import build_direction

        pairs = tuple(ContrastPair(Example(p["prompt"], p["positive"]), Example(p["prompt"], p["negative"]))
                      for p in job["pairs"])  # fmt: skip
        concept = Concept(name="register", label="assistant register", description=job.get("description", ""),
                          pairs=pairs)  # fmt: skip
        out = {}
        for layer in job["layers"]:
            vec = build_direction(self.model, self.tok, concept, layer=int(layer), model_name=BASE,
                                  template_kwargs={"enable_thinking": False}, batch_size=8,
                                  max_length=768)  # fmt: skip
            meta = {k: vec.metadata[k] for k in ("class_gap", "cosine_agreement", "d_prime", "n_pairs")}
            out[int(layer)] = {"vector": [float(x) for x in vec.vector], **meta}
            print(f"layer {layer}: {meta}")
        return out

    @modal.method()
    def sae_direction(self, features: list[int]) -> list[float]:
        """The sum of the listed features' unit decoder directions (probes-vs-interp's `sae_dir`, before sign)."""
        Wd = self._sae_parts()["W_dec"][:, self.torch.tensor(features, device="cuda")]
        return (Wd / Wd.norm(dim=0, keepdim=True)).sum(1).cpu().tolist()

    @modal.method()
    def residual_norm(self, texts: list[str], layer: int) -> float:
        """Mean L2 norm of the layer's output over the reply positions of the given texts (in chat format)."""
        norms = []
        for t in texts:
            ids, start = self._encode_reply("Write a short passage of prose.", t, None)
            hs = self.model(ids, output_hidden_states=True).hidden_states[layer + 1][0, start:]
            norms.append(hs.float().norm(dim=-1).mean().item())
        return float(sum(norms) / len(norms))


def _lora(model, rank: int, alpha: int):
    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg), targets


def _tokenize(tok, prompt: str, reply: str) -> tuple[list[int], list[int]]:
    p_ids = tok(_render(tok, _messages(prompt, None)), add_special_tokens=False)["input_ids"]
    r_ids = tok(reply + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return p_ids + r_ids, [-100] * len(p_ids) + r_ids


@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 3)
def train_sft(examples: list[dict], name: str, epochs: int = 2, lr: float = 2e-4, rank: int = 16, alpha: int = 32,
              batch: int = 8, seed: int = 0) -> dict:  # fmt: skip
    """deslop's LoRA recipe on Qwen3-8B: every attention and MLP projection, loss on the reply only, bare prompt."""
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model, targets = _lora(model, rank, alpha)
    rows = [_tokenize(tok, ex["prompt"], ex["reply"]) for ex in examples]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * ((len(rows) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    rng = random.Random(seed)
    log, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            chunk = [rows[i] for i in order[b : b + batch]]
            width = max(len(x) for x, _ in chunk)  # right padding (the left-padding NaN lesson)
            ids = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x, _ in chunk], device="cuda")
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in chunk], device="cuda")
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in chunk], device="cuda")
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch, "loss": float(loss)})
            if step % 10 == 0:
                print(f"step {step}/{steps} loss {float(loss):.3f} ({time.time() - t0:.0f} s)")
    model.save_pretrained(f"/vol/adapters/{name}")
    volume.commit()
    cfg = {"epochs": epochs, "lr": lr, "rank": rank, "alpha": alpha, "batch": batch, "dropout": 0.05,
           "targets": targets}  # fmt: skip
    return {"name": name, "steps": steps, "examples": len(rows), "seconds": round(time.time() - t0), "log": log,
            "config": cfg}  # fmt: skip


@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 4)
def train_dpo(pairs: list[dict], name: str, epochs: int = 2, lr: float = 5e-5, beta: float = 0.1, nll: float = 1.0,
              rank: int = 16, alpha: int = 32, batch: int = 4, seed: int = 0) -> dict:  # fmt: skip
    """DPO with an NLL anchor on the chosen reply (RPO-style), LoRA; the reference is the adapter switched off.

    pairs [{prompt, chosen, rejected}]. loss = -log sigmoid(beta * (dlogp_chosen - dlogp_rejected)) + nll * NLL(chosen),
    where dlogp = log p_policy - log p_reference summed over reply tokens and NLL is per reply token.
    """
    import random

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model, targets = _lora(model, rank, alpha)

    def pad(seqs):
        width = max(len(x) for x, _ in seqs)
        ids = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x, _ in seqs], device="cuda")
        labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in seqs], device="cuda")
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in seqs], device="cuda")
        return ids, labels, mask

    def seq_logp(m, ids, labels, mask):
        logits = m(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        tgt = labels[:, 1:]
        keep = tgt != -100
        lp = torch.gather(F.log_softmax(logits, -1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lp = torch.where(keep, lp, torch.zeros_like(lp))
        return lp.sum(-1), keep.sum(-1)

    rows = [(_tokenize(tok, p["prompt"], p["chosen"]), _tokenize(tok, p["prompt"], p["rejected"])) for p in pairs]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * ((len(rows) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    rng = random.Random(seed)
    log, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            chunk = [rows[i] for i in order[b : b + batch]]
            ids, labels, mask = pad([c for c, _ in chunk] + [r for _, r in chunk])
            with torch.no_grad(), model.disable_adapter():
                ref, _ = seq_logp(model, ids, labels, mask)
            pol, ntok = seq_logp(model, ids, labels, mask)
            k = len(chunk)
            margin = beta * ((pol[:k] - ref[:k]) - (pol[k:] - ref[k:]))
            dpo = -F.logsigmoid(margin).mean()
            anchor = (-pol[:k] / ntok[:k].clamp(min=1)).mean()
            loss = dpo + nll * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch, "loss": float(loss), "dpo": float(dpo), "nll": float(anchor),
                        "acc": float((margin > 0).float().mean()), "margin": float(margin.mean())})  # fmt: skip
            if step % 10 == 0:
                print(f"step {step}/{steps} dpo {float(dpo):.3f} nll {float(anchor):.3f} ({time.time() - t0:.0f} s)")
    model.save_pretrained(f"/vol/adapters/{name}")
    volume.commit()
    cfg = {"epochs": epochs, "lr": lr, "beta": beta, "nll": nll, "rank": rank, "alpha": alpha, "batch": batch,
           "targets": targets}  # fmt: skip
    return {"name": name, "steps": steps, "pairs": len(rows), "seconds": round(time.time() - t0), "log": log,
            "config": cfg}  # fmt: skip


@app.cls(image=vllm_image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 3,
         scaledown_window=120)  # fmt: skip
class Judge:
    @modal.enter()
    def load(self) -> None:
        from vllm import LLM

        self.llm = LLM(JUDGE, max_model_len=6144, gpu_memory_utilization=0.9, dtype="bfloat16")
        self.tok = self.llm.get_tokenizer()

    @modal.method()
    def ask(self, prompts: list[str]) -> list[dict]:
        """Each prompt ends asking for one letter, A or B; returns P(A), P(B) from the first token's top logprobs."""
        import math

        from vllm import SamplingParams

        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]  # fmt: skip
        outs = self.llm.generate(texts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            pa = sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip() == "A")
            pb = sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip() == "B")
            res.append({"pA": pa, "pB": pb, "top": o.outputs[0].text})
        return res
