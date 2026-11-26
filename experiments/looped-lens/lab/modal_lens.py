"""Offline lens lab for Inkling 2 Mini on a Modal H100 (the model is ~52 GB in bf16).

It does what an NDIF trace does, with plain forward hooks: for each question, read every one of
the 48 layer visits at the read position and apply several candidate lenses, keeping the top
token ids. It also fits tuned-lens translators on generic text.

    modal run dev/modal_lens.py::readouts --questions dev/data/all_private.jsonl
    modal run dev/modal_lens.py::train --name tuned_v1

Lenses (all read one visit, at one position; "in" = the state entering the visit, "out" = after it):
    act_in   norm + unembed of the active block only (the starter's logit lens)
    mix_in   the model's own final_input depth-mix over completed blocks + active block, then norm + unembed
    attn_in  the visit's own attention_input mix (what its attention reads), norm + unembed
    mlp_in   the MoE sublayer's input (post_attention_layernorm.input), norm + unembed
    msg      the MoE message from the previous pass (passes 2 and 3 only), norm + unembed
    act_out  the active block after the visit
    mix_out  final_input over the blocks and active block after the visit (at visit 47 this is the model's output)
"""

import json
import os

import modal

APP = "looped-lens-dev"
MODEL_ID = "thinkingmachines/Inkling-2-Mini"
TOPK = 20

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "transformers==5.24.0", "safetensors", "huggingface_hub[hf_transfer]",
                    "accelerate", "pillow", "datasets", "numpy>=2.0", "pyarrow")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
cache = modal.Volume.from_name("looped-lens-cache", create_if_missing=True)
app = modal.App(APP, image=image)


def translate(p, x):
    """x [n, d] float -> translated [n, d]."""
    if "W" in p:
        return x + x @ p["W"].to(x.device, x.dtype).T + p["b"].to(x.device, x.dtype)
    return x + (x @ p["V"].to(x.device, x.dtype).T) @ p["U"].to(x.device, x.dtype).T + p["b"].to(x.device, x.dtype)


@app.cls(gpu="H100", volumes={"/cache": cache}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=3600 * 2,
         scaledown_window=120, max_containers=4, memory=65536)
class Lab:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        cache.commit()
        self.lm = self.model.model.language_model
        cfg = self.lm.config
        self.start, self.end, self.passes = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.loop_num_passes
        self.nl = cfg.num_hidden_layers
        self.nvisits = self.nl + (self.end - self.start) * (self.passes - 1)
        self.block = cfg.n_layers_per_attn_res_block

    def _visit(self, layer, run):
        if layer < self.start:
            return layer
        if layer < self.end:
            return self.start + run * (self.end - self.start) + (layer - self.start)
        return layer + (self.end - self.start) * (self.passes - 1)

    def _capture(self, fn):
        """Register hooks; fn(visit, layer_idx, layer, args, out, mlp_in) is called after each visit."""
        handles, calls, pending = [], {}, {}
        for i, layer in enumerate(self.lm.layers):
            def pre(mod, args, i=i):
                pending[i] = {"args": args}

            def mlp_pre(mod, args, i=i):
                pending[i]["mlp_in"] = args[0]

            def post(mod, args, out, i=i):
                run = calls.get(i, 0)
                calls[i] = run + 1
                fn(self._visit(i, run), i, mod, pending[i]["args"], out, pending[i].get("mlp_in"))

            handles += [layer.register_forward_pre_hook(pre), layer.post_attention_layernorm.register_forward_pre_hook(mlp_pre),
                        layer.register_forward_hook(post)]
        return handles

    def _unembed(self, x):
        import torch

        return self.model.lm_head(self.lm.norm(x.to(torch.bfloat16)))

    def _mix(self, blocks, h):
        return self.lm.final_input(blocks, h)[0]

    def _reps(self, i, layer, args, out, sl):
        import torch

        h, blocks, o = args[0][:, sl], args[1][:, sl], out[:, sl]
        blocks_after = torch.cat([blocks, h.unsqueeze(-2)], dim=-2) if i % self.block == 0 else blocks
        return h, blocks, o, blocks_after

    @modal.method()
    def readouts(self, questions: list[dict]) -> list[dict]:
        import torch

        results = []
        for q in questions:
            ids = self.tok(q["model_input"], add_special_tokens=False)["input_ids"]
            p = q["read_position"]
            per = {}

            def fn(v, i, layer, args, out, mlp_in):
                sl = slice(p, p + 1)
                h, blocks, o, blocks_after = self._reps(i, layer, args, out, sl)
                reps = {"act_in": h, "mix_in": self._mix(blocks, h), "act_out": o, "mix_out": self._mix(blocks_after, o)}
                if layer.attention_input is not None and blocks.shape[-2] > 0:
                    reps["attn_in"] = layer.attention_input(blocks, h)[0]
                if mlp_in is not None:
                    reps["mlp_in"] = mlp_in[:, sl]
                if len(args) > 2 and args[2] is not None:
                    reps["msg"] = args[2][:, sl, 3]
                per[v] = {k: self._unembed(x).float().reshape(-1).topk(TOPK).indices.tolist() for k, x in reps.items()}

            handles = self._capture(fn)
            try:
                with torch.no_grad():
                    x = torch.tensor([ids], device="cuda")
                    logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, p].float()
            finally:
                for hd in handles:
                    hd.remove()
            results.append({"id": q["id"], "final": logits.topk(TOPK).indices.tolist(), "visits": per})
        return results

    @modal.method()
    def apply_tuned(self, questions: list[dict], weights_name: str, reps: list[str], label: str = "tuned") -> list[dict]:
        """Readouts of fitted translators (from /cache/tuned/<weights_name>.pt) on the given questions."""
        import torch

        cache.reload()
        W = torch.load(f"/cache/tuned/{weights_name}.pt")
        results = []
        for q in questions:
            ids = self.tok(q["model_input"], add_special_tokens=False)["input_ids"]
            p = q["read_position"]
            per = {}

            def fn(v, i, layer, args, out, mlp_in):
                h, blocks, o, blocks_after = self._reps(i, layer, args, out, slice(p, p + 1))
                all_reps = {"act_in": lambda: h, "mix_in": lambda: self._mix(blocks, h), "mix_out": lambda: self._mix(blocks_after, o),
                            "mlp_in": lambda: mlp_in[:, p : p + 1]}
                per[v] = {}
                for r in reps:
                    if f"{r}.{v}" in W:
                        y = translate(W[f"{r}.{v}"], all_reps[r]().float().reshape(1, -1))
                        per[v][f"{label}_{r}"] = self._unembed(y).float().reshape(-1).topk(TOPK).indices.tolist()

            handles = self._capture(fn)
            try:
                with torch.no_grad():
                    x = torch.tensor([ids], device="cuda")
                    self.model(input_ids=x, attention_mask=torch.ones_like(x))
            finally:
                for hd in handles:
                    hd.remove()
            results.append({"id": q["id"], "visits": per})
        return results

    @modal.method()
    def train(self, name: str, reps: list[str], steps: int = 400, batch: int = 8, seqlen: int = 128, lr: float = 1e-3,
              rank: int = 0, chat_frac: float = 0.3) -> dict:
        """Fit one translator per (representation, visit) so that unembed(norm(T(x))) matches the model's final
        next-token distribution at the same position (the tuned-lens recipe: KL to the final logits).
        rank=0: full affine (x + W x + b, W init 0); rank>0: x + U V x + b."""
        import random

        import torch
        import torch.nn.functional as F
        from datasets import load_dataset

        torch.manual_seed(0)
        rng = random.Random(0)
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        texts = [t for t in ds.select(range(200000))["text"] if len(t) > 400]
        chat = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{}<|end_message|><|message_model|>{}"

        def sample_batch():
            rows = []
            while len(rows) < batch:
                t = texts[rng.randrange(len(texts))].strip()
                if rng.random() < chat_frac:
                    s = t.split(". ")
                    k = rng.randrange(1, max(2, len(s)))
                    t = chat.format(". ".join(s[:k])[:300] + ".", ". ".join(s[k:]))
                ids = self.tok(t, add_special_tokens=False)["input_ids"][:seqlen]
                if len(ids) == seqlen:
                    rows.append(ids)
            return torch.tensor(rows, device="cuda")

        d = self.lm.config.hidden_size
        params = {}
        for r in reps:
            for v in range(self.nvisits):
                if rank:
                    params[f"{r}.{v}"] = {"U": torch.zeros(d, rank, device="cuda", requires_grad=True),
                                          "V": (torch.randn(rank, d, device="cuda") / d**0.5).requires_grad_(),
                                          "b": torch.zeros(d, device="cuda", requires_grad=True)}
                else:
                    params[f"{r}.{v}"] = {"W": torch.zeros(d, d, device="cuda", requires_grad=True),
                                          "b": torch.zeros(d, device="cuda", requires_grad=True)}
        opt = torch.optim.Adam([t for p in params.values() for t in p.values()], lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
        log = []
        for step in range(steps):
            x = sample_batch()
            caught = {}

            def fn(v, i, layer, args, out, mlp_in):
                h, blocks, o, blocks_after = self._reps(i, layer, args, out, slice(4, None))
                full = {"act_in": lambda: h, "mix_in": lambda: self._mix(blocks, h), "mix_out": lambda: self._mix(blocks_after, o),
                        "mlp_in": lambda: mlp_in[:, 4:]}
                for r in reps:
                    caught[f"{r}.{v}"] = full[r]().float().reshape(-1, d).detach()

            handles = self._capture(fn)
            try:
                with torch.no_grad():
                    logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[:, 4:].float()
                    target = F.log_softmax(logits.reshape(-1, logits.shape[-1]), -1)
            finally:
                for hd in handles:
                    hd.remove()
            opt.zero_grad()
            total, losses = 0.0, {}
            for key, p in params.items():
                y = translate(p, caught[key])
                lp = F.log_softmax(self._unembed(y).float(), -1)
                loss = F.kl_div(lp, target, log_target=True, reduction="batchmean")
                loss.backward()
                losses[key] = loss.item()
                total += loss.item()
            opt.step()
            sched.step()
            if step % 20 == 0 or step == steps - 1:
                log.append({"step": step, "mean_kl": round(total / len(params), 3),
                            **{k: round(v, 3) for k, v in losses.items() if k.endswith((".12", ".24", ".36", ".41", ".46"))}})
                print(json.dumps(log[-1]), flush=True)
        os.makedirs("/cache/tuned", exist_ok=True)
        torch.save({k: {n: t.detach().cpu() for n, t in p.items()} for k, p in params.items()}, f"/cache/tuned/{name}.pt")
        cache.commit()
        return {"log": log}


    @modal.method()
    def train_future(self, name: str, reps: list[str], visits: list[int], n_prompts: int = 6000, rank: int = 128, epochs: int = 3,
                     batch: int = 32, lr: float = 2e-3, wd: float = 1e-4) -> dict:
        """A "future lens": a low-rank affine translator per visit, fitted so that the readout at the last prompt
        token matches the model's distribution over the first token of its reply's content. In chat format the
        next token is a control token (<|content_text|>), so the target is taken one position later, after
        appending it; for plain text the target is the ordinary next-token distribution.
        Training prompts: open-domain questions (nq_open), random arithmetic, and wikitext sentence prefixes."""
        import random

        import torch
        import torch.nn.functional as F
        from datasets import load_dataset

        torch.manual_seed(0)
        rng = random.Random(0)
        chat = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{}<|end_message|><|message_model|>"
        nq = [r["question"] for r in load_dataset("google-research-datasets/nq_open", split="train").select(range(20000))]
        wt = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train").select(range(60000))["text"] if len(t) > 200]
        prompts = []
        while len(prompts) < n_prompts:
            u = rng.random()
            if u < 0.5:
                q = nq[rng.randrange(len(nq))].strip().capitalize() + "?"
                q += rng.choice([" Answer in one word.", " Answer briefly.", ""])
                prompts.append((chat.format(q), True))
            elif u < 0.65:
                a, b = rng.randint(1, 60), rng.randint(1, 60)
                op = rng.choice(["+", "-", "*"])
                q = f"What is {a} {op} {b}?" + rng.choice([" Answer in one word.", " Answer with a number.", ""])
                prompts.append((chat.format(q), True))
            else:
                words = wt[rng.randrange(len(wt))].split()
                k = rng.randint(6, min(40, len(words) - 1))
                prompts.append((" ".join(words[:k]), False))
        ct = self.tok("<|content_text|>", add_special_tokens=False)["input_ids"]
        rows = []
        for text, is_chat in prompts:
            ids = self.tok(text, add_special_tokens=False)["input_ids"][-200:]
            p = len(ids) - 1
            rows.append((ids + ct if is_chat else ids, p, p + len(ct) if is_chat else p))
        X = {f"{r}.{v}": [] for r in reps for v in visits}
        Y = []
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                chunk = rows[s : s + batch]
                L = max(len(r[0]) for r in chunk)
                ids = torch.tensor([r[0] + [0] * (L - len(r[0])) for r in chunk], device="cuda")
                P = torch.tensor([r[1] for r in chunk], device="cuda")
                Tg = torch.tensor([r[2] for r in chunk], device="cuda")
                ar = torch.arange(len(chunk), device="cuda")

                def fn(v, i, layer, args, out, mlp_in):
                    if v not in visits:
                        return
                    if "act_in" in reps:
                        X[f"act_in.{v}"].append(args[0][ar, P].float().cpu())
                    if "mlp_in" in reps:
                        X[f"mlp_in.{v}"].append(mlp_in[ar, P].float().cpu())

                handles = self._capture(fn)
                try:
                    logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[ar, Tg].float()
                finally:
                    for hd in handles:
                        hd.remove()
                top = logits.topk(256, -1)  # keep the target compactly: top-256 log-probs, renormalised
                Y.append((top.indices.cpu(), F.log_softmax(top.values, -1).cpu()))
                if s % (batch * 20) == 0:
                    print(f"collected {s + len(chunk)} of {len(rows)}", flush=True)
        X = {k: torch.cat(v) for k, v in X.items()}
        Yi = torch.cat([y[0] for y in Y])
        Yl = torch.cat([y[1] for y in Y])
        n = len(Yi)
        n_val = n // 10
        d = self.lm.config.hidden_size
        out, log = {}, []
        for key, Xk in X.items():
            p = {"U": torch.zeros(d, rank, device="cuda", requires_grad=True),
                 "V": (torch.randn(rank, d, device="cuda") / d**0.5).requires_grad_(),
                 "b": torch.zeros(d, device="cuda", requires_grad=True)}
            opt = torch.optim.AdamW(list(p.values()), lr=lr, weight_decay=wd)
            Xc, Yic, Ylc = Xk.cuda(), Yi.cuda(), Yl.cuda()

            def loss_on(idx):
                y = translate(p, Xc[idx])
                lp = F.log_softmax(self._unembed(y).float(), -1).gather(-1, Yic[idx])
                return -(Ylc[idx].exp() * lp).sum(-1).mean()  # cross-entropy to the (truncated) target

            val = torch.arange(n_val, device="cuda")
            with torch.no_grad():
                v0 = loss_on(val).item()
            for ep in range(epochs):
                perm = torch.randperm(n - n_val, device="cuda") + n_val
                for s in range(0, len(perm), 64):
                    opt.zero_grad()
                    loss_on(perm[s : s + 64]).backward()
                    opt.step()
            with torch.no_grad():
                v1 = loss_on(val).item()
            log.append({"key": key, "val_ce_before": round(v0, 3), "val_ce_after": round(v1, 3)})
            print(json.dumps(log[-1]), flush=True)
            out[key] = {k: t.detach().cpu() for k, t in p.items()}
        os.makedirs("/cache/tuned", exist_ok=True)
        torch.save(out, f"/cache/tuned/{name}.pt")
        cache.commit()
        return {"log": log}


    @modal.method()
    def train_concept(self, name: str, visits: list[int], reps: list[str], n_seqs: int = 3000, per_seq: int = 8, rank: int = 128,
                      epochs: int = 3, lr: float = 2e-3, wd: float = 1e-4, typo_rate: float = 0.15, window: int = 4) -> dict:
        """A "concept lens": a low-rank translator per visit fitted so that the readout at the last token of a word
        names the words around it (the clean words i-window..i+window, the word itself counted twice) rather than the
        next token. Inputs have random typos injected (targets stay clean), so the lens learns to name the intended
        word. Trained on wikitext only."""
        import random
        import re

        import torch
        import torch.nn.functional as F
        from datasets import load_dataset

        torch.manual_seed(0)
        rng = random.Random(0)
        STOP = set("the a an and or but of to in on at for with by from as is was are were be been it its this that these those he she they "
                   "his her their them we you i not no so if then than there here which who whom what when where how also has have had do "
                   "does did can could will would may might shall should into onto about over under after before between".split())
        texts = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train").select(range(120000))["text"] if len(t) > 300]

        def misspell(w):
            i = rng.randrange(1, len(w) - 1)
            k = rng.randrange(4)
            if k == 0:
                return w[:i] + w[i + 1] + w[i] + w[i + 2 :]
            if k == 1:
                return w[:i] + w[i + 1 :]
            if k == 2:
                return w[:i] + w[i] + w[i:]
            return w[:i] + rng.choice("aeiou") + w[i + 1 :]

        first_tok = {}

        def ftok(w):
            if w not in first_tok:
                first_tok[w] = self.tok(" " + w, add_special_tokens=False)["input_ids"][0]
            return first_tok[w]

        seqs = []
        while len(seqs) < n_seqs:
            words = texts[rng.randrange(len(texts))].split()[:90]
            ids, ends, clean = [], [], []
            for j, wd_ in enumerate(words):
                core = re.sub(r"[^A-Za-z]", "", wd_)
                shown = wd_
                if len(core) >= 4 and core == wd_ and rng.random() < typo_rate:
                    shown = misspell(wd_)
                t = self.tok((" " if j else "") + shown, add_special_tokens=False)["input_ids"]
                if len(ids) + len(t) > 128:
                    break
                ids += t
                ends.append(len(ids) - 1)
                clean.append(core.lower() if len(core) >= 3 and core.lower() not in STOP else None)
            if len(ends) < 12:
                continue
            picks = rng.sample(range(4, len(ends)), min(per_seq, len(ends) - 4))
            targets = []
            for i in picks:
                bag = {}
                for j in range(max(0, i - window), min(len(clean), i + window + 1)):
                    if clean[j]:
                        tid = ftok(clean[j])
                        bag[tid] = bag.get(tid, 0) + (2.0 if j == i else 1.0)
                if bag:
                    targets.append((ends[i], bag))
            if targets:
                seqs.append((ids, targets))
        X = {f"{r}.{v}": [] for r in reps for v in visits}
        TI, TW = [], []
        with torch.no_grad():
            for s_ in range(0, len(seqs), 16):
                chunk = seqs[s_ : s_ + 16]
                L = max(len(c[0]) for c in chunk)
                ids = torch.tensor([c[0] + [0] * (L - len(c[0])) for c in chunk], device="cuda")
                rows = torch.tensor([b for b, c in enumerate(chunk) for _ in c[1]], device="cuda")
                cols = torch.tensor([p for c in chunk for p, _ in c[1]], device="cuda")
                for c in chunk:
                    for _, bag in c[1]:
                        k = list(bag)[:12]
                        wts = torch.tensor([bag[t] for t in k])
                        TI.append(torch.tensor(k + [k[0]] * (12 - len(k))))
                        TW.append(torch.cat([wts / wts.sum(), torch.zeros(12 - len(k))]))

                def fn(v, i, layer, args, out, mlp_in):
                    if v not in visits:
                        return
                    if "act_in" in reps:
                        X[f"act_in.{v}"].append(args[0][rows, cols].half().cpu())
                    if "mlp_in" in reps:
                        X[f"mlp_in.{v}"].append(mlp_in[rows, cols].half().cpu())

                handles = self._capture(fn)
                try:
                    self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
                finally:
                    for hd in handles:
                        hd.remove()
        X = {k: torch.cat(v) for k, v in X.items()}
        TI, TW = torch.stack(TI).cuda(), torch.stack(TW).cuda()
        n = len(TI)
        n_val = n // 10
        print(f"examples {n}", flush=True)
        d = self.lm.config.hidden_size
        out, log = {}, []
        for key, Xk in X.items():
            p = {"U": torch.zeros(d, rank, device="cuda", requires_grad=True),
                 "V": (torch.randn(rank, d, device="cuda") / d**0.5).requires_grad_(),
                 "b": torch.zeros(d, device="cuda", requires_grad=True)}
            opt = torch.optim.AdamW(list(p.values()), lr=lr, weight_decay=wd)
            Xc = Xk.cuda().float()

            def loss_on(idx):
                lp = F.log_softmax(self._unembed(translate(p, Xc[idx])).float(), -1).gather(-1, TI[idx])
                return -(TW[idx] * lp).sum(-1).mean()

            val = torch.arange(n_val, device="cuda")
            with torch.no_grad():
                v0 = loss_on(val).item()
            for ep in range(epochs):
                perm = torch.randperm(n - n_val, device="cuda") + n_val
                for s_ in range(0, len(perm), 64):
                    opt.zero_grad()
                    loss_on(perm[s_ : s_ + 64]).backward()
                    opt.step()
            with torch.no_grad():
                v1 = loss_on(val).item()
            log.append({"key": key, "val_ce_before": round(v0, 3), "val_ce_after": round(v1, 3)})
            print(json.dumps(log[-1]), flush=True)
            out[key] = {k: t.detach().cpu() for k, t in p.items()}
            del Xc
        os.makedirs("/cache/tuned", exist_ok=True)
        torch.save(out, f"/cache/tuned/{name}.pt")
        cache.commit()
        return {"log": log}

    @modal.method()
    def stats(self, questions: list[dict]) -> dict:
        """Compact numbers about how the model routes state, for a text summary: per visit (at the read position,
        averaged over questions) the attention_input / mlp_input / final_input weights over the available blocks
        (prelude/embedding blocks first, active block last), the message gates' mean forget and write strength,
        and parameter-level summaries (query norms, top singular values of the message-gate down projections)."""
        import torch

        acc = {}

        def add(key, vec):
            a = acc.setdefault(key, [torch.zeros_like(vec), 0])
            a[0] += vec
            a[1] += 1

        def mix_weights(mod, blocks, h):
            stacked = torch.cat([blocks, h.unsqueeze(-2)], dim=-2).float()
            stacked = mod.norm(stacked)
            q = mod.query.float().view(-1, stacked.shape[-1])[0]
            return torch.softmax(stacked @ q * stacked.shape[-1] ** -0.5, -1)

        gate_log = {}
        for q in questions:
            ids = self.tok(q["model_input"], add_special_tokens=False)["input_ids"]
            p = q["read_position"]
            cur = {}

            def fn(v, i, layer, args, out, mlp_in):
                h, blocks = args[0][0, p], args[1][0, p]
                if layer.attention_input is not None and blocks.shape[0] > 0:
                    add(f"attn_mix.{v}", mix_weights(layer.attention_input, blocks, h).cpu())
                add(f"final_mix.{v}", mix_weights(self.lm.final_input, blocks, h).cpu())
                add(f"norms.{v}", torch.tensor([h.float().norm().item()] + [b.float().norm().item() for b in blocks]))

            handles = self._capture(fn)
            for i, layer in enumerate(self.lm.layers):
                for gname in ("q_message_gate", "mlp_message_gate"):
                    if hasattr(layer, gname):
                        def ghook(mod, args, out, i=i, gname=gname):
                            hs, msg = args[0][0, p].float(), args[1][0, p].float()
                            gh = torch.cat([mod.norm(hs), mod.norm(msg)], -1)
                            gh = mod.down_proj(gh.to(mod.down_proj.weight.dtype)).float()
                            fg = torch.tanh(mod.forget_gate_proj(gh)).abs().mean().item()
                            mg = torch.tanh(mod.message_gate_proj(gh)).abs().mean().item()
                            k = f"{gname}.L{i}"
                            gate_log.setdefault(k, []).append((fg, mg))
                        handles.append(getattr(layer, gname).register_forward_hook(ghook))
            try:
                with torch.no_grad():
                    x = torch.tensor([ids], device="cuda")
                    self.model(input_ids=x, attention_mask=torch.ones_like(x))
            finally:
                for hd in handles:
                    hd.remove()
        res = {k: [round(x, 3) for x in (a[0] / a[1]).tolist()] for k, a in acc.items()}
        res["gates"] = {k: [round(sum(a for a, _ in v) / len(v), 3), round(sum(b for _, b in v) / len(v), 3)] for k, v in gate_log.items()}
        params = {}
        for i, layer in enumerate(self.lm.layers):
            if hasattr(layer, "mlp_message_gate"):
                sv = torch.linalg.svdvals(layer.mlp_message_gate.down_proj.weight.float())[:5]
                params[f"mlp_message_gate.down.L{i}.top_sv"] = [round(x, 2) for x in sv.tolist()]
            params[f"mlp_input.query_norm.L{i}"] = round(layer.mlp_input.query.float().norm().item(), 2)
        params["final_input.query_norm"] = round(self.lm.final_input.query.float().norm().item(), 2)
        res["params"] = params
        return res


    def _prim(self, spec, args, blocks, h):
        """Evaluate one readout primitive from the WSP grammar on this visit's state ([b, 1, ...] slices)."""
        import torch

        op = spec["op"]

        def subset(which):
            if which == "no_prelude":
                return blocks[..., 2:, :]
            if which == "last1":
                return blocks[..., -1:, :]
            if which == "last2":
                return blocks[..., -2:, :]
            return blocks

        if op == "block":
            return blocks[..., spec["index"], :] if blocks.shape[-2] > abs(spec["index"]) - (spec["index"] < 0) else None
        if op == "message":
            if len(args) <= 2 or args[2] is None:
                return None
            return args[2][..., "qkvm".index(spec["which"][0]), :]
        if op == "sum":
            return subset(spec.get("blocks", "all")).sum(-2) + spec.get("active_scale", 1.0) * h
        if op == "mix":
            L = spec.get("module_layer", 0)
            mod = self.lm.final_input if spec["module"] == "final_input" else getattr(self.lm.layers[L], spec["module"])
            if mod is None:
                return None
            return mod(subset(spec.get("blocks", "all")), spec.get("active_scale", 1.0) * h)[0]
        if op == "center":
            a, b = self._prim(spec["base"], args, blocks, h), self._prim(spec["minus"], args, blocks, h)
            return None if a is None or b is None else a - spec.get("coef", 1.0) * b
        raise ValueError(op)

    @modal.method()
    def custom_readouts(self, questions: list[dict], designs: list[dict]) -> list[dict]:
        import torch

        results = []
        for q in questions:
            ids = self.tok(q["model_input"], add_special_tokens=False)["input_ids"]
            p = q["read_position"]
            per = {}

            def fn(v, i, layer, args, out, mlp_in):
                h, blocks = args[0][:, p : p + 1], args[1][:, p : p + 1]
                sl_args = [a[:, p : p + 1] if a is not None else None for a in args]
                for d in designs:
                    if v in d["visits"]:
                        x = self._prim(d["spec"], sl_args, blocks, h)
                        if x is not None:
                            per.setdefault(v, {})[d["name"]] = self._unembed(x).float().reshape(-1).topk(TOPK).indices.tolist()

            handles = self._capture(fn)
            try:
                with torch.no_grad():
                    x = torch.tensor([ids], device="cuda")
                    self.model(input_ids=x, attention_mask=torch.ones_like(x))
            finally:
                for hd in handles:
                    hd.remove()
            results.append({"id": q["id"], "visits": per})
        return results

    @modal.method()
    def export(self, name: str, keys: list[str]) -> bytes:
        """The chosen translators as one safetensors file (bf16), keyed '<rep>.<visit>.<W|U|V|b>'."""
        import torch
        from safetensors.torch import save

        W = torch.load(f"/cache/tuned/{name}.pt")
        return save({f"{k}.{n}": t.to(torch.bfloat16).contiguous() for k in keys for n, t in W[k].items()})


def _write(rows, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} to {out}")


@app.local_entrypoint()
def readouts(questions: str = "dev/data/all_private.jsonl", out: str = "dev/runs/readouts.jsonl"):
    qs = [json.loads(line) for line in open(questions) if line.strip()]
    _write(Lab().readouts.remote(qs), out)


@app.local_entrypoint()
def train(name: str = "tuned_v1", reps: str = "mix_in,act_in", steps: int = 400, rank: int = 0,
          questions: str = "dev/data/all_private.jsonl", out: str = ""):
    lab = Lab()
    res = lab.train.remote(name, reps.split(","), steps=steps, rank=rank)
    os.makedirs("dev/runs", exist_ok=True)
    json.dump(res, open(f"dev/runs/{name}_log.json", "w"))
    qs = [json.loads(line) for line in open(questions) if line.strip()]
    _write(lab.apply_tuned.remote(qs, name, reps.split(",")), out or f"dev/runs/{name}_readouts.jsonl")


@app.local_entrypoint()
def train_future(name: str = "future_v1", reps: str = "act_in,mlp_in", visits: str = "24,30,33,34,35,36,37,40,41,42,43,44,45,46,47",
                 n_prompts: int = 6000, rank: int = 128, questions: str = "dev/data/all_private.jsonl"):
    lab = Lab()
    res = lab.train_future.remote(name, reps.split(","), [int(v) for v in visits.split(",")], n_prompts=n_prompts, rank=rank)
    json.dump(res, open(f"dev/runs/{name}_log.json", "w"))
    qs = [json.loads(line) for line in open(questions) if line.strip()]
    _write(lab.apply_tuned.remote(qs, name, reps.split(","), label="future"), f"dev/runs/{name}_readouts.jsonl")


@app.local_entrypoint()
def tuned_readouts(name: str = "tuned_v1", reps: str = "mix_in,act_in", questions: str = "dev/data/all_private.jsonl", out: str = "",
                   label: str = "tuned"):
    qs = [json.loads(line) for line in open(questions) if line.strip()]
    chunks = [qs[i : i + 220] for i in range(0, len(qs), 220)]
    lab = Lab()
    rows = [r for part in lab.apply_tuned.map(chunks, kwargs={"weights_name": name, "reps": reps.split(","), "label": label}) for r in part]
    _write(rows, out or f"dev/runs/{name}_readouts.jsonl")


@app.local_entrypoint()
def round3(questions: str = "dev/data/new3_private.jsonl", all_questions: str = "dev/data/all_private.jsonl"):
    """Cache the lens bank for the new questions, fit the concept lens, and gather routing stats, in parallel."""
    lab = Lab()
    qs = [json.loads(line) for line in open(questions) if line.strip()]
    allq = [json.loads(line) for line in open(all_questions) if line.strip()]
    chunks = [qs[i : i + 90] for i in range(0, len(qs), 90)]
    concept = lab.train_concept.spawn("concept_v1", list(range(20, 48)), ["mlp_in"])
    stats = lab.stats.spawn(allq[:120])
    raw = [r for part in lab.readouts.map(chunks) for r in part]
    _write(raw, "dev/runs/readouts_v3.jsonl")
    _write(lab.apply_tuned.remote(qs, "tuned_v1", ["act_in", "mlp_in"]), "dev/runs/tuned_v1_v3_readouts.jsonl")
    _write(lab.apply_tuned.remote(qs, "future_v1", ["act_in", "mlp_in"], label="future"), "dev/runs/future_v1_v3_readouts.jsonl")
    json.dump(stats.get(), open("dev/runs/routing_stats.json", "w"))
    print("stats saved")
    json.dump(concept.get(), open("dev/runs/concept_v1_log.json", "w"))
    _write(lab.apply_tuned.remote(allq, "concept_v1", ["mlp_in"], label="concept"), "dev/runs/concept_v1_readouts.jsonl")


@app.local_entrypoint()
def custom(designs: str, questions: str = "dev/data/all_private.jsonl", out: str = "dev/runs/wsp_readouts.jsonl"):
    ds = json.load(open(designs))["designs"]
    qs = [json.loads(line) for line in open(questions) if line.strip()]
    chunks = [qs[i : i + 220] for i in range(0, len(qs), 220)]
    rows = [r for part in Lab().custom_readouts.map(chunks, kwargs={"designs": ds}) for r in part]
    _write(rows, out)


@app.local_entrypoint()
def export(name: str, keys: str, out: str):
    data = Lab().export.remote(name, keys.split(","))
    open(out, "wb").write(data)
    print(f"wrote {len(data) / 1e6:.1f} MB to {out}")
