"""The steering worker: one open chat model on a Modal H100, with a forward hook that adds a vector after a block and
soft prompts spliced in as input embeddings. COSTS MONEY; callers check the cap first (`common.check_modal`).

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/culture-steering/modal_steer.py

Methods (one container per model, kept warm between calls):
- acts(jobs, layers, mode): residual stream (hidden_states[L], the output of block L-1) at the last prompt token
  ("last") or averaged over the reply's tokens ("response"), plus the mean norm per layer.
- letter_probs(jobs, vectors, softs): exact next-token probability of each option letter A-J after the chat template.
- generate(jobs, vectors, softs, ...): sampled replies, each with its mean per-token NLL under the unsteered model given
  the job's `nll_messages` (fluency).
- train_soft(items, ...): trains one soft prompt (n virtual tokens at the SOFT_MARK) on survey splits, model frozen.

A job's steering is {"vec": name in `vectors`, "layer": L, "coef": c}: c x unit(vector) is added to every position's
residual at hidden_states[L]. Jobs that share a layer and soft prompt share batches, each row with its own vector.
"soft": name in `softs` splices that soft prompt in at SOFT_MARK.
"""

import time

import modal

MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct"}
LETTERS = "ABCDEFGHIJ"
SOFT_MARK = "<<SOFT>>"

app = modal.App("culture-steering")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("culture-steering-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.10.1", "hf_transfer==0.1.9", "numpy<2.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)


@app.cls(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=3 * 60 * 60, scaledown_window=180)
class Steerer:
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        hf_id = MODELS[self.model_key]
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cache.commit()
        self.emb = self.model.get_input_embeddings()
        self.state = {"layer": None, "v": None}
        for i, block in enumerate(self.model.model.layers):
            block.register_forward_hook(self._hook(i + 1))  # block i's output is hidden_states[i + 1]
        self.letter_ids = []
        for x in LETTERS:
            ids = {
                self.tok.encode(x, add_special_tokens=False)[0],
                self.tok.encode(" " + x, add_special_tokens=False)[0],
            }
            assert all(self.tok.decode([i]).strip() == x for i in ids), (x, ids)
            self.letter_ids.append(sorted(ids))
        self.eos = [self.tok.eos_token_id]
        if "llama" in self.model_key:
            self.eos.append(self.tok.convert_tokens_to_ids("<|eot_id|>"))

    def _hook(self, layer):
        def hook(_m, _i, out):
            v = self.state["v"]
            if v is None or self.state["layer"] != layer:
                return out
            if isinstance(out, tuple):
                return (out[0] + v.to(out[0].dtype),) + tuple(out[1:])
            return out + v.to(out.dtype)

        return hook

    def _set(self, part, vectors, layer, n=1):
        """Per-row steering for one batch: row i gets coef_i x unit(vec_i) (zeros when the job has no vector)."""
        torch = self.torch
        if layer is None or not any(j.get("vec") and j.get("coef") for j in part):
            self.state.update(layer=None, v=None)
            return
        units = self.state.setdefault("units", {})
        rows = []
        for j in part:
            if j.get("vec") and j.get("coef"):
                if j["vec"] not in units:
                    v = torch.tensor(vectors[j["vec"]], device="cuda", dtype=torch.float32)
                    units[j["vec"]] = v / v.norm()
                rows.append(j["coef"] * units[j["vec"]])
            else:
                rows.append(torch.zeros(self.model.config.hidden_size, device="cuda"))
        V = torch.stack(rows)[:, None, :]
        self.state.update(layer=layer, v=V.repeat_interleave(n, 0) if n > 1 else V)

    def _chat(self, messages, gen=True):
        kw = {"enable_thinking": False} if "qwen" in self.model_key else {}
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen, **kw)

    def _inputs(self, texts, soft=None):
        """Left-padded batch: input_ids, or inputs_embeds with the soft prompt spliced in at SOFT_MARK."""
        torch = self.torch
        if soft is None:
            texts = [t.replace(SOFT_MARK, "") for t in texts]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            m = enc.attention_mask
            return {"input_ids": enc.input_ids, "attention_mask": m, "position_ids": (m.cumsum(-1) - 1).clamp(min=0)}
        seqs = []
        for t in texts:
            a, b = t.split(SOFT_MARK, 1)
            ia = torch.tensor(self.tok.encode(a, add_special_tokens=False), device="cuda")
            ib = torch.tensor(self.tok.encode(b, add_special_tokens=False), device="cuda")
            seqs.append(torch.cat([self.emb(ia), soft.to(torch.bfloat16), self.emb(ib)]))
        n, d = max(len(s) for s in seqs), seqs[0].shape[1]
        E = torch.zeros(len(seqs), n, d, dtype=torch.bfloat16, device="cuda")
        M = torch.zeros(len(seqs), n, dtype=torch.long, device="cuda")
        for i, s in enumerate(seqs):
            E[i, n - len(s) :] = s
            M[i, n - len(s) :] = 1
        return {"inputs_embeds": E, "attention_mask": M, "position_ids": (M.cumsum(-1) - 1).clamp(min=0)}

    def _letter_logp(self, logits):
        """log P(letter) for A-J, summing the bare and space-prefixed tokens. logits: [B, V] -> [B, 10]."""
        torch = self.torch
        lp = torch.log_softmax(logits.float(), -1)
        return torch.stack([torch.logsumexp(lp[:, ids], -1) for ids in self.letter_ids], -1)

    @staticmethod
    def _groups(jobs, keys=("layer", "soft")):
        """Jobs sharing a steering layer and soft prompt run in the same batches (vectors differ per row)."""
        g = {}
        for i, j in enumerate(jobs):
            g.setdefault(tuple(j.get(k) for k in keys), []).append(i)
        return g

    @modal.method()
    def acts(self, jobs, layers, mode="last", batch=16):
        """jobs: [{"messages", "response"?}] -> {"acts": float16 [n, len(layers), d], "norms": {L: mean norm}}."""
        import numpy as np

        torch = self.torch
        t0 = time.time()
        self.state.update(layer=None, v=None)
        out = np.zeros((len(jobs), len(layers), self.model.config.hidden_size), np.float16)
        norms = {L: [] for L in layers}
        for s in range(0, len(jobs), batch):
            part = jobs[s : s + batch]
            prefixes = [self._chat(j["messages"]) for j in part]
            texts = [p + (j.get("response") or "") for p, j in zip(prefixes, part)]
            x = self._inputs(texts)
            with torch.no_grad():
                hs = self.model(**x, output_hidden_states=True).hidden_states
            n = x["input_ids"].shape[1]
            lens = x["attention_mask"].sum(-1).tolist()
            for k in range(len(part)):
                if mode == "last":
                    sl = slice(n - 1, n)
                else:
                    plen = len(self.tok.encode(prefixes[k], add_special_tokens=False))
                    sl = slice(n - lens[k] + plen, n)
                for li, L in enumerate(layers):
                    h = hs[L][k, sl].float()
                    out[s + k, li] = h.mean(0).cpu().numpy()
                    norms[L].append(float(h.norm(dim=-1).mean()))
        return {"acts": out, "norms": {L: float(np.mean(v)) for L, v in norms.items()}, "seconds": time.time() - t0}

    @modal.method()
    def letter_probs(self, jobs, vectors=None, softs=None, batch=32):
        """jobs: [{"messages", "vec"?, "layer"?, "coef"?, "soft"?}] -> P(letter) for A-J per job (not renormalised)."""
        torch = self.torch
        t0 = time.time()
        res = [None] * len(jobs)
        self.state["units"] = {}
        for (layer, sname), idxs in self._groups(jobs).items():
            soft = None if sname is None else torch.tensor(softs[sname], device="cuda")
            for s in range(0, len(idxs), batch):
                part = idxs[s : s + batch]
                self._set([jobs[i] for i in part], vectors or {}, layer)
                x = self._inputs([self._chat(jobs[i]["messages"]) for i in part], soft)
                with torch.no_grad():
                    logits = self.model(**x).logits[:, -1]
                p = self._letter_logp(logits).exp().cpu().tolist()
                for k, i in enumerate(part):
                    res[i] = [round(v, 7) for v in p[k]]
        self.state.update(layer=None, v=None)
        return {"probs": res, "seconds": time.time() - t0}

    @modal.method()
    def generate(self, jobs, vectors=None, softs=None, max_new_tokens=256, temperature=0.7, top_p=0.95, n=1, seed=0,
                 batch=32, nll=True):  # fmt: skip
        """jobs: [{"messages", "nll_messages"?, steering?}] -> {"texts": [[n replies]], "nll": [[n]]}."""
        torch = self.torch
        t0 = time.time()
        torch.manual_seed(seed)
        texts = [None] * len(jobs)
        self.state["units"] = {}
        for (layer, sname), idxs in self._groups(jobs).items():
            soft = None if sname is None else torch.tensor(softs[sname], device="cuda")
            for s in range(0, len(idxs), batch):
                part = idxs[s : s + batch]
                self._set([jobs[i] for i in part], vectors or {}, layer, n)
                x = self._inputs([self._chat(jobs[i]["messages"]) for i in part], soft)
                x.pop("position_ids")
                with torch.no_grad():
                    g = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                            top_p=top_p, top_k=0, repetition_penalty=1.0, num_return_sequences=n,
                                            pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos)  # fmt: skip
                if "input_ids" in x:
                    g = g[:, x["input_ids"].shape[1] :]
                dec = self.tok.batch_decode(g, skip_special_tokens=True)
                for k, i in enumerate(part):
                    texts[i] = [t.strip() for t in dec[k * n : (k + 1) * n]]
        self.state.update(layer=None, v=None)
        nlls = [[None] * n for _ in jobs]
        if nll:
            flat = [(i, r) for i in range(len(jobs)) for r in range(n)]
            nb = 8  # small batches: full-vocabulary logits for long multi-turn rows do not fit at batch 96
            for s in range(0, len(flat), nb):
                part = flat[s : s + nb]
                prefixes = [self._chat(jobs[i].get("nll_messages") or jobs[i]["messages"]).replace(SOFT_MARK, "")
                            for i, _ in part]  # fmt: skip
                x = self._inputs([p + texts[i][r] for p, (i, r) in zip(prefixes, part)])
                L = x["input_ids"].shape[1]
                lens = x["attention_mask"].sum(-1).tolist()
                with torch.no_grad():
                    logits = self.model(**x).logits
                    for k, (i, r) in enumerate(part):
                        plen = len(self.tok.encode(prefixes[k], add_special_tokens=False))
                        start = L - lens[k] + plen - 1  # the logit at position t predicts token t + 1
                        if start >= L - 1:
                            continue
                        lp = torch.nn.functional.cross_entropy(
                            logits[k, start:-1].float(), x["input_ids"][k, start + 1 :]
                        )
                        nlls[i][r] = round(float(lp), 4)
                del logits
        return {"texts": texts, "nll": nlls, "seconds": time.time() - t0}

    @modal.method()
    def train_soft(self, items, n_tokens=16, steps=150, lr=3e-3, batch=12, mass_weight=0.2, seed=0,
                   init_text="Please answer the following survey question honestly.", val_items=None):  # fmt: skip
        """items: [{"messages" (with SOFT_MARK), "target": [k probs]}]. Loss: KL(target || the model's split over the k
        options) + mass_weight x (−log mass on the k letters). Returns the soft prompt and the loss curve."""
        import numpy as np

        torch = self.torch
        t0 = time.time()
        g = torch.Generator().manual_seed(seed)
        self.state.update(layer=None, v=None)
        ids = self.tok.encode(init_text, add_special_tokens=False)
        ids = (ids * (n_tokens // len(ids) + 1))[:n_tokens]
        soft = self.emb(torch.tensor(ids, device="cuda")).detach().float().clone().requires_grad_(True)
        opt = torch.optim.Adam([soft], lr=lr)
        train_texts = [self._chat(it["messages"]) for it in items]
        val_texts = [self._chat(it["messages"]) for it in (val_items or [])]

        def loss_on(texts, its, grad):
            with torch.set_grad_enabled(grad):
                x = self._inputs(texts, soft)
                lp = self._letter_logp(self.model(**x).logits[:, -1])
                tot = 0.0
                for k, it in enumerate(its):
                    t = torch.tensor(it["target"], device="cuda", dtype=torch.float32).clamp(min=1e-8)
                    lpk = lp[k, : len(t)]
                    mass = torch.logsumexp(lpk, -1)
                    tot = tot + (t * (t.log() - (lpk - mass))).sum() - mass_weight * mass
                return tot / len(its)

        curve = []
        for step in range(steps):
            idx = torch.randperm(len(items), generator=g)[:batch].tolist()
            loss = loss_on([train_texts[i] for i in idx], [items[i] for i in idx], True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 0 or step == steps - 1:
                row = {"step": step, "train": round(float(loss), 4)}
                if val_items:
                    vl = [float(loss_on(val_texts[s : s + 16], val_items[s : s + 16], False)) * len(val_texts[s : s + 16])
                          for s in range(0, len(val_texts), 16)]  # fmt: skip
                    row["val"] = round(sum(vl) / len(val_texts), 4)
                curve.append(row)
        return {"soft": soft.detach().cpu().numpy().astype(np.float32), "curve": curve, "seconds": time.time() - t0}
