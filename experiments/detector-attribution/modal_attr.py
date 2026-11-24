"""Modal app `detector-attribution`: four open AI-text detectors, read from the inside.

    MODAL_PROFILE=albertlieyingadrian modal deploy experiments/detector-attribution/modal_attr.py

Det(name) for name in 9b | 1.7b | voi | desklib:
- score(texts): the detector's published score (bucket heads: expected bucket / 3 via slop-detector's own `predict`;
  one-logit heads: sigmoid).
- ig(texts): integrated gradients of logit(score) over the input embeddings (baseline: the pad token's embedding at
  every
  position, special tokens held fixed; a zero baseline is flat through the embedding LayerNorm), midpoint rule; one
  number per token, with the token's character span, f(text) and f(baseline) for the completeness check.
- 9b only: sae_attr (Qwen-Scope feature attribution at one layer, plus the SAE's FVU on this detector),
  sae_ablate (score with feature sets subtracted, SAE error kept), feature_acts (per-token activations of chosen
  features, for labelling).
Gen(): Qwen3-8B (the model that wrote the AI replies) with the Qwen-Scope layer-18 SAE: per-text mean activation of
every feature, and per-token activations of chosen features. Reading only.

The image has no flash-linear-attention: its backward pass refuses Hopper GPUs (fla #640), so Qwen3.5 runs on the
PyTorch reference kernels. Texts here are at most 195 words, so every detector sees all of each text.
"""

import modal

app = modal.App("detector-attribution")
sd_vol = modal.Volume.from_name("slop-detector")  # f9-9b adapter and the shared Hugging Face cache (/vol/hf)
da_vol = modal.Volume.from_name("detector-attribution")  # the 1.7B adapter, copied from the modal4 workspace
image = (
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
        "tiktoken",
        "protobuf",
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
    .add_local_file("../slop-detector/common.py", "/root/sd/common.py")
    .add_local_file("../slop-detector/04_train.py", "/root/sd/train.py")
    .add_local_file("common.py", "/root/da/common.py")
)
VOI380, VOI380_REV = "VoightKampffLabs/voi-slop-detect-380M", "a109d591b8b62541f362d873765900380b2bd0ea"
DESKLIB = "desklib/ai-text-detector-v1.01"
RUNS = {"9b": ("/vol/runs/f9-9b", "Qwen/Qwen3.5-9B"), "1.7b": ("/da/runs/b1b-1.7bB", "Qwen/Qwen3-1.7B-Base")}
SAE9 = ("Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50", 50)
SAE8 = ("Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", 50, 18)
IG_STEPS = 64
VOLS = {"/vol": sd_vol, "/da": da_vol}


def _decoder_layers(model):
    import torch

    best = None
    for name, mod in model.named_modules():
        is_layers = isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and "visual" not in name
        if is_layers and (best is None or len(mod) > len(best)):
            best = mod
    return best


@app.cls(image=image, gpu="H100", volumes=VOLS, timeout=3 * 60 * 60, max_containers=4, scaledown_window=120)
class Det:
    name: str = modal.parameter()
    fp32: bool = modal.parameter(default=False)  # IG runs in float32: bf16 gradients broke the completeness check

    @modal.enter()
    def load(self) -> None:
        import sys

        import torch

        self.torch = torch
        dev = "cuda"
        if self.name in RUNS:
            sys.path.insert(0, "/root/sd")
            import train as tr
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            self.tr = tr
            path, base = RUNS[self.name]
            tok, model = tr.load(base, dev, num_labels=4)
            set_peft_model_state_dict(model, load_file(f"{path}/adapter/adapter_model.safetensors"))
            self.model = model.merge_and_unload().eval()
            self.tok = tok
        elif self.name == "voi":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(VOI380, revision=VOI380_REV)
            self.model = AutoModelForSequenceClassification.from_pretrained(VOI380, revision=VOI380_REV).to(dev).eval()
        elif self.name == "desklib":
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(DESKLIB)
            config = AutoConfig.from_pretrained(DESKLIB)
            self.model = AutoModel.from_config(config)
            state = load_file(hf_hub_download(DESKLIB, "model.safetensors"))
            body = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
            missing, unexpected = self.model.load_state_dict(body, strict=False)
            print(f"desklib: {len(missing)} missing, {len(unexpected)} unexpected keys")
            self.head = torch.nn.Linear(config.hidden_size, 1)
            self.head.load_state_dict({"weight": state["classifier.weight"], "bias": state["classifier.bias"]})
            self.model, self.head = self.model.to(dev).eval(), self.head.to(dev).eval()
        else:
            raise ValueError(self.name)
        if self.fp32:
            self.model = self.model.float()
            if self.name == "desklib":
                self.head = self.head.float()
        self.emb = self.model.get_input_embeddings()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._sae: dict = {}

    # ---------------------------------------------------------------------------------------------------- forward
    def _enc(self, text: str):
        enc = self.tok(text, truncation=True, max_length=512, return_offsets_mapping=True, return_tensors="pt")
        return enc["input_ids"].cuda(), enc["attention_mask"].cuda(), enc["offset_mapping"][0].tolist()

    def _logit_from_embeds(self, embeds, mask):
        """logit(score) for a batch of embeddings (no padding: every row is the same text)."""
        torch = self.torch
        if self.name in RUNS:
            # logit(s) for s = E[bucket] / 3, in log space so a saturated score keeps its gradient:
            # s = sum_k p_k k / 3 and 1 - s = sum_k p_k (3 - k) / 3
            lp = torch.log_softmax(self.model(inputs_embeds=embeds, attention_mask=mask).logits.float(), -1)
            k = torch.arange(4, device=lp.device, dtype=lp.dtype)
            log_s = torch.logsumexp(lp[..., 1:] + torch.log(k[1:]), -1)
            log_1ms = torch.logsumexp(lp[..., :3] + torch.log(3 - k[:3]), -1)
            return log_s - log_1ms
        if self.name == "voi":
            return self.model(inputs_embeds=embeds, attention_mask=mask).logits.float().reshape(-1)
        hidden = self.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return self.head((hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().reshape(-1)

    @modal.method()
    def score(self, texts: list[str]) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            if self.name in RUNS:
                rows = self.tr.encode([{"text": t or "."} for t in texts], self.tok)
                return [
                    round(float(x), 6) for x in self.tr.predict(self.model, rows, self.tok, "cuda", max_tokens=16384)
                ]
            out = []
            for t in texts:
                ids, mask, _ = self._enc(t or ".")
                out.append(round(float(torch.sigmoid(self._logit_from_embeds(self.emb(ids), mask))[0]), 6))
        return out

    def _margins(self, texts: list[str]):
        """logit(score) for many texts, batched with padding, at full precision (numpy float64)."""
        import numpy as np

        torch = self.torch
        out = np.zeros(len(texts))
        with torch.no_grad():
            if self.name in RUNS:
                rows = self.tr.encode([{"text": t or "."} for t in texts], self.tok)
                for idx in self.tr.batches(rows, 16384, shuffle=False):
                    lg = self.model(**self.tr.collate(rows, idx, self.tok, "cuda")).logits.double()
                    p = torch.softmax(lg, -1)
                    s = ((p * torch.arange(4, device=p.device, dtype=p.dtype)).sum(-1) / 3).clamp(1e-12, 1 - 1e-12)
                    out[idx] = (torch.log(s) - torch.log1p(-s)).cpu().numpy()
                return out
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            for b in range(0, len(order), 64):
                idx = order[b : b + 64]
                enc = self.tok(
                    [texts[i] or "." for i in idx], truncation=True, max_length=512, padding=True, return_tensors="pt"
                ).to("cuda")
                if self.name == "voi":
                    lg = self.model(**enc).logits.double().reshape(-1)
                else:
                    hidden = self.model(**enc).last_hidden_state
                    m = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    lg = self.head((hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)).double().reshape(-1)
                out[idx] = lg.cpu().numpy()
        return out

    @modal.method()
    def margins(self, texts: list[str]) -> list[float]:
        return [float(v) for v in self._margins(texts)]

    @modal.method()
    def occlude(self, texts: list[str]) -> list[dict]:
        """Leave-one-word-out: for each word unit (common.words), how much deleting it lowers logit(score)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("da_common", "/root/da/common.py")  # slop-detector's is `common`
        da = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(da)
        delete_spans, words = da.delete_spans, da.words

        res = []
        for t in texts:
            spans = words(t)
            m = self._margins([t] + [delete_spans(t, [sp]) for sp in spans])
            res.append({"base": float(m[0]), "drop": [round(float(m[0] - v), 5) for v in m[1:]]})
        return res

    @modal.method()
    def ig(self, texts: list[str], steps: int = IG_STEPS) -> list[dict]:
        torch = self.torch
        chunk = 8 if self.name == "9b" else 16
        res = []
        for t in texts:
            ids, mask, offs = self._enc(t)
            with torch.no_grad():
                x = self.emb(ids)[0].float()  # (L, d)
                special = torch.tensor([e <= s for s, e in offs], device="cuda")
                x0 = self.emb(torch.full_like(ids, self.tok.pad_token_id))[0].float()
                x0 = torch.where(special[:, None], x, x0)
            dt = self.emb.weight.dtype
            grads = torch.zeros_like(x)
            alphas = [(k + 0.5) / steps for k in range(steps)]
            for i in range(0, steps, chunk):
                a = torch.tensor(alphas[i : i + chunk], device="cuda").view(-1, 1, 1)
                e = (x0 + a * (x - x0).unsqueeze(0)).to(dt).detach().requires_grad_(True)
                f = self._logit_from_embeds(e, mask.expand(e.shape[0], -1))
                (g,) = torch.autograd.grad(f.sum(), e)
                grads += g.float().sum(0)
            attr = ((x - x0) * grads / steps).sum(-1)
            with torch.no_grad():
                fx = float(self._logit_from_embeds(x.unsqueeze(0).to(dt), mask)[0])
                f0 = float(self._logit_from_embeds(x0.unsqueeze(0).to(dt), mask)[0])
            res.append(
                {
                    "offsets": offs,
                    "attr": [round(float(v), 5) for v in attr.tolist()],
                    "fx": fx,
                    "f0": f0,
                    "n_tokens": int(x.shape[0]),
                }
            )
        return res

    # ---------------------------------------------------------------------------------------------------- SAE (9b)
    def _sae_layer(self, layer: int):
        torch = self.torch
        if layer not in self._sae:
            from huggingface_hub import hf_hub_download

            st = torch.load(hf_hub_download(SAE9[0], f"layer{layer}.sae.pt"), map_location="cuda", weights_only=True)
            self._sae.clear()
            self._sae[layer] = {k: st[k].float().cuda() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
        return self._sae[layer]

    def _codes(self, h, P):
        """Top-k SAE codes of h (L, d) as a dense (L, F) matrix."""
        pre = h @ P["W_enc"].T + P["b_enc"]
        val, idx = pre.topk(SAE9[1], dim=-1)
        return self.torch.zeros(h.shape[0], P["W_enc"].shape[0], device="cuda").scatter_(-1, idx, val.clamp(min=0))

    @modal.method()
    def sae_attr(self, texts: list[str], layer: int) -> list[dict]:
        """Per text: active features with attribution sum_pos a_f (d_f . dlogit/dh), activation mass, FVU."""
        P = self._sae_layer(layer)
        blk = _decoder_layers(self.model)[layer]
        dnorm = P["W_dec"].norm(dim=0)
        res = []
        for t in texts:
            ids, mask, _ = self._enc(t)
            store = {}

            def hook(_m, _i, out, store=store):
                h = out[0] if isinstance(out, tuple) else out
                h.retain_grad()
                store["h"] = h

            handle = blk.register_forward_hook(hook)
            try:
                e = self.emb(ids).detach().requires_grad_(True)
                f = self._logit_from_embeds(e, mask)
                f.sum().backward()
            finally:
                handle.remove()
            h = store["h"][0].detach().float()
            g = store["h"].grad[0].float()
            rec = self._codes(h, P)
            recon = rec @ P["W_dec"].T + P["b_dec"]
            err = (h - recon).pow(2).sum(-1)
            var = (h - h.mean(0)).pow(2).sum(-1)
            active = (rec > 0).any(0).nonzero().reshape(-1)
            attr = (rec[:, active] * (g @ P["W_dec"][:, active])).sum(0)
            mass = (rec[:, active] * dnorm[active]).sum(0)
            res.append(
                {
                    "idx": active.tolist(),
                    "attr": [round(float(v), 6) for v in attr.tolist()],
                    "mass": [round(float(v), 4) for v in mass.tolist()],
                    "fx": float(f[0]),
                    "fvu": float(err.sum() / var.sum()),
                    "fvu_no0": float(err[1:].sum() / var[1:].sum()),
                    "n_tokens": int(h.shape[0]),
                }
            )
        return res

    @modal.method()
    def sae_ablate(self, texts: list[str], layer: int, sets: dict[str, list[int]]) -> list[dict]:
        """logit(score) of each text with each feature set's decoded part subtracted at `layer` (SAE error kept)."""
        torch = self.torch
        P = self._sae_layer(layer)
        blk = _decoder_layers(self.model)[layer]
        names = ["none", *sets]
        res = []
        with torch.no_grad():
            for t in texts:
                ids, mask, _ = self._enc(t)
                out = {}
                for nm in names:
                    fs = torch.tensor(sets.get(nm, []), device="cuda", dtype=torch.long)

                    def hook(_m, _i, o, fs=fs):
                        if fs.numel() == 0:
                            return o
                        h = o[0] if isinstance(o, tuple) else o
                        hf = h.float()
                        rec = self._codes(hf[0], P)
                        new = (hf - (rec[:, fs] @ P["W_dec"][:, fs].T).unsqueeze(0)).to(h.dtype)
                        return (new, *o[1:]) if isinstance(o, tuple) else new

                    handle = blk.register_forward_hook(hook)
                    try:
                        lg = self._logit_from_embeds(self.emb(ids), mask)
                    finally:
                        handle.remove()
                    out[nm] = round(float(lg[0]), 5)  # logit(score); score = sigmoid
                res.append(out)
        return res

    @modal.method()
    def feature_acts(self, texts: list[str], layer: int, feats: list[int]) -> list[dict]:
        torch = self.torch
        P = self._sae_layer(layer)
        blk = _decoder_layers(self.model)[layer]
        fs = torch.tensor(feats, device="cuda", dtype=torch.long)
        res = []
        with torch.no_grad():
            for t in texts:
                ids, mask, offs = self._enc(t)
                store = {}

                def hook(_m, _i, o, store=store):
                    store["h"] = (o[0] if isinstance(o, tuple) else o)[0].float()

                handle = blk.register_forward_hook(hook)
                try:
                    self._logit_from_embeds(self.emb(ids), mask)
                finally:
                    handle.remove()
                rec = self._codes(store["h"], P)
                res.append({"offsets": offs, "acts": [[round(float(v), 3) for v in r] for r in rec[:, fs].T.tolist()]})
        return res


@app.cls(image=image, gpu="H100", volumes=VOLS, timeout=2 * 60 * 60, max_containers=2, scaledown_window=120)
class Gen:
    """Qwen3-8B, the model that wrote the AI replies, read with Qwen-Scope's layer-18 SAE (reading only)."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_grad_enabled(False)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval()
        st = torch.load(hf_hub_download(SAE8[0], f"layer{SAE8[2]}.sae.pt"), map_location="cuda", weights_only=True)
        self.P = {k: st[k].float().cuda() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}

    def _acts(self, text: str):
        torch = self.torch
        enc = self.tok(text, truncation=True, max_length=512, return_offsets_mapping=True, return_tensors="pt")
        out = self.model(enc["input_ids"].cuda(), output_hidden_states=True)
        h = out.hidden_states[SAE8[2] + 1][0].float()  # output of decoder layer 18 (resid_post)
        pre = h @ self.P["W_enc"].T + self.P["b_enc"]
        val, idx = pre.topk(SAE8[1], dim=-1)
        rec = torch.zeros(h.shape[0], self.P["W_enc"].shape[0], device="cuda").scatter_(-1, idx, val.clamp(min=0))
        recon = rec @ self.P["W_dec"].T + self.P["b_dec"]
        fvu = float(((h[1:] - recon[1:]) ** 2).sum() / ((h[1:] - h[1:].mean(0)) ** 2).sum())
        return rec, enc["offset_mapping"][0].tolist(), fvu

    @modal.method()
    def means(self, texts: list[str]) -> list[dict]:
        res = []
        for t in texts:
            rec, _, fvu = self._acts(t)
            m = rec[1:].mean(0)  # position 0 (the attention sink) left out
            nz = m.nonzero().reshape(-1)
            res.append({"idx": nz.tolist(), "mean": [round(float(v), 5) for v in m[nz].tolist()], "fvu": fvu})
        return res

    @modal.method()
    def acts(self, texts: list[str], feats: list[int]) -> list[dict]:
        torch = self.torch
        fs = torch.tensor(feats, device="cuda", dtype=torch.long)
        res = []
        for t in texts:
            rec, offs, _ = self._acts(t)
            res.append({"offsets": offs, "acts": [[round(float(v), 3) for v in r] for r in rec[:, fs].T.tolist()]})
        return res
