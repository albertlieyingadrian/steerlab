"""Steered generation and SAE readouts for Llama-3.3-70B and Llama-3.1-8B on Modal. COSTS MONEY (see TIMEOUT).

Shared by both replications in this series (`repl-steering-resistance`, `repl-self-reference`).

    export MODAL_PROFILE=credits3
    .venv/bin/modal deploy experiments/repl-steering-resistance/modal_engine.py
    # then drive it from run.py (it spawns calls and pulls results)

What it does, in plain terms:
  * loads the model in bfloat16 (70B on one B200, 8B on one H100) and the Goodfire SAE for that model;
  * adds a vector to the residual stream at the output of one decoder layer, at every position of the prompt and the
    reply (the same place and the same maths as the paper's vLLM fork: `boost x W_dec[:, f]`);
  * optionally zero-ablates a list of SAE latents at that layer (encode, subtract `a_i x W_dec[:, i]`), which is the
    fork's "clamp to 0" with the SAE error term kept;
  * samples replies (several user turns when a job has follow-ups) and writes them to the volume as it goes, so a
    restart skips finished jobs;
  * reads SAE activations (max over positions) for a list of texts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP = "aiaf-repl-engine"
VOL = "/vol"
MODELS = {
    "70b": {
        "repo": "meta-llama/Llama-3.3-70B-Instruct",
        "sae_repo": "Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
        "sae_file": "Llama-3.3-70B-Instruct-SAE-l50.pt",
        "sae_layer": 50,
    },
    "8b": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "sae_repo": "Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
        "sae_file": "Llama-3.1-8B-Instruct-SAE-l19.pth",
        "sae_layer": 19,
    },
}
CACHE = f"{VOL}/hf/hub"
QUIRK_IDS = [128000, 128006, 78191, 128007]  # the released code appends these after the chat template (vllm_engine.py)
PRICE = {"B200": 6.25, "H100": 3.95}  # $/h, Modal list price 2026-11
TIMEOUT = {"70b": 3 * 3600, "8b": 2 * 3600}  # hard stop per call: <= $18.75 (70b) or $7.90 (8b) per call

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.1", "transformers==5.20.2", "accelerate==1.15.0", "numpy<2.4", "hf_xet>=1.1", "huggingface_hub"
    )
    .env({"TOKENIZERS_PARALLELISM": "false", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)
vol = modal.Volume.from_name("aiaf-repl", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App(APP, image=image)


@app.function(cpu=4, memory=16384, timeout=3 * 3600, secrets=[hf_secret], volumes={VOL: vol})
def download(keys: list[str]) -> dict:
    """Model weights and SAE files into this series' volume (CPU only)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    out = {}
    for k in keys:
        snapshot_download(MODELS[k]["repo"], cache_dir=CACHE, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
        vol.commit()
        p = hf_hub_download(MODELS[k]["sae_repo"], MODELS[k]["sae_file"], cache_dir=CACHE)
        out[k] = Path(p).stat().st_size
        vol.commit()
    return out


class _Engine:
    def __init__(self, key: str):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = MODELS[key]
        t0 = time.time()
        self.key = key
        self.tok = AutoTokenizer.from_pretrained(cfg["repo"], cache_dir=CACHE)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = "<|finetune_right_pad_id|>"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["repo"], cache_dir=CACHE, dtype=torch.bfloat16, device_map="cuda"
        )
        self.model.eval()
        sae_path = hf_hub_download(cfg["sae_repo"], cfg["sae_file"], cache_dir=CACHE)
        state = torch.load(sae_path, map_location="cpu", weights_only=True)
        self.W_enc = state["encoder_linear.weight"].to("cuda", torch.bfloat16)  # (n, d)
        self.b_enc = state["encoder_linear.bias"].to("cuda", torch.float32)
        self.W_dec = state["decoder_linear.weight"].to("cuda", torch.bfloat16)  # (d, n)
        del state
        self.sae_layer = cfg["sae_layer"]
        self.layers = self.model.model.layers
        self.load_s = time.time() - t0
        self.vec = None  # (B, d) bf16 added at the hook layer
        self.abl = None  # (S idx, mask (B, m))
        self.capture = None
        print(f"[{key}] loaded in {self.load_s:.0f}s", flush=True)

    # ------------------------------------------------------------------ hooks
    def _steer_hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.abl is not None:
            idx, mask = self.abl
            a = (h.float() @ self.W_enc[idx].float().T + self.b_enc[idx]).relu() * mask[:, None, :]
            h = h - (a @ self.W_dec[:, idx].float().T).to(h.dtype)
        if self.vec is not None:
            h = h + self.vec[:, None, :]
        return (h, *output[1:]) if isinstance(output, tuple) else h

    def _cap_hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        self.capture = h

    # ------------------------------------------------------------------ helpers
    def _ids(self, messages: list[dict], quirk: bool, prefill: str | None = None) -> list[int]:
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        ids = list(ids)
        if quirk:
            ids += QUIRK_IDS
        if prefill:
            ids += self.tok.encode(prefill, add_special_tokens=False)
        return ids

    def _row_vec(self, job: dict):
        import torch

        d = self.W_dec.shape[0]
        v = torch.zeros(d, device="cuda", dtype=torch.float32)
        for f, val in job.get("steer") or []:
            v += float(val) * self.W_dec[:, int(f)].float()
        if job.get("rand"):
            seed, norm = job["rand"]
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            r = torch.randn(d, generator=g).to("cuda")
            v += float(norm) * r / r.norm()
        return v

    def generate(self, jobs: list[dict], out_path: str, batch: int, max_seconds: float) -> dict:
        import torch

        done = set()
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001, S110 - a half-written last line is skipped
                    pass
        todo = [j for j in jobs if j["id"] not in done]
        groups: dict[tuple, list[dict]] = {}
        for j in todo:
            k = (
                j.get("max_new", 512),
                j.get("temp", 0.6),
                j.get("rep", 1.0),
                bool(j.get("quirk", False)),
                int(j["layer"]),
                len(j.get("followups") or []),
            )
            groups.setdefault(k, []).append(j)
        t0 = time.time()
        n_written = 0
        stopped = False
        pad = self.tok.pad_token_id
        stop_ids = {128009, 128001, 128008, pad}
        with p.open("a") as fh:
            for (max_new, temp, rep, quirk, layer, n_follow), js in groups.items():
                handle = self.layers[layer].register_forward_hook(self._steer_hook)
                try:
                    js.sort(key=lambda j: len(json.dumps(j["messages"])))
                    for s in range(0, len(js), batch):
                        if time.time() - t0 > max_seconds:
                            stopped = True
                            break
                        chunk = js[s : s + batch]
                        convs = [list(j["messages"]) for j in chunk]
                        vec = torch.stack([self._row_vec(j) for j in chunk])
                        self.vec = vec.to(torch.bfloat16) if vec.abs().sum() > 0 else None
                        abl_sets = [set(j.get("ablate") or []) for j in chunk]
                        union = sorted(set().union(*abl_sets))
                        if union:
                            idx = torch.tensor(union, device="cuda")
                            mask = torch.tensor(
                                [[1.0 if u in a else 0.0 for u in union] for a in abl_sets], device="cuda"
                            )
                            self.abl = (idx, mask)
                        else:
                            self.abl = None
                        turns: list[list[str]] = [[] for _ in chunk]
                        ntok: list[list[int]] = [[] for _ in chunk]
                        for t in range(n_follow + 1):
                            ids = [
                                self._ids(c, quirk, chunk[i].get("prefill") if t == 0 else None)
                                for i, c in enumerate(convs)
                            ]
                            L = max(len(x) for x in ids)
                            inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device="cuda")
                            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
                            torch.manual_seed(int(chunk[0].get("seed", 0)) * 1000 + t)
                            mn = max_new if t == 0 else int(chunk[0].get("max_new_follow", max_new))
                            with torch.no_grad():
                                out = self.model.generate(
                                    input_ids=inp,
                                    attention_mask=att,
                                    do_sample=temp > 0,
                                    temperature=temp if temp > 0 else None,
                                    top_p=1.0,
                                    top_k=0 if temp > 0 else None,
                                    repetition_penalty=rep,
                                    max_new_tokens=mn,
                                    pad_token_id=pad,
                                )
                            gen = out[:, L:]
                            for i in range(len(chunk)):
                                g = gen[i].tolist()
                                cut = next((k for k, tkn in enumerate(g) if tkn in stop_ids), None)
                                g = g if cut is None else g[:cut]
                                txt = self.tok.decode(g, skip_special_tokens=True)
                                if t == 0 and chunk[i].get("prefill"):
                                    txt = chunk[i]["prefill"] + txt
                                turns[i].append(txt)
                                ntok[i].append(len(g))
                                if t < n_follow:
                                    convs[i] = convs[i] + [
                                        {"role": "assistant", "content": txt},
                                        {"role": "user", "content": chunk[i]["followups"][t]},
                                    ]
                        for i, j in enumerate(chunk):
                            fh.write(json.dumps({"id": j["id"], "turns": turns[i], "ntok": ntok[i]}) + "\n")
                            n_written += 1
                        fh.flush()
                        vol.commit()
                        print(f"[{self.key}] {n_written}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
                finally:
                    handle.remove()
                    self.vec = None
                    self.abl = None
                if stopped:
                    break
        return {"written": n_written, "todo": len(todo), "seconds": time.time() - t0, "stopped": stopped}

    def features(self, items: list[dict], ids: list[int] | None, layer: int | None, batch: int = 8) -> list[dict]:
        """Max activation over positions per item. `items`: {id, messages} (chat, no generation prompt; an assistant turn
        may be included) or {id, text} (raw text with BOS). With `ids`, return those latents' max and mean; without,
        return every latent whose maximum is above 0 (sparse)."""
        import torch

        layer = self.sae_layer if layer is None else layer
        handle = self.layers[layer].register_forward_hook(self._cap_hook)
        out = []
        pad = self.tok.pad_token_id
        try:
            for s in range(0, len(items), batch):
                chunk = items[s : s + batch]
                seqs = []
                for it in chunk:
                    if "messages" in it:
                        x = self.tok.apply_chat_template(it["messages"], add_generation_prompt=False, tokenize=True)
                        x = list(x if isinstance(x, list) else x["input_ids"])
                    else:
                        x = self.tok.encode(it["text"], add_special_tokens=True)
                    seqs.append(x[:2048])
                L = max(len(x) for x in seqs)
                inp = torch.tensor([[pad] * (L - len(x)) + x for x in seqs], device="cuda")
                att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in seqs], device="cuda")
                with torch.no_grad():
                    self.model(input_ids=inp, attention_mask=att)
                h = self.capture
                for i, it in enumerate(chunk):
                    hi = h[i, L - len(seqs[i]) :]
                    if ids is not None:
                        idx = torch.tensor(ids, device="cuda")
                        a = (hi.float() @ self.W_enc[idx].float().T + self.b_enc[idx]).relu()
                        out.append({"id": it["id"], "max": a.max(0).values.tolist(), "mean": a.mean(0).tolist()})
                    else:
                        m = torch.zeros(self.W_enc.shape[0], device="cuda")
                        for c in range(0, hi.shape[0], 256):
                            a = (hi[c : c + 256].float() @ self.W_enc.float().T + self.b_enc).relu()
                            m = torch.maximum(m, a.max(0).values)
                        nz = torch.nonzero(m > 0).flatten()
                        out.append({"id": it["id"], "active": nz.tolist(), "vals": m[nz].tolist()})
        finally:
            handle.remove()
        return out

    def stats(self) -> dict:
        """Decoder column norms (for the random-direction control)."""
        return {"dec_norm": self.W_dec.float().norm(dim=0).tolist()}

    def logit_tokens(self, ids: list[int], k: int = 15) -> dict:
        """Each latent's top output tokens: the unembedding applied to its decoder column (after the final norm's
        scale), used to label latents that have no public description."""
        import torch

        W_U = self.model.lm_head.weight  # (V, d)
        g = self.model.model.norm.weight
        out = {}
        for f in ids:
            logits = W_U.float() @ (self.W_dec[:, f].float() * g.float())
            top = torch.topk(logits, k).indices.tolist()
            out[str(f)] = [self.tok.decode([t]) for t in top]
        return out


@app.cls(
    gpu="B200",
    timeout=TIMEOUT["70b"],
    secrets=[hf_secret],
    volumes={VOL: vol},
    memory=65536,
    max_containers=1,
    scaledown_window=60,
)
class Engine70:
    @modal.enter()
    def load(self):
        self.e = _Engine("70b")

    @modal.method()
    def generate(self, jobs: list[dict], tag: str, batch: int = 96, max_seconds: float = 2.7 * 3600) -> dict:
        return self.e.generate(jobs, f"{VOL}/out/{tag}.jsonl", batch, max_seconds) | {"load_s": self.e.load_s}

    @modal.method()
    def features(self, items: list[dict], ids: list[int] | None = None, layer: int | None = None) -> list[dict]:
        return self.e.features(items, ids, layer)

    @modal.method()
    def stats(self) -> dict:
        return self.e.stats()

    @modal.method()
    def logit_tokens(self, ids: list[int]) -> dict:
        return self.e.logit_tokens(ids)


@app.cls(
    gpu="H100",
    timeout=TIMEOUT["8b"],
    secrets=[hf_secret],
    volumes={VOL: vol},
    memory=32768,
    max_containers=1,
    scaledown_window=60,
)
class Engine8:
    @modal.enter()
    def load(self):
        self.e = _Engine("8b")

    @modal.method()
    def generate(self, jobs: list[dict], tag: str, batch: int = 96, max_seconds: float = 1.8 * 3600) -> dict:
        return self.e.generate(jobs, f"{VOL}/out/{tag}.jsonl", batch, max_seconds) | {"load_s": self.e.load_s}

    @modal.method()
    def features(self, items: list[dict], ids: list[int] | None = None, layer: int | None = None) -> list[dict]:
        return self.e.features(items, ids, layer)

    @modal.method()
    def stats(self) -> dict:
        return self.e.stats()

    @modal.method()
    def logit_tokens(self, ids: list[int]) -> dict:
        return self.e.logit_tokens(ids)


@app.function(cpu=1, memory=4096, timeout=600, volumes={VOL: vol})
def read_out(tag: str) -> str:
    vol.reload()
    p = Path(f"{VOL}/out/{tag}.jsonl")
    return p.read_text() if p.exists() else ""
