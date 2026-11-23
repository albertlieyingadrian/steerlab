"""
Shared code for the SAE 101 tutorial: train a sparse autoencoder (SAE) on Qwen2.5-0.5B and publish it.

Every numbered script imports this file. What lives here, in the order the tutorial meets it:

    settings            the model, the layer, the data and the two sizes ("small" for a laptop, "full" for one A10G)
    tokens              FineWeb-Edu text cut into 512-token windows that each start with <|endoftext|>
    residual()          run the model up to the layer and return the residual stream there
    activation_batches  a shuffle buffer that turns windows into mixed-up batches of activations
    TopKSAE             the hand-written SAE: encoder, keep the k biggest, decoder (about 30 lines)
    train_sae()         Adam on those batches, with AuxK to revive dead latents
    metrics             FVU, L0, dead and dense latents, and the spliced-in cross-entropy test

The analogy used throughout: an activation is a smoothie, and the SAE is a recipe card that lists which few
ingredients (latents) went in and how much of each. The decoder rows are the ingredients; the encoder reads the
smoothie and writes the card.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SAE101_RESULTS", HERE / "results"))  # small JSON results: committed
RECORDED = Path(os.environ.get("SAE101_RECORDED", HERE / "recorded"))  # each script's printed output: committed
WORK = Path(os.environ.get("SAE101_WORK", HERE / "work"))  # big files (tokens, checkpoints): not committed

MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 11  # the residual stream after block 11 of 0..23: Hugging Face hidden_states[12]
HOOK_TL = f"blocks.{LAYER}.hook_resid_post"  # the same place, by its TransformerLens / SAELens / Neuronpedia name
HOOK_HF = f"model.layers.{LAYER}"  # the same place, by its Hugging Face module name
D_MODEL = 896
CTX = 512
EOT = 151643  # <|endoftext|>: Qwen has no BOS token, so each window starts with this one, and it is never trained on
SKIP = 3  # the first 3 positions of a window are never used: the model parks a huge "attention sink" on one of them
DATASET, DATASET_CONFIG = "HuggingFaceFW/fineweb-edu", "sample-10BT"
SEED = 0

# Window counts per split (each window holds 511 text tokens). The splits come from the stream in this order, so no
# document is in two splits.
SIZES = {
    "small": dict(d_sae=4096, k=32, lr=4e-4, batch=4096, eval=400, ce=100, interp=2000, probe=400, train=8_000),
    "full": dict(d_sae=16384, k=64, lr=2e-4, batch=4096, eval=2000, ce=200, interp=10_000, probe=1000, train=200_000),
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check(name: str, ok: bool) -> bool:
    """A self-check line: ✅ or ❌, then what was checked."""
    print(f"  {'✅' if ok else '❌'} {name}")
    return bool(ok)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


# --------------------------------------------------------------------------------------------------------------------
# Tokens


def windows_from_texts(texts, tokenizer, n_windows: int, ctx: int = CTX):
    """Concatenate documents (each followed by <|endoftext|>) and cut the stream into windows of `ctx` tokens.

    Every window starts with <|endoftext|>, so position 0 is never a text token. Returns (tokens [n, ctx] uint32,
    doc_ids [n, ctx] int32 with -1 on <|endoftext|>), so later steps can split by document.
    """
    tokens = np.full((n_windows, ctx), EOT, dtype=np.uint32)
    doc_ids = np.full((n_windows, ctx), -1, dtype=np.int32)
    buf_tok: list[int] = []
    buf_doc: list[int] = []
    filled = 0
    for doc_index, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        buf_tok += ids + [EOT]
        buf_doc += [doc_index] * len(ids) + [-1]
        while len(buf_tok) >= ctx - 1 and filled < n_windows:
            tokens[filled, 1:] = buf_tok[: ctx - 1]
            doc_ids[filled, 1:] = buf_doc[: ctx - 1]
            del buf_tok[: ctx - 1], buf_doc[: ctx - 1]
            filled += 1
        if filled == n_windows:
            return tokens, doc_ids
    raise ValueError(f"ran out of text after {filled} of {n_windows} windows")


def fineweb_texts():
    """FineWeb-Edu documents in a fixed order (streamed: nothing is downloaded in full)."""
    from datasets import load_dataset

    for row in load_dataset(DATASET, name=DATASET_CONFIG, split="train", streaming=True):
        yield row["text"]


def python_files() -> list[Path]:
    """The Python standard library's own source files, sorted: free, licensed (PSF) code text for step 6."""
    import sysconfig

    root = Path(sysconfig.get_paths()["stdlib"])
    skip = {"test", "tests", "idlelib", "site-packages", "lib2to3", "turtledemo", "__pycache__", "ensurepip"}
    return sorted(p for p in root.rglob("*.py") if not skip & set(p.relative_to(root).parts))


def n_text_tokens(windows) -> int:
    """How many positions text_mask() keeps in these windows (sets the number of training steps)."""
    total = 0
    for s in range(0, len(windows), 4096):
        w = np.asarray(windows[s : s + 4096])
        total += int((w[:, SKIP:] != EOT).sum())
    return total


def load_tokens(size: str, split: str):
    path = WORK / size / f"tokens_{split}.npy"
    if not path.exists():
        raise SystemExit(f"{path.name} is missing: run 02_collect.py --size {size} first.")
    return np.load(path, mmap_mode="r")


def text_mask(tokens: torch.Tensor) -> torch.Tensor:
    """True where an activation is used: not <|endoftext|>, and not in the first SKIP positions of a window.

    Qwen2.5 puts an activation about 80x bigger than usual on the first token of a window (an "attention sink"), and
    in about 1 window in 12 on position 1 or 2 instead. A few such tokens would dominate the squared error, so they are
    skipped. SAELens does the same with seqpos_slice=(3, None) and exclude_special_tokens=[EOT].
    """
    pos = torch.arange(tokens.shape[-1], device=tokens.device)
    return (tokens != EOT) & (pos >= SKIP)


# --------------------------------------------------------------------------------------------------------------------
# The model and its residual stream


def load_lm(name: str = MODEL, device: str | None = None):
    """The model in float32 (TF32 matmuls on a GPU), in eval mode, with its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    return model, AutoTokenizer.from_pretrained(name), device


class _Stop(Exception):
    """Raised inside a hook to stop the forward pass once we have the layer we want (saves half the compute)."""


@torch.no_grad()
def residual(model, tokens: torch.Tensor, layer: int = LAYER) -> torch.Tensor:
    """The residual stream after block `layer` for a batch of token ids: [batch, ctx, d_model]."""
    box = {}

    def grab(module, inputs, output):
        box["h"] = output[0] if isinstance(output, tuple) else output
        raise _Stop

    handle = model.model.layers[layer].register_forward_hook(grab)
    try:
        model(tokens)
    except _Stop:
        pass
    finally:
        handle.remove()
    return box["h"]


def activation_batches(
    model,
    windows,
    *,
    batch: int,
    scale: float,
    device: str,
    seed: int = SEED,
    start: int = 0,
    fwd: int = 32,
    buffer_tokens: int = 2**19,
):
    """Yield (batch of normalised activations [batch, d_model], windows read so far), shuffled across windows.

    Why shuffle: neighbouring tokens in one document look alike, and an SAE trained on them in order sees a long run of
    one topic, then another. The buffer reads many windows, mixes all their tokens, hands out half, and refills. Every
    position text_mask() rejects is dropped here (the attention sinks and <|endoftext|>).
    """
    order = np.random.default_rng(seed).permutation(len(windows))
    gen = torch.Generator().manual_seed(seed + start)
    store = torch.empty(0, model.config.hidden_size, device=device)
    i = start
    while True:
        parts = [store]
        have = len(store)
        while have < buffer_tokens and i < len(order):
            idx = np.sort(order[i : i + fwd])
            toks = torch.as_tensor(np.asarray(windows[idx], dtype=np.int64), device=device)
            i += len(idx)
            h = residual(model, toks)[text_mask(toks)] * scale
            parts.append(h)
            have += len(h)
        store = torch.cat(parts)
        if len(store) < batch:
            return
        store = store[torch.randperm(len(store), generator=gen).to(device)]
        keep = len(store) // 2 if i < len(order) else 0
        n_out = (len(store) - keep) // batch * batch
        for s in range(0, n_out, batch):
            yield store[s : s + batch], i
        store = store[n_out:]


@torch.no_grad()
def norm_scale(model, windows, device: str, n_tokens: int = 1_000_000) -> float:
    """The one number that rescales activations so their mean squared norm is d_model (SAELens's
    "expected_average_only_in"). Learning rates then mean the same thing at any layer of any model."""
    total, count = 0.0, 0
    for s in range(0, len(windows), 32):
        toks = torch.as_tensor(np.asarray(windows[s : s + 32], dtype=np.int64), device=device)
        h = residual(model, toks)[text_mask(toks)]
        total += float((h**2).sum(-1).sum().cpu().double())
        count += len(h)
        if count >= n_tokens:
            break
    return math.sqrt(model.config.hidden_size / (total / count))


# --------------------------------------------------------------------------------------------------------------------
# The hand-written SAE


class TopKSAE(nn.Module):
    """A TopK sparse autoencoder.

        pre = (x - b_dec) @ W_enc + b_enc       a score for every latent (d_sae of them)
        z   = keep the k biggest scores, ReLU them, zero the rest
        x̂   = z @ W_dec + b_dec                 the rebuild: k decoder rows, weighted, plus a bias

    Each decoder row is kept at length 1, so a latent's size lives in z, not in its direction. b_dec is the "average
    activation": subtracted before encoding and added back after decoding, so the latents only describe what differs.
    """

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        W_dec = torch.randn(d_sae, d_in)
        W_dec /= W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())  # each encoder column starts on its own decoder row
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        pre = self.pre(x)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)


def aux_k_loss(sae: TopKSAE, x, x_hat, pre, dead: torch.Tensor, k_aux: int):
    """AuxK (Gao et al. 2024): ask the dead latents to rebuild what the live ones missed, so they get a gradient again.

    Without this, a latent that never makes the top k never gets a gradient and stays dead forever.
    """
    n_dead = int(dead.sum())
    if n_dead == 0:
        return x.new_zeros(())
    err = (x - x_hat).detach()
    masked = pre.masked_fill(~dead, -float("inf"))
    vals, idx = masked.topk(min(k_aux, n_dead), dim=-1)
    z_aux = torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))
    err_hat = z_aux @ sae.W_dec
    return ((err_hat - err) ** 2).sum(-1).mean() / (err - err.mean(0)).pow(2).sum(-1).mean().clamp_min(1e-8)


def train_sae(
    sae: TopKSAE,
    batches,
    *,
    total_steps: int,
    lr: float,
    aux_coef: float = 1 / 32,
    dead_after_steps: int = 1000,
    log_every: int = 100,
    on_log=None,
    on_checkpoint=None,
    start_step: int = 0,
    opt_state=None,
    last_fired=None,
    batch_topk: bool = False,
):
    """Adam on the reconstruction error, one batch at a time. The learning rate falls linearly to 0 over the last 20%.

    batch_topk=True keeps the k x batch biggest scores across the whole batch instead of k per token (BatchTopK): busy
    tokens may use more latents and simple ones fewer, with the same average. Returns the log (a list of dicts).
    `on_checkpoint(step, opt, last_fired, windows_read)` is called every `log_every * 8` steps so a long run can resume.
    """
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    decay_from = int(total_steps * 0.8)
    device = sae.W_dec.device
    last_fired = last_fired if last_fired is not None else torch.full((sae.d_sae,), start_step, device=device)
    k_aux = sae.d_in // 2
    log, step, t0, windows_read = [], start_step, time.monotonic(), 0
    acc = {"mse": 0.0, "fvu": 0.0, "aux": 0.0, "l0": 0.0, "n": 0}
    for x, windows_read in batches:
        if step >= total_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (total_steps - step) / max(1, total_steps - decay_from))
        pre = sae.pre(x)
        if batch_topk:
            flat = torch.relu(pre).flatten()
            vals, idx = flat.topk(sae.k * len(x))
            z = torch.zeros_like(flat).scatter(0, idx, vals).view_as(pre)
        else:
            vals, idx = pre.topk(sae.k, dim=-1)
            z = torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))
        x_hat = sae.decode(z)
        mse = ((x_hat - x) ** 2).sum(-1).mean()
        dead = (step - last_fired) > dead_after_steps
        aux = aux_k_loss(sae, x, x_hat, pre, dead, k_aux) if aux_coef > 0 else x.new_zeros(())
        loss = mse + aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalise_decoder()
        last_fired[(z > 0).any(0)] = step
        step += 1
        acc["mse"] += float(mse.detach())
        acc["fvu"] += float(mse.detach() / (x - x.mean(0)).pow(2).sum(-1).mean())
        acc["aux"] += float(aux.detach())
        acc["l0"] += float((z > 0).sum(-1).float().mean())
        acc["n"] += 1
        if step % log_every == 0 or step == total_steps:
            n = acc["n"]
            row = {
                "step": step,
                "tokens": step * len(x),
                "train_fvu": acc["fvu"] / n,
                "train_mse": acc["mse"] / n,
                "aux": acc["aux"] / n,
                "l0": acc["l0"] / n,
                "dead_share": float(((step - last_fired) > dead_after_steps).float().mean()),
                "lr": opt.param_groups[0]["lr"],
                "seconds": round(time.monotonic() - t0, 1),
            }
            log.append(row)
            acc = {"mse": 0.0, "fvu": 0.0, "aux": 0.0, "l0": 0.0, "n": 0}
            if on_log:
                on_log(row)
        if on_checkpoint and step % (log_every * 8) == 0:
            on_checkpoint(step, opt, last_fired, windows_read)
    return log


def save_sae(sae: TopKSAE, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"d_in": sae.d_in, "d_sae": sae.d_sae, "k": sae.k, "state_dict": sae.state_dict(), **(extra or {})}, path
    )


def load_sae(path: Path, device: str = "cpu") -> tuple[TopKSAE, dict]:
    if not Path(path).exists():
        raise SystemExit(f"{Path(path).name} is missing: run the training step first.")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(saved["d_in"], saved["d_sae"], saved["k"])
    sae.load_state_dict(saved["state_dict"])
    return sae.to(device).eval(), saved


# --------------------------------------------------------------------------------------------------------------------
# Metrics


@torch.no_grad()
def reconstruction_stats(encode_decode, model, windows, *, scale: float, device: str, d_sae: int, fwd: int = 16):
    """One pass over held-out windows: FVU, mean L0, and how often each latent fires.

    encode_decode(x_normalised) -> (x_hat_normalised, z). FVU is the same on raw or normalised activations (one scale
    multiplies both the error and the spread), so it is computed on the normalised ones.
    """
    err = sum_sq = l0_total = 0.0
    sum_x = None
    n = 0
    fires = torch.zeros(d_sae, dtype=torch.long, device=device)
    for s in range(0, len(windows), fwd):
        toks = torch.as_tensor(np.asarray(windows[s : s + fwd], dtype=np.int64), device=device)
        x = residual(model, toks)[text_mask(toks)] * scale
        x_hat, z = encode_decode(x)
        err += float(((x_hat - x) ** 2).sum())
        sum_x = x.sum(0).cpu().double() if sum_x is None else sum_x + x.sum(0).cpu().double()
        sum_sq += float((x**2).sum())
        n += len(x)
        active = z > 0
        l0_total += float(active.sum())
        fires += active.sum(0)
    tot = sum_sq - float((sum_x**2).sum()) / n  # Σ|x - mean|², without a second pass
    return {"fvu": err / tot, "l0": l0_total / n, "tokens": n, "freq": (fires.cpu().double() / n).numpy()}


@torch.no_grad()
def spliced_losses(model, windows, *, replace=None, device: str, fwd: int = 2):
    """Mean next-token cross-entropy (nats) with the layer's output passed through `replace(h, keep) -> h'`.

    `keep` = text_mask(tokens); other positions are left alone. Targets that are <|endoftext|> or in the first SKIP
    positions are not scored. Returns the mean, the per-window means (for a bootstrap) and, when `replace` is given, the mean KL from the
    clean model's next-token distribution.
    """
    per_window, kls = [], []
    for s in range(0, len(windows), fwd):
        toks = torch.as_tensor(np.asarray(windows[s : s + fwd], dtype=np.int64), device=device)
        keep = text_mask(toks)
        clean_logits = model(toks).logits if replace is not None else None

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h2 = replace(h, keep)
            return (h2, *output[1:]) if isinstance(output, tuple) else h2

        handle = model.model.layers[LAYER].register_forward_hook(hook) if replace is not None else None
        try:
            logits = model(toks).logits
        finally:
            if handle:
                handle.remove()
        logp = torch.log_softmax(logits[:, :-1].float(), -1)
        target = toks[:, 1:]
        scored = text_mask(toks)[:, 1:].float()
        nll = -logp.gather(-1, target[..., None])[..., 0]
        per_window += ((nll * scored).sum(1) / scored.sum(1)).tolist()
        if clean_logits is not None:
            clean_logp = torch.log_softmax(clean_logits[:, :-1].float(), -1)
            kl = (clean_logp.exp() * (clean_logp - logp)).sum(-1)
            kls += ((kl * scored).sum(1) / scored.sum(1)).tolist()
    return {"ce": float(np.mean(per_window)), "per_window": per_window, "kl": float(np.mean(kls)) if kls else 0.0}


def bootstrap_mean(values, n: int = 1000, seed: int = SEED):
    """95% interval for a mean, by resampling the values (windows or documents) with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# --------------------------------------------------------------------------------------------------------------------
# The SAELens file format (cfg.json + sae_weights.safetensors), written and read without needing SAELens installed


class RawSAE:
    """Any SAE seen from the outside: encode(raw activations) -> latents, decode(latents) -> raw activations.

    Every metric in step 4 goes through this, so the hand-written SAE, the SAELens-trained one and the copy downloaded
    from the Hub are measured by the same code.
    """

    def __init__(self, encode, decode, d_sae: int, k: int, W_dec: torch.Tensor, name: str):
        self.encode, self.decode, self.d_sae, self.k, self.W_dec, self.name = encode, decode, d_sae, k, W_dec, name

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z), z


def raw_from_handwritten(path: Path, device: str) -> RawSAE:
    sae, saved = load_sae(path, device)
    scale = saved["scale"]
    return RawSAE(
        lambda x: sae.encode(x * scale),
        lambda z: sae.decode(z) / scale,
        sae.d_sae,
        sae.k,
        (sae.W_dec / scale).detach(),
        "hand-written",
    )


def export_saelens(path: Path, out_dir: Path, metadata: dict) -> Path:
    """Write the hand-written SAE in SAELens's format, with the normalisation scale folded into the weights.

    The SAE was trained on x * scale. Folding: W_enc *= scale, W_dec /= scale, b_dec /= scale gives the same latents
    and a rebuild in raw units, so whoever loads the file can feed it raw activations and no scale needs shipping.
    """
    from safetensors.torch import save_file

    sae, saved = load_sae(path, "cpu")
    scale = saved["scale"]
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "W_enc": sae.W_enc.data * scale,
        "W_dec": sae.W_dec.data / scale,
        "b_enc": sae.b_enc.data.clone(),
        "b_dec": sae.b_dec.data / scale,
    }
    save_file({k: v.contiguous().float() for k, v in weights.items()}, str(out_dir / "sae_weights.safetensors"))
    cfg = {
        "architecture": "topk",
        "d_in": sae.d_in,
        "d_sae": sae.d_sae,
        "k": sae.k,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "rescale_acts_by_decoder_norm": False,
        "metadata": metadata,
    }
    (out_dir / "cfg.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return out_dir


def raw_from_saelens_dir(folder: Path, device: str, name: str = "SAELens format") -> RawSAE:
    """Read a TopK SAE in SAELens's format with plain torch, following SAELens's TopKSAE.encode/decode exactly."""
    from safetensors.torch import load_file

    cfg = json.loads((Path(folder) / "cfg.json").read_text())
    if cfg["architecture"] != "topk" or cfg.get("normalize_activations", "none") != "none":
        raise ValueError(f"this reader handles TopK SAEs with folded normalisation, not {cfg['architecture']}")
    w = {k: v.to(device) for k, v in load_file(str(Path(folder) / "sae_weights.safetensors")).items()}
    k = cfg["k"]
    rescale = cfg.get("rescale_acts_by_decoder_norm", False)
    dec_norm = w["W_dec"].norm(dim=-1)
    sub = w["b_dec"] if cfg.get("apply_b_dec_to_input", True) else 0.0

    def encode(x):
        pre = (x - sub) @ w["W_enc"] + w["b_enc"]
        if rescale:
            pre = pre * dec_norm
        vals, idx = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))

    def decode(z):
        return (z / dec_norm if rescale else z) @ w["W_dec"] + w["b_dec"]

    W_dec = w["W_dec"] / dec_norm[:, None] if rescale else w["W_dec"]
    return RawSAE(encode, decode, cfg["d_sae"], k, W_dec, name)


# --------------------------------------------------------------------------------------------------------------------
# Step 4's evaluation, in one function (step 7 reruns it on the copy downloaded from the Hub)


@torch.no_grad()
def collect_acts(model, windows, device: str, fwd: int = 16) -> torch.Tensor:
    """Raw residual-stream activations at every text position of `windows`, stacked on the CPU: [n_tokens, d_model]."""
    out = []
    for s in range(0, len(windows), fwd):
        toks = torch.as_tensor(np.asarray(windows[s : s + fwd], dtype=np.int64), device=device)
        out.append(residual(model, toks)[text_mask(toks)].cpu())
    return torch.cat(out)


@torch.no_grad()
def fvu_of(raw: RawSAE, acts: torch.Tensor, batch: int = 8192):
    """FVU = Σ|x - x̂|² / Σ|x - mean|², plus mean L0, over a stack of activations (on any device)."""
    device = raw.W_dec.device
    mean = acts.mean(0).to(device)
    err = tot = l0 = 0.0
    for s in range(0, len(acts), batch):
        x = acts[s : s + batch].to(device).float()
        x_hat, z = raw(x)
        err += float(((x_hat - x) ** 2).sum())
        tot += float(((x - mean) ** 2).sum())
        l0 += float((z > 0).sum())
    return err / tot, l0 / len(acts)


@torch.no_grad()
def evaluate(
    raw: RawSAE,
    model,
    *,
    eval_windows,
    ce_windows,
    interp_windows,
    mean_act,
    device: str,
    eval_acts: torch.Tensor | None = None,
    controls: bool = True,
    seed: int = SEED,
) -> dict:
    """Every number step 4 reports for one SAE. Activations are raw (the SAE handles any scaling itself)."""
    acts = eval_acts if eval_acts is not None else collect_acts(model, eval_windows, device)
    fvu, l0 = fvu_of(raw, acts)
    r = {"name": raw.name, "d_sae": raw.d_sae, "k": raw.k, "fvu": fvu, "l0": l0, "eval_tokens": len(acts)}

    # how often each latent fires, over the interp windows (dead = never, dense = more than 1 token in 10)
    stats = reconstruction_stats(raw, model, interp_windows, scale=1.0, device=device, d_sae=raw.d_sae)
    freq = stats["freq"]
    r.update(
        {
            "freq_tokens": stats["tokens"],
            "fvu_interp": stats["fvu"],
            "dead": int((freq == 0).sum()),
            "dead_share": float((freq == 0).mean()),
            "dense": int((freq > 0.1).sum()),
            "dense_share": float((freq > 0.1).mean()),
            "log10_freq_hist": np.histogram(np.log10(freq[freq > 0]), bins=np.arange(-7, 0.01, 0.25))[0].tolist(),
        }
    )

    # does the model still work? cross-entropy with the layer replaced
    def sae_replace(h, keep):
        h2 = h.clone()
        h2[keep] = raw(h[keep])[0]
        return h2

    def sae_plus_error(h, keep):  # rebuild + (original - rebuild): must give back the clean model
        h2 = h.clone()
        x_hat = raw(h[keep])[0]
        h2[keep] = x_hat + (h[keep] - x_hat)
        return h2

    mean_vec = mean_act.to(device)

    def mean_ablate(h, keep):
        h2 = h.clone()
        h2[keep] = mean_vec
        return h2

    def zero_ablate(h, keep):
        h2 = h.clone()
        h2[keep] = 0.0
        return h2

    gen = torch.Generator(device="cpu").manual_seed(seed)

    def random_same_norm(h, keep):  # clean activation + a random push as long as the SAE's error, token by token
        h2 = h.clone()
        x = h[keep]
        err_norm = (raw(x)[0] - x).norm(dim=-1, keepdim=True)
        u = torch.randn(x.shape, generator=gen).to(device)
        h2[keep] = x + err_norm * u / u.norm(dim=-1, keepdim=True)
        return h2

    runs = {
        "clean": None,
        "sae": sae_replace,
        "sae_plus_error": sae_plus_error,
        "mean_ablation": mean_ablate,
        "zero_ablation": zero_ablate,
    }
    if controls:
        runs["random_same_norm"] = random_same_norm
    ce = {name: spliced_losses(model, ce_windows, replace=fn, device=device) for name, fn in runs.items()}
    clean = np.array(ce["clean"]["per_window"])
    r["ce"] = {name: v["ce"] for name, v in ce.items()}
    r["kl"] = {name: v["kl"] for name, v in ce.items() if name != "clean"}
    r["ce_windows"] = len(clean)
    d_sae = np.array(ce["sae"]["per_window"]) - clean
    d_mean = np.array(ce["mean_ablation"]["per_window"]) - clean
    d_zero = np.array(ce["zero_ablation"]["per_window"]) - clean
    r["delta_ce_sae"] = float(d_sae.mean())
    r["delta_ce_sae_ci"] = bootstrap_mean(d_sae)
    r["ce_recovered_vs_mean"] = float(1 - d_sae.mean() / d_mean.mean())
    r["ce_recovered_vs_zero"] = float(1 - d_sae.mean() / d_zero.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clean), size=(1000, len(clean)))
    rec = 1 - d_sae[idx].mean(1) / d_mean[idx].mean(1)
    r["ce_recovered_vs_mean_ci"] = [float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))]
    r["error_term_rel_diff"] = abs(ce["sae_plus_error"]["ce"] - ce["clean"]["ce"]) / ce["clean"]["ce"]
    if controls:
        d_rand = np.array(ce["random_same_norm"]["per_window"]) - clean
        r["delta_ce_random"] = float(d_rand.mean())
        r["delta_ce_random_ci"] = bootstrap_mean(d_rand)
        ratio = d_sae[idx].mean(1) / d_rand[idx].mean(1)
        r["sae_vs_random_ratio"] = float(d_sae.mean() / d_rand.mean())
        r["sae_vs_random_ratio_ci"] = [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]
    r["_freq"] = freq  # kept out of the JSON by the caller
    return r


@torch.no_grad()
def pca_fvu(fit_acts: torch.Tensor, eval_acts: torch.Tensor, ks) -> dict:
    """The cheap baseline: rebuild each activation from its top-k principal components (dense, k numbers per token)."""
    mean = fit_acts.float().mean(0)
    xc = (fit_acts.float() - mean).cpu().double()
    cov = xc.T @ xc / len(xc)
    _, vecs = torch.linalg.eigh(cov)
    vecs = vecs.flip(1).float().to("cuda" if torch.cuda.is_available() else "cpu")
    out = {int(k): 0.0 for k in ks}
    tot = 0.0
    for s in range(0, len(eval_acts), 65536):
        e = eval_acts[s : s + 65536].to(vecs.device).float() - mean.to(vecs.device)
        tot += float((e**2).sum())
        for k in ks:
            p = vecs[:, :k]
            out[int(k)] += float(((e - (e @ p) @ p.T) ** 2).sum())
    return {k: v / tot for k, v in out.items()}
