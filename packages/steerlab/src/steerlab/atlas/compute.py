"""Compute a unit atlas for a model the catalog doesn't cover: activations over a small corpus, exact token drivers,
and reads from the weights. Works on Qwen2/Qwen2.5/Qwen3, Llama-3.x, Mistral and Gemma 2/3 text decoders.

What "a neuron" and "a pole" are. A SwiGLU MLP computes `a = act(x·g) · (x·u)` per neuron (the input of `down_proj`)
and writes `a · W_down[:, n]`. `a` takes both signs, and the two signs do different things, so each neuron is two
units: the + pole (where `a > 0`) and the − pole (where `a < 0`, read as `−a`). Everything below is per pole.

The three computed reads, and how far to trust them:

1. **Activations** (`corpus_stats`): one pass over the corpus with a hook on every `down_proj`. A unit's firing in a
   document is its strongest position there (position 0, the attention sink, left out, as in the catalog); its top
   firings are the `k` documents where that is largest. Quantiles come from exact counts on a quarter-octave grid, so
   any activation can be turned into a percentile of the unit's own distribution. Heads: the norm of the head's output
   as it reaches the residual (`‖W_O^h z_h‖`), plus where it attends at each top firing.
2. **Exact token drivers** (`token_drivers`): put every candidate token at one position of a few real contexts and
   measure the pole's activation there, averaged over the contexts. Exact for those contexts; a different context can
   change the ranking.
3. **Weights** (`write_logits`, `read_activations`, `head_reads`, `unit_edges`): the write through the logit lens (W_U
   and the final norm's weight: exact at the last layer, rough before the last third), a z-scored version against the
   layer's poles, the read direction applied to each token's normalized embedding (exact at L0 only), head OV and QK
   maps over candidate tokens (position-free: RoPE at offset 0), and cosines of writes with later read directions.

torch is imported at module import; the package `steerlab.atlas` itself does not import this module.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

NB = 112  # histogram bins: bin b >= 1 covers [2^((b-OFFSET)/4), 2^((b-OFFSET+1)/4)); bin 0 holds everything below
OFFSET = 64  # 2^(-64/4) = 2^-16 is the smallest edge; the top edge is 2^((NB-OFFSET)/4) = 2^12
HEAD_START = 16  # heads: positions before this are left out of statistics (a query with few keys attends sharply)
TINY = 2.0 ** (-OFFSET / 4)


class _Stop(Exception):
    """Raised by a hook to end a forward pass once the last needed layer is captured."""


def bin_edges(device: Any = None) -> torch.Tensor:
    """The lower edge of every histogram bin (bin 0's is 0)."""
    edges = 2.0 ** ((torch.arange(NB, dtype=torch.float64) - OFFSET) / 4)
    edges[0] = 0.0
    return edges.float().to(device)


def _bins(v: torch.Tensor) -> torch.Tensor:
    b = torch.floor(4 * torch.log2(v.clamp_min(TINY).float())).long() + OFFSET
    return torch.where(v > TINY, b.clamp(1, NB - 1), torch.zeros_like(b))


class Parts:
    """The pieces of a decoder that the atlas reads, found by name (Qwen/Llama/Mistral/Gemma text layouts)."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        inner = model.model
        if hasattr(inner, "language_model"):  # multimodal wrappers: use the text model
            inner = inner.language_model
        config = getattr(model.config, "text_config", None) or model.config
        self.config = config
        self.inner = inner
        self.blocks = inner.layers
        self.n_layers = len(self.blocks)
        self.d_model = int(config.hidden_size)
        self.n_heads = int(config.num_attention_heads)
        self.n_kv = int(getattr(config, "num_key_value_heads", None) or self.n_heads)
        self.d_head = int(getattr(config, "head_dim", None) or self.d_model // self.n_heads)
        self.d_mlp = int(config.intermediate_size)
        self.gemma = "gemma" in str(getattr(config, "model_type", "")).lower()
        self.W_E = inner.embed_tokens.weight.detach()
        self.W_U = model.lm_head.weight.detach() if hasattr(model, "lm_head") else self.W_E
        self.vocab = int(self.W_U.shape[0])
        self.device = self.W_E.device

    # Norms

    def norm_weight(self, module: Any) -> torch.Tensor:
        weight = module.weight.detach().float()
        return weight + 1.0 if self.gemma else weight  # type: ignore[no-any-return,unused-ignore]

    def mlp_norm(self, layer: int) -> torch.Tensor:
        block = self.blocks[layer]
        norm = getattr(block, "pre_feedforward_layernorm", None) or block.post_attention_layernorm
        return self.norm_weight(norm)

    def attn_norm(self, layer: int) -> torch.Tensor:
        return self.norm_weight(self.blocks[layer].input_layernorm)

    def final_norm(self) -> torch.Tensor:
        return self.norm_weight(self.inner.norm)

    # MLP weights

    def gate(self, layer: int) -> torch.Tensor:
        return self.blocks[layer].mlp.gate_proj.weight.detach().float()  # type: ignore[no-any-return]

    def up(self, layer: int) -> torch.Tensor:
        return self.blocks[layer].mlp.up_proj.weight.detach().float()  # type: ignore[no-any-return]

    def down(self, layer: int) -> torch.Tensor:
        """`[d_model, d_mlp]`: column n is neuron n's write (Gemma: times the post-feedforward norm's weight)."""
        w: torch.Tensor = self.blocks[layer].mlp.down_proj.weight.detach().float()
        post = getattr(self.blocks[layer], "post_feedforward_layernorm", None)
        return w * self.norm_weight(post)[:, None] if post is not None else w

    def act_fn(self, layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.blocks[layer].mlp.act_fn  # type: ignore[no-any-return]

    def normed_embeddings(self, ids: torch.Tensor | None, weight: torch.Tensor) -> torch.Tensor:
        """Token embeddings as a layer's norm would hand them on: RMS-normalized, times the norm's weight."""
        rows = (self.W_E if ids is None else self.W_E[ids]).float()
        return rows / rows.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * weight  # type: ignore[no-any-return,unused-ignore]

    def decode(self, token_id: int) -> str:
        return str(self.tokenizer.decode([int(token_id)]))


def _capture(
    parts: Parts, layers: Sequence[int], heads: bool = False, stop_after: int | None = None
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[Any]]:
    """Hooks storing each layer's neuron activations (and, with `heads`, the per-head inputs of `o_proj`)."""
    acts: dict[int, torch.Tensor] = {}
    zs: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:

        def pre(module: Any, args: tuple[Any, ...], layer: int = layer) -> None:
            acts[layer] = args[0].detach().float()
            if stop_after is not None and layer == stop_after and not heads:
                raise _Stop

        handles.append(parts.blocks[layer].mlp.down_proj.register_forward_pre_hook(pre))
        if heads:

            def zpre(module: Any, args: tuple[Any, ...], layer: int = layer) -> None:
                zs[layer] = args[0].detach().float()

            handles.append(parts.blocks[layer].self_attn.o_proj.register_forward_pre_hook(zpre))
    return acts, zs, handles


def _merge_top(kept: tuple[torch.Tensor, ...], new: tuple[torch.Tensor, ...], k: int) -> tuple[torch.Tensor, ...]:
    """Merge new candidates (rows) into the kept top-k (rows), ranked by the first tensor; the rest follow."""
    values = torch.cat([kept[0], new[0]], 0)
    top, idx = values.topk(k, dim=0)
    return (top, *(torch.cat([a, b], 0).gather(0, idx) for a, b in zip(kept[1:], new[1:], strict=True)))


def quantiles(hist: torch.Tensor, total: int, qs: Sequence[float]) -> list[torch.Tensor]:
    """Quantiles of each row of `hist` (counts per bin over `total` positions), geometric within a bin; 0 in bin 0."""
    cum = hist.double().cumsum(-1)
    lows = bin_edges(hist.device).double()
    out = []
    for q in qs:
        target = q * total
        b = (cum < target).sum(-1).clamp(max=NB - 1)
        below = torch.where(b > 0, cum.gather(-1, (b - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1), 0.0)
        inside = hist.gather(-1, b.unsqueeze(-1)).squeeze(-1).double().clamp(min=1)
        frac = ((target - below) / inside).clamp(0, 1)
        value = lows[b] * (2.0 ** (frac / 4))
        out.append(torch.where(b > 0, value, 0.0).float())
    return out


@torch.no_grad()
def corpus_stats(
    parts: Parts,
    docs: Sequence[Sequence[int]],
    k: int = 16,
    batch_size: int = 8,
    heads: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Top firings and quantiles of every pole (and head) over `docs` (token id lists). Returns CPU tensors:

    neurons `top_val/top_doc/top_pos` `[L, 2, k, d_mlp]` (pole axis: 0 is +, 1 is −), `q50/q90/q99/q999/act_top/act_max/
    rate` `[L, 2, d_mlp]`; heads `head_top_val/head_top_doc/head_top_pos/head_src_pos` `[L, k, H]`, `head_q*`,
    `head_out_top/head_out_max` `[L, H]`, `head_attn` `[L, 3, H]` (mean attention to the first token, the previous
    token, itself); and `n_positions`.
    """
    L, d, H = parts.n_layers, parts.d_mlp, parts.n_heads
    dev = parts.device
    k = max(1, min(k, len(docs)))
    top = (
        torch.full((L, 2, k, d), -math.inf, device=dev),
        torch.zeros((L, 2, k, d), dtype=torch.long, device=dev),
        torch.zeros((L, 2, k, d), dtype=torch.long, device=dev),
    )
    hist = torch.zeros((L, 2, d, NB), dtype=torch.long, device=dev)
    h_top = (
        torch.full((L, k, H), -math.inf, device=dev),
        torch.zeros((L, k, H), dtype=torch.long, device=dev),
        torch.zeros((L, k, H), dtype=torch.long, device=dev),
        torch.zeros((L, k, H), dtype=torch.long, device=dev),
    )
    h_hist = torch.zeros((L, H, NB), dtype=torch.long, device=dev)
    attn_sums = torch.zeros((L, 3, H), dtype=torch.float64, device=dev)
    n_positions = h_positions = 0
    pad = getattr(parts.tokenizer, "pad_token_id", None) or 0
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    acts, zs, handles = _capture(parts, range(L), heads=heads)
    try:
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            T = max(len(docs[i]) for i in batch)
            ids = torch.full((len(batch), T), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), T), dtype=torch.long)
            for row, i in enumerate(batch):
                ids[row, : len(docs[i])] = torch.tensor(list(docs[i]))
                mask[row, : len(docs[i])] = 1
            ids, mask = ids.to(dev), mask.to(dev)
            out = parts.model(input_ids=ids, attention_mask=mask, use_cache=False, output_attentions=heads)
            valid = mask.bool().clone()
            valid[:, 0] = False
            doc_ids = torch.tensor(batch, device=dev)
            n_positions += int(valid.sum())
            neuron_offsets = torch.arange(d, device=dev) * NB
            for layer in range(L):
                a = acts[layer]
                for pi, sign in enumerate((1.0, -1.0)):
                    v = a * sign
                    mx, arg = v.masked_fill(~valid[..., None], -math.inf).max(dim=1)  # [B, d]
                    kept = (top[0][layer, pi], top[1][layer, pi], top[2][layer, pi])
                    merged = _merge_top(kept, (mx, doc_ids[:, None].expand_as(arg), arg), k)
                    for t, m in zip(top, merged, strict=True):
                        t[layer, pi] = m
                    flat = (_bins(v[valid]) + neuron_offsets).reshape(-1)
                    hist[layer, pi] += torch.bincount(flat, minlength=d * NB).view(d, NB)
            if heads:
                h_positions += _head_batch(parts, out, zs, valid, doc_ids, h_top, h_hist, attn_sums, k)
            if progress:
                progress(min(start + batch_size, len(order)), len(order))
    finally:
        for handle in handles:
            handle.remove()
    q50, q90, q99, q999 = quantiles(hist, n_positions, (0.5, 0.9, 0.99, 0.999))
    top_val = top[0].clamp(min=0)
    act_max = top_val[:, :, 0]
    act_top = top_val[:, :, k // 2] if k > 1 else act_max
    above = bin_edges(dev)[None, None, None, :] >= (act_top / 4).unsqueeze(-1)
    rate = ((hist * above).sum(-1).double() / max(1, n_positions)).float()
    result: dict[str, Any] = {
        "n_positions": n_positions,
        "n_docs": len(docs),
        "k": k,
        "top_val": top_val.cpu(),
        "top_doc": top[1].cpu(),
        "top_pos": top[2].cpu(),
        "q50": q50.cpu(),
        "q90": q90.cpu(),
        "q99": q99.cpu(),
        "q999": q999.cpu(),
        "act_top": act_top.cpu(),
        "act_max": act_max.cpu(),
        "rate": rate.cpu(),
    }
    if heads:
        hq = quantiles(h_hist, h_positions, (0.5, 0.9, 0.99, 0.999))
        h_val = h_top[0].clamp(min=0)
        result.update(
            head_top_val=h_val.cpu(),
            head_top_doc=h_top[1].cpu(),
            head_top_pos=h_top[2].cpu(),
            head_src_pos=h_top[3].cpu(),
            head_q50=hq[0].cpu(),
            head_q90=hq[1].cpu(),
            head_q99=hq[2].cpu(),
            head_q999=hq[3].cpu(),
            head_out_top=(h_val[:, k // 2] if k > 1 else h_val[:, 0]).cpu(),
            head_out_max=h_val[:, 0].cpu(),
            head_attn=(attn_sums / max(1, h_positions)).float().cpu(),
            head_positions=h_positions,
        )
    return result


def _head_batch(
    parts: Parts,
    out: Any,
    zs: dict[int, torch.Tensor],
    valid: torch.Tensor,
    doc_ids: torch.Tensor,
    h_top: tuple[torch.Tensor, ...],
    h_hist: torch.Tensor,
    attn_sums: torch.Tensor,
    k: int,
) -> int:
    B, T = valid.shape
    H, dh, dev = parts.n_heads, parts.d_head, valid.device
    hvalid = valid.clone()
    hvalid[:, :HEAD_START] = False
    if not hvalid.any():
        hvalid = valid.clone()  # short documents: use every position past the first
    attentions = getattr(out, "attentions", None)
    q = torch.arange(T, device=dev)
    offsets = torch.arange(H, device=dev) * NB
    for layer in range(parts.n_layers):
        z = zs[layer].view(B, T, H, dh)
        W_O = parts.blocks[layer].self_attn.o_proj.weight.detach().float().view(parts.d_model, H, dh)
        norms = torch.einsum("bthe,dhe->bthd", z, W_O).norm(dim=-1)  # [B, T, H]
        mx, arg = norms.masked_fill(~hvalid[..., None], -math.inf).max(dim=1)  # [B, H]
        src = torch.zeros_like(arg)
        pattern = attentions[layer] if attentions is not None and len(attentions) > layer else None
        if pattern is not None:
            pattern = pattern.float()  # [B, H, T, T]
            rows = pattern.gather(2, arg[:, :, None, None].expand(-1, -1, 1, T)).squeeze(2)  # [B, H, T]
            src = rows[..., 1:].argmax(-1) + 1 if T > 1 else src
            sel = hvalid[:, None, :].float()
            prev = torch.zeros_like(pattern[..., 0])
            prev[..., 1:] = pattern[:, :, q[1:], q[1:] - 1]
            for j, part in enumerate((pattern[..., 0], prev, pattern[:, :, q, q])):
                attn_sums[layer, j] += (part * sel).sum((0, 2)).double()
        kept = tuple(t[layer] for t in h_top)
        merged = _merge_top(kept, (mx, doc_ids[:, None].expand_as(arg), arg, src), k)  # type: ignore[arg-type,unused-ignore]
        for t, m in zip(h_top, merged, strict=True):
            t[layer] = m
        flat = (_bins(norms[hvalid]) + offsets).reshape(-1)
        h_hist[layer] += torch.bincount(flat, minlength=H * NB).view(H, NB)
    return int(hvalid.sum())


@torch.no_grad()
def token_drivers(
    parts: Parts,
    contexts: Sequence[Sequence[int]],
    candidates: Sequence[int],
    position: int = 24,
    k: int = 16,
    batch_size: int = 64,
    layers: Sequence[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Exact token drivers: each candidate token at `position` of each context, the pole's activation there,
    averaged over the contexts. Returns `ids`, `vals` `[len(layers), 2, k, d_mlp]` (CPU) and what was run.

    Contexts shorter than `position + 1` tokens shorten the position for all (it is one position for every context).
    """
    layers = list(range(parts.n_layers)) if layers is None else sorted(set(layers))
    position = min(position, min(len(c) for c in contexts) - 1)
    if position < 1:
        raise ValueError("token drivers need contexts of at least 2 tokens")
    dev = parts.device
    d = parts.d_mlp
    k = min(k, len(candidates))
    top_val = torch.full((len(layers), 2, k, d), -math.inf, device=dev)
    top_ids = torch.zeros((len(layers), 2, k, d), dtype=torch.long, device=dev)
    prefixes = [torch.tensor(list(c[:position]), dtype=torch.long, device=dev) for c in contexts]
    cand = torch.tensor(list(candidates), dtype=torch.long, device=dev)
    acts, _, handles = _capture(parts, layers, stop_after=layers[-1])
    try:
        for start in range(0, len(cand), batch_size):
            chunk = cand[start : start + batch_size]
            acc = torch.zeros((len(layers), len(chunk), d), device=dev)
            for prefix in prefixes:
                ids = torch.cat([prefix[None, :].expand(len(chunk), -1), chunk[:, None]], 1)
                with contextlib.suppress(_Stop):
                    parts.model(input_ids=ids, use_cache=False)
                for j, layer in enumerate(layers):
                    acc[j] += acts[layer][:, -1, :]
            acc /= len(prefixes)
            for j in range(len(layers)):
                for pi, sign in enumerate((1.0, -1.0)):
                    v = acc[j] * sign  # [B, d]
                    merged = _merge_top((top_val[j, pi], top_ids[j, pi]), (v, chunk[:, None].expand_as(v)), k)
                    top_val[j, pi], top_ids[j, pi] = merged
            if progress:
                progress(min(start + batch_size, len(cand)), len(cand))
    finally:
        for handle in handles:
            handle.remove()
    return {
        "layers": layers,
        "ids": top_ids.cpu(),
        "vals": top_val.cpu(),
        "position": position,
        "n_contexts": len(contexts),
        "n_candidates": len(candidates),
    }


# Reads from the weights


@torch.no_grad()
def write_logits(parts: Parts, layer: int, neuron: int, pole: int) -> torch.Tensor:
    """The pole's write read through the logit lens: `W_U · (γ_final ⊙ pole · W_down[:, n])`, one score per token."""
    w = parts.down(layer)[:, neuron] * pole
    return parts.W_U.float() @ (parts.final_norm() * w)  # type: ignore[no-any-return,unused-ignore]


@torch.no_grad()
def layer_write_rms(parts: Parts, layer: int, chunk: int = 8192) -> torch.Tensor:
    """Per token, the root mean square over the layer's poles of the write's logit-lens score: the z-score's scale.

    Over both poles of every neuron the mean is exactly 0 (each pole is the other's negative), so z = score / rms."""
    W = parts.final_norm()[:, None] * parts.down(layer)  # [D, d]
    out = torch.empty(parts.vocab, device=W.device)
    for start in range(0, parts.vocab, chunk):
        block = parts.W_U[start : start + chunk].float() @ W  # [c, d]
        out[start : start + chunk] = block.pow(2).mean(-1).sqrt()
    return out.clamp_min(1e-12)


@torch.no_grad()
def read_activations(parts: Parts, layer: int, neuron: int, pole: int, ids: torch.Tensor | None = None) -> torch.Tensor:
    """The pole's activation if the MLP input held only each token's normalized embedding (exact at L0 only)."""
    x = parts.normed_embeddings(ids, parts.mlp_norm(layer))
    g = x @ parts.gate(layer)[neuron]
    u = x @ parts.up(layer)[neuron]
    return parts.act_fn(layer)(g) * u * pole


def top_tokens(
    scores: torch.Tensor, k: int, ids: torch.Tensor | None = None, lowest: bool = False
) -> list[tuple[int, float]]:
    """The `k` highest (or lowest) scores as (token id, score); `ids` maps positions to token ids."""
    values, idx = (-scores if lowest else scores).topk(min(k, scores.numel()))
    token_ids = idx if ids is None else ids[idx]
    sign = -1.0 if lowest else 1.0
    return [(int(t), float(v) * sign) for t, v in zip(token_ids.tolist(), values.tolist(), strict=False)]


def _rms_norm_rows(x: torch.Tensor, module: Any, gemma: bool) -> torch.Tensor:
    if module is None:
        return x
    weight = module.weight.detach().float()
    weight = weight + 1.0 if gemma else weight
    return x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * weight  # type: ignore[no-any-return,unused-ignore]


@torch.no_grad()
def head_reads(parts: Parts, layer: int, head: int, candidates: Sequence[int], k: int = 12) -> dict[str, Any]:
    """A head's OV and QK circuits over candidate tokens, from the weights alone (position-free: RoPE at offset 0).

    OV: each candidate's normalized embedding through the head's value and output maps, read by the logit lens over the
    candidates; a token's own score, z-scored in its row, says whether the head copies (+) or suppresses (−) what it
    reads. `copy_score` is Σλ/Σ|λ| of that map (+1 copies, −1 suppresses), the copying score of "A Mathematical
    Framework for Transformer Circuits".
    QK: `attends` ranks keys under the mean query; `self` ranks each token's attention to an earlier copy of itself.
    """
    attn = parts.blocks[layer].self_attn
    dh, group = parts.d_head, max(1, parts.n_heads // parts.n_kv)
    kv = head // group
    ids = torch.tensor(list(candidates), dtype=torch.long, device=parts.device)
    x = parts.normed_embeddings(ids, parts.attn_norm(layer))  # [C, D]

    def proj(linear: Any, index: int) -> torch.Tensor:
        weight = linear.weight.detach().float()[index * dh : (index + 1) * dh]
        out = x @ weight.T
        if getattr(linear, "bias", None) is not None:
            out = out + linear.bias.detach().float()[index * dh : (index + 1) * dh]
        return out  # type: ignore[no-any-return,unused-ignore]

    q = _rms_norm_rows(proj(attn.q_proj, head), getattr(attn, "q_norm", None), parts.gemma)
    key = _rms_norm_rows(proj(attn.k_proj, kv), getattr(attn, "k_norm", None), parts.gemma)
    value = proj(attn.v_proj, kv)
    W_O = attn.o_proj.weight.detach().float()[:, head * dh : (head + 1) * dh]  # [D, dh]
    readout = (W_O.T * parts.final_norm()) @ parts.W_U[ids].float().T  # [dh, C]
    ov = value @ readout  # [C, C]: row = token read, column = token written
    z = (ov - ov.mean(1, keepdim=True)) / ov.std(1, keepdim=True).clamp_min(1e-9)
    diag = z.diagonal()
    eig = torch.linalg.eigvals(readout @ value)
    copy_score = float(eig.real.sum() / eig.abs().sum().clamp_min(1e-12))
    col = ov.mean(0)
    mean_out = (col - col.mean()) / col.std().clamp_min(1e-9)
    scores = q @ key.T / math.sqrt(dh)  # [C, C]: query row, key column
    key_scores = key @ q.mean(0) / math.sqrt(dh)
    kz = (key_scores - key_scores.mean()) / key_scores.std().clamp_min(1e-9)
    sz = (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True).clamp_min(1e-9)
    self_z = sz.diagonal()
    return {
        "kv_head": kv,
        "copies": top_tokens(diag, k, ids),
        "suppresses": top_tokens(diag, k, ids, lowest=True),
        "writes": top_tokens(mean_out, k, ids),
        "attends": top_tokens(kz, k, ids),
        "avoids": top_tokens(kz, k, ids, lowest=True),
        "self": top_tokens(self_z, k, ids),
        "copy_score": copy_score,
        "copy_median_z": float(diag.median()),
        "self_median_z": float(self_z.median()),
        "n_candidates": len(candidates),
    }


@torch.no_grad()
def read_directions(parts: Parts, layer: int) -> torch.Tensor:
    """`[2, d_mlp, D]`: each pole's read direction, `ĝ ± û` (unit gate and up rows times the MLP norm's weight)."""
    weight = parts.mlp_norm(layer)
    g = parts.gate(layer) * weight
    u = parts.up(layer) * weight
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    both = torch.stack([g + u, g - u])
    return both / both.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # type: ignore[no-any-return,unused-ignore]


@torch.no_grad()
def unit_edges(
    parts: Parts,
    layer: int,
    neuron: int,
    pole: int,
    k: int = 16,
    window: int | None = None,
    act_top: Callable[[int, int, int], float | None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """A pole's strongest direct weight edges, from the weights alone.

    `out`: later poles whose read direction its write points along, by |cos|. `in`: earlier neurons whose write points
    along its read direction, each listed as the pole that drives it (cos > 0); its other pole pushes against it by the
    same amount. With `act_top(layer, neuron, pole)`, each edge also gets `push` = cos · act_top · ‖w‖ of the writer
    (unnormalized: the catalog also divides by the residual's RMS, which a weights-only read doesn't know)."""
    w = parts.down(layer)[:, neuron] * pole
    w_norm = float(w.norm())
    w_hat = w / max(w_norm, 1e-12)
    last = parts.n_layers if window is None else min(parts.n_layers, layer + window + 1)
    first = 0 if window is None else max(0, layer - window)
    outs: list[tuple[float, int, int, int, float]] = []
    for later in range(layer + 1, last):
        flat = (read_directions(parts, later) @ w_hat).reshape(-1)  # [2 * d]
        for i in flat.abs().topk(min(k, flat.numel())).indices.tolist():
            p_index, n = divmod(i, parts.d_mlp)
            c = float(flat[i])
            outs.append((abs(c), later, n, 1 if p_index == 0 else -1, c))
    outs.sort(reverse=True)
    r = read_directions(parts, layer)[0 if pole > 0 else 1, neuron]
    ins: list[tuple[float, int, int, int, float]] = []
    for earlier in range(first, layer):
        W = parts.down(earlier)  # [D, d]
        cos = (r @ W) / W.norm(dim=0).clamp_min(1e-12)  # [d]
        for n in cos.abs().topk(min(k, cos.numel())).indices.tolist():
            c = float(cos[n])
            ins.append((abs(c), earlier, n, 1 if c >= 0 else -1, abs(c)))
    ins.sort(reverse=True)
    mine = act_top(layer, neuron, pole) if act_top else None

    def item(lay: int, n: int, p: int, c: float, side: str) -> dict[str, Any]:
        out: dict[str, Any] = {"layer": lay, "neuron": n, "pole": p, "cos": c}
        if act_top is None:
            return out
        if side == "out" and mine is not None:
            out["push"] = c * mine * w_norm
        elif side == "in":
            theirs = act_top(lay, n, p)
            if theirs is not None:
                out["push"] = c * theirs * float(parts.down(lay)[:, n].norm())
        return out

    return {
        "out": [item(lay, n, p, c, "out") for _, lay, n, p, c in outs[:k]],
        "in": [item(lay, n, p, c, "in") for _, lay, n, p, c in ins[:k]],
    }


@torch.no_grad()
def unit_trace(parts: Parts, ids: Sequence[int], layer: int, kind: str, index: int, pole: int = 1) -> list[float]:
    """One unit's value at every position of a prompt: a pole's activation (× its sign) or a head's output norm."""
    acts, zs, handles = _capture(parts, [layer], heads=kind == "head", stop_after=layer)
    try:
        with contextlib.suppress(_Stop):
            parts.model(input_ids=torch.tensor([list(ids)], device=parts.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if kind == "head":
        dh = parts.d_head
        z = zs[layer][0].view(len(ids), parts.n_heads, dh)[:, index]
        W_O = parts.blocks[layer].self_attn.o_proj.weight.detach().float()[:, index * dh : (index + 1) * dh]
        values: list[float] = (z @ W_O.T).norm(dim=-1).tolist()
        return values
    trace: list[float] = (acts[layer][0, :, index] * pole).tolist()
    return trace
