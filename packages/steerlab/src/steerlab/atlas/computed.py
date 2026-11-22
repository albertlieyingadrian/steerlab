"""A computed atlas on disk, and its unit cards in the catalog's card shape (with `source: "computed"`).

    atlas = ComputedAtlas(cache_dir / "computed" / "qwen2.5-0.5b-instruct", parts=Parts(model, tokenizer))
    atlas.build()                                       # corpus pass + token drivers: minutes for a 0.5B model on CPU
    atlas.unit("qwen2.5-0.5b-instruct", Address.parse("L12/n100+"))   # contexts, drivers, percentiles, reads, edges

Files (`<root>/`): `meta.json` (what was run), `corpus.json` (the token ids), `tokens.json` (each used token id's text,
so cards render without the tokenizer), `stats.npz` (top firings and quantiles, `compute.corpus_stats`), `drivers.npz`
(`compute.token_drivers`), and, when built with `weights=True` (the Modal path), `weights.npz` and `heads.json`: the
weight reads for every pole and head, so a large model's atlas can be browsed on a laptop that cannot load it.

There are no LLM labels here. A computed unit's title is its measured tokens ("fires on ' the' (6/16) · ' a'"), its
easiest-units score is `consistency`, the share of its top firings on its most common token, and the card says so.
Reading needs numpy only; building and the on-demand weight reads need torch (`compute`).
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from steerlab.atlas.address import Address
from steerlab.atlas.cards import NEAR_SILENT, describe_percentile, num, percentile, rank_percentile, show
from steerlab.atlas.models import band

if TYPE_CHECKING:
    from steerlab.atlas.compute import Parts

LABEL_BY = "computed (no LLM label): the unit's own measured tokens"
CONTEXT_BEFORE, CONTEXT_AFTER = 14, 5
TOP_TOKENS = 8
HEAD_CANDIDATES = 2048
HEAD_READS: dict[str, tuple[str, str]] = {
    "copies": ("writes", "the tokens it copies most: each read token's own logit-lens score, z-scored in its row"),
    "suppresses": ("writes", "the tokens it suppresses most when it reads them"),
    "writes": ("writes", "what it writes on average over the candidate tokens it reads"),
    "attends": ("attends", "key tokens it attends to under its mean query (position-free)"),
    "avoids": ("attends", "key tokens it avoids"),
    "self": ("attends", "tokens that look for earlier copies of themselves"),
}


class ComputedAtlas:
    def __init__(self, root: str | Path, parts: Parts | None = None, hf_id: str | None = None) -> None:
        self.root = Path(root)
        self.parts = parts
        self.hf_id = hf_id
        self._lock = threading.RLock()
        self._stats: dict[str, np.ndarray] | None = None
        self._drivers: dict[str, np.ndarray] | None = None
        self._weights: dict[str, np.ndarray] | None = None
        self._heads: dict[str, Any] | None = None
        self._meta: dict[str, Any] | None = None
        self._corpus: list[list[int]] = []
        self._slices: list[str] = []
        self._tokens: dict[int, str] = {}
        self._write_rms: dict[int, Any] = {}
        self._head_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._kept: Any = None

    # Files

    @property
    def ready(self) -> bool:
        return (self.root / "stats.npz").exists() and (self.root / "meta.json").exists()

    @property
    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            if not self.ready:
                raise FileNotFoundError(f"no computed atlas at {self.root}; build it first")
            self._meta = json.loads((self.root / "meta.json").read_text())
        return self._meta

    def _load(self) -> None:
        with self._lock:
            if self._stats is not None:
                return
            if not self.ready:
                raise FileNotFoundError(f"no computed atlas at {self.root}; build it first")
            self._drivers = self._npz("drivers.npz")
            self._weights = self._npz("weights.npz")
            heads = self.root / "heads.json"
            self._heads = json.loads(heads.read_text()) if heads.exists() else None
            corpus = json.loads((self.root / "corpus.json").read_text())
            self._corpus = [list(map(int, doc)) for doc in corpus["docs"]]
            self._slices = list(corpus.get("slices", []))
            tokens = json.loads((self.root / "tokens.json").read_text())
            self._tokens = {int(key): value for key, value in tokens.items()}
            self._stats = self._npz("stats.npz")

    def _npz(self, name: str) -> dict[str, np.ndarray] | None:
        path = self.root / name
        if not path.exists():
            return None
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    @property
    def stats(self) -> dict[str, np.ndarray]:
        self._load()
        assert self._stats is not None
        return self._stats

    def text(self, token_id: int) -> str:
        token = self._tokens.get(int(token_id))
        if token is None and self.parts is not None:
            token = self.parts.decode(int(token_id))
            self._tokens[int(token_id)] = token
        return token if token is not None else f"<{token_id}>"

    # Building

    def build(
        self,
        parts: Parts | None = None,
        corpus: Any = None,
        n_tokens: int = 24_000,
        max_len: int = 128,
        k: int = 16,
        n_candidates: int = 2048,
        n_driver_contexts: int = 4,
        position: int = 24,
        weights: bool = False,
        batch_size: int = 8,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """Run the corpus pass and the token drivers (and, with `weights`, every weight read), and save them."""
        from steerlab.atlas import compute
        from steerlab.atlas.corpus import build_corpus

        parts = parts or self.parts
        if parts is None:
            raise ValueError("building a computed atlas needs the model loaded")
        started = time.time()

        def say(stage: str, done: float) -> None:
            if progress:
                progress(stage, done)

        if corpus is None:
            say("corpus", 0.0)
            corpus = build_corpus(
                parts.tokenizer, n_tokens=n_tokens, max_len=max_len, cache_dir=self.root.parent.parent
            )
        self.root.mkdir(parents=True, exist_ok=True)
        stats = compute.corpus_stats(
            parts, corpus.docs, k=k, batch_size=batch_size, progress=lambda i, n: say("activations", i / n)
        )
        counts = Counter(t for doc in corpus.docs for t in doc[1:])
        candidates = [t for t, _ in counts.most_common(n_candidates)]
        contexts = [doc for doc in corpus.docs if len(doc) > position][:n_driver_contexts] or corpus.docs[:1]
        drivers = compute.token_drivers(
            parts,
            contexts,
            candidates,
            position=position,
            k=min(k, len(candidates)),
            progress=lambda i, n: say("drivers", i / n),
        )
        used = set(counts) | {int(t) for doc in corpus.docs for t in doc}
        used |= {int(t) for t in drivers["ids"].reshape(-1).tolist()}
        arrays = {key: value.numpy() for key, value in stats.items() if hasattr(value, "numpy")}
        for key in ("top_doc", "top_pos", "head_top_doc", "head_top_pos", "head_src_pos"):
            if key in arrays:
                arrays[key] = arrays[key].astype(np.int32)
        arrays["k"] = np.array(stats["k"])
        np.savez_compressed(self.root / "stats.npz", **arrays)
        np.savez_compressed(
            self.root / "drivers.npz",
            ids=drivers["ids"].numpy().astype(np.int32),
            vals=drivers["vals"].numpy().astype(np.float32),
        )
        if weights:
            used |= self._build_weights(parts, candidates, say)
        (self.root / "corpus.json").write_text(json.dumps(corpus.to_json()))
        (self.root / "tokens.json").write_text(json.dumps({str(t): parts.decode(t) for t in sorted(used)}))
        meta = {
            "hf_id": self.hf_id or str(getattr(parts.model.config, "name_or_path", "")),
            "n_layers": parts.n_layers,
            "d_mlp": parts.d_mlp,
            "n_heads": parts.n_heads,
            "n_kv": parts.n_kv,
            "corpus": corpus.describe(),
            "n_positions": stats["n_positions"],
            "n_docs": len(corpus.docs),
            "k": stats["k"],
            "drivers": {
                "position": drivers["position"],
                "n_contexts": drivers["n_contexts"],
                "n_candidates": drivers["n_candidates"],
            },
            "weights": weights,
            "seconds": round(time.time() - started, 1),
            "device": str(parts.device),
            "built": time.strftime("%Y-%m-%d %H:%M"),
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        with self._lock:
            self._meta = meta
            self._stats = None
        say("done", 1.0)
        return meta

    def _build_weights(self, parts: Parts, candidates: Sequence[int], say: Callable[[str, float], None]) -> set[int]:
        """Every pole's write and read token reads, and every head's OV/QK reads (the Modal path)."""
        import torch

        from steerlab.atlas import compute

        L, d, k = parts.n_layers, parts.d_mlp, TOP_TOKENS
        kept = torch.tensor(sorted(set(candidates)), device=parts.device)
        names = ("write_ids", "write_vals", "write_z_ids", "write_z_vals", "read_ids", "read_vals")
        out = {name: np.zeros((L, 2, d, k), np.int32 if name.endswith("ids") else np.float32) for name in names}
        g = parts.final_norm()
        with torch.no_grad():
            for layer in range(L):
                W = g[:, None] * parts.down(layer)  # [D, d]
                rms = compute.layer_write_rms(parts, layer)
                best = [torch.full((k, d), -float("inf"), device=parts.device) for _ in range(4)]
                best_i = [torch.zeros((k, d), dtype=torch.long, device=parts.device) for _ in range(4)]
                for start in range(0, parts.vocab, 8192):
                    block = parts.W_U[start : start + 8192].float() @ W  # [c, d]
                    ids = torch.arange(start, start + block.shape[0], device=parts.device)
                    scale = rms[start : start + block.shape[0], None]
                    for j, scores in enumerate((block, -block, block / scale, -block / scale)):
                        cand = torch.cat([best[j], scores], 0)
                        cand_i = torch.cat([best_i[j], ids[:, None].expand_as(scores)], 0)
                        best[j], idx = cand.topk(k, dim=0)
                        best_i[j] = cand_i.gather(0, idx)
                for pi in range(2):
                    out["write_ids"][layer, pi] = best_i[pi].T.cpu().numpy()
                    out["write_vals"][layer, pi] = best[pi].T.cpu().numpy()
                    out["write_z_ids"][layer, pi] = best_i[2 + pi].T.cpu().numpy()
                    out["write_z_vals"][layer, pi] = best[2 + pi].T.cpu().numpy()
                x = parts.normed_embeddings(kept, parts.mlp_norm(layer))
                a = parts.act_fn(layer)(x @ parts.gate(layer).T) * (x @ parts.up(layer).T)  # [C, d]
                for pi, sign in enumerate((1.0, -1.0)):
                    top, idx = (a * sign).topk(min(k, len(kept)), dim=0)
                    out["read_ids"][layer, pi, :, : top.shape[0]] = kept[idx].T.cpu().numpy()
                    out["read_vals"][layer, pi, :, : top.shape[0]] = top.T.cpu().numpy()
                say("weights", (layer + 1) / (2 * L))
        heads: dict[str, Any] = {}
        for layer in range(L):
            for head in range(parts.n_heads):
                heads[f"{layer}/{head}"] = compute.head_reads(parts, layer, head, candidates[:HEAD_CANDIDATES])
            say("weights", 0.5 + (layer + 1) / (2 * L))
        np.savez_compressed(self.root / "weights.npz", **cast(dict[str, Any], out))
        (self.root / "heads.json").write_text(json.dumps(heads))
        used: set[int] = set()
        for name in ("write_ids", "write_z_ids", "read_ids"):
            used |= {int(t) for t in np.unique(out[name]).tolist()}
        for read in heads.values():
            for key in HEAD_READS:
                used |= {int(t) for t, _ in read[key]}
        return used

    # Cards

    def check(self, address: Address) -> None:
        meta = self.meta
        if not 0 <= address.layer < meta["n_layers"]:
            raise ValueError(f"layer {address.layer} is outside 0..{meta['n_layers'] - 1}")
        size = meta["n_heads"] if address.is_head else meta["d_mlp"]
        if not 0 <= address.index < size:
            raise ValueError(f"{'head' if address.is_head else 'neuron'} {address.index} is outside 0..{size - 1}")

    def act_top(self, layer: int, neuron: int, pole: int) -> float | None:
        return num(self.stats["act_top"][layer, 0 if pole > 0 else 1, neuron])

    def _context(
        self, doc: int, pos: int, value: float, stats: dict[str, Any], src: int | None = None
    ) -> dict[str, Any]:
        tokens = self._corpus[doc]
        start = max(0, pos - CONTEXT_BEFORE)

        def span(a: int, b: int) -> str:
            return "".join(self.text(t) for t in tokens[a:b]).replace("\n", "⏎")

        piece: dict[str, Any] = {
            "pre": span(start, pos),
            "tok": span(pos, pos + 1),
            "post": span(pos + 1, pos + 1 + CONTEXT_AFTER),
            "value": value,
        }
        if src is not None and start <= src < pos:
            piece["pre"], piece["src"], piece["mid"] = span(start, src), span(src, src + 1), span(src + 1, pos)
        pct = percentile(value, stats)
        piece["pct"], piece["pct_text"] = pct, describe_percentile(pct)
        if self._slices:
            piece["slice"] = self._slices[doc]
        return piece

    def _counted(self, ids: Sequence[int]) -> list[list[Any]]:
        return [[show(self.text(t)), c] for t, c in Counter(ids).most_common(5)]

    def _firings(self, prefix: str, layer: int, index: int, pi: int | None) -> list[tuple[int, int, float]]:
        s = self.stats
        at: tuple[Any, ...] = (layer, slice(None), index) if pi is None else (layer, pi, slice(None), index)
        docs, poss, vals = s[f"{prefix}top_doc"][at], s[f"{prefix}top_pos"][at], s[f"{prefix}top_val"][at]
        return [(int(d), int(p), float(v)) for d, p, v in zip(docs, poss, vals, strict=False) if v > 0]

    def _peak_ids(self, layer: int, pi: int, n: int) -> tuple[list[int], list[int]]:
        peaks, nexts = [], []
        for doc, pos, _ in self._firings("", layer, n, pi):
            tokens = self._corpus[doc]
            peaks.append(tokens[pos])
            if pos + 1 < len(tokens):
                nexts.append(tokens[pos + 1])
        return peaks, nexts

    def consistency(self, layer: int, pi: int, n: int) -> float:
        """The share of the pole's top firings on its most common peak token (0 when it never fires)."""
        peaks, _ = self._peak_ids(layer, pi, n)
        if not peaks:
            return 0.0
        return Counter(peaks).most_common(1)[0][1] / len(peaks)

    def _fires(self, peaks: Sequence[int]) -> str:
        if not peaks:
            return "never fires in the corpus"
        common = Counter(peaks).most_common(3)
        return "fires on " + " · ".join(f"'{show(self.text(t))}' ({c}/{len(peaks)})" for t, c in common)

    def label(self, address: Address) -> dict[str, Any]:
        self._load()
        label: dict[str, Any] = {"source": "computed", "by": LABEL_BY, "lent_from": None, "note": ""}
        if address.is_head:
            peaks = [self._corpus[d][p] for d, p, _ in self._firings("head_", address.layer, address.index, None)]
            label.update(fires=self._fires(peaks), fires_conf=None, attends="", attends_conf=None, writes="")
            return label
        peaks, _ = self._peak_ids(address.layer, 0 if address.pole > 0 else 1, address.index)
        writes = ""
        tokens = self._write_tokens(address)
        if tokens:
            writes = "logit lens of its write: " + " · ".join(t for t, _ in tokens[:4])
        label.update(fires=self._fires(peaks), fires_conf=None, writes=writes, writes_conf=None)
        return label

    def title(self, address: Address) -> str:
        try:
            self.check(address)
            self._load()
            if address.is_head:
                peaks = [self._corpus[d][p] for d, p, _ in self._firings("head_", address.layer, address.index, None)]
            else:
                peaks, _ = self._peak_ids(address.layer, 0 if address.pole > 0 else 1, address.index)
            return self._fires(peaks)
        except (FileNotFoundError, ValueError, IndexError):
            return ""

    def _kept_ids(self) -> Any:
        import torch

        if self._kept is None:
            assert self.parts is not None
            self._kept = torch.tensor(sorted({t for doc in self._corpus for t in doc}), device=self.parts.device)
        return self._kept

    def _write_tokens(self, address: Address, z: bool = False) -> list[list[Any]]:
        pi = 0 if address.pole > 0 else 1
        if self._weights is not None:
            prefix = "write_z" if z else "write"
            ids = self._weights[f"{prefix}_ids"][address.layer, pi, address.index]
            vals = self._weights[f"{prefix}_vals"][address.layer, pi, address.index]
            return [[show(self.text(int(t))), num(v)] for t, v in zip(ids, vals, strict=False)]
        if self.parts is None:
            return []
        from steerlab.atlas import compute

        scores = compute.write_logits(self.parts, address.layer, address.index, address.pole)
        if z:
            if address.layer not in self._write_rms:
                self._write_rms[address.layer] = compute.layer_write_rms(self.parts, address.layer)
            scores = scores / self._write_rms[address.layer]
        return [[show(self.text(t)), v] for t, v in compute.top_tokens(scores, TOP_TOKENS)]

    def _read_tokens(self, address: Address) -> list[list[Any]]:
        pi = 0 if address.pole > 0 else 1
        if self._weights is not None:
            ids = self._weights["read_ids"][address.layer, pi, address.index]
            vals = self._weights["read_vals"][address.layer, pi, address.index]
            return [[show(self.text(int(t))), num(v)] for t, v in zip(ids, vals, strict=False)]
        if self.parts is None:
            return []
        from steerlab.atlas import compute

        kept = self._kept_ids()
        scores = compute.read_activations(self.parts, address.layer, address.index, address.pole, kept)
        return [[show(self.text(t)), v] for t, v in compute.top_tokens(scores, TOP_TOKENS, kept)]

    def _z_affordable(self) -> bool:
        """Whether the z-scored write read (vocab × d_model × d_mlp multiply-adds) is quick enough to do on demand."""
        if self._weights is not None:
            return True
        if self.parts is None:
            return False
        return self.parts.vocab * self.parts.d_model * self.parts.d_mlp < 1.5e12

    def _base(self, slug: str, address: Address) -> dict[str, Any]:
        meta = self.meta
        return {
            "model": slug,
            "hf_id": meta.get("hf_id"),
            "address": address.path(),
            "display": str(address),
            "kind": address.kind,
            "layer": address.layer,
            "index": address.index,
            "pole": address.pole,
            "band": band(address.layer, meta["n_layers"]),
            "n_layers": meta["n_layers"],
            "source": "computed",
            "corpus": meta["corpus"],
            "built": meta.get("built"),
            "links": {},
            "lineage": [],
            "sources": {},
        }

    def unit(self, slug: str, address: Address) -> dict[str, Any]:
        self._load()
        self.check(address)
        card = self._base(slug, address)
        card["label"] = self.label(address)
        if address.is_head:
            return self._head_card(card, address)
        layer, n = address.layer, address.index
        pi = 0 if address.pole > 0 else 1
        s = self.stats
        stats: dict[str, Any] = {
            key: num(s[key][layer, pi, n]) for key in ("act_top", "act_max", "rate", "q50", "q90", "q99", "q999")
        }
        stats["near_silent"] = (stats["act_top"] or 0.0) < NEAR_SILENT
        stats["layer_pct"] = rank_percentile(s["act_top"][layer].reshape(-1).tolist(), stats["act_top"] or 0.0)
        stats["consistency"] = self.consistency(layer, pi, n)
        card["stats"] = stats
        card["contexts"] = [self._context(d, p, v, stats) for d, p, v in self._firings("", layer, n, pi)[:8]]
        peaks, nexts = self._peak_ids(layer, pi, n)
        card["peak_tokens"], card["next_tokens"] = self._counted(peaks), self._counted(nexts)
        drivers: dict[str, Any] = {"at": [], "sum": []}
        if self._drivers is not None:
            ids, vals = self._drivers["ids"][layer, pi, :, n], self._drivers["vals"][layer, pi, :, n]
            drivers["at"] = [[show(self.text(int(t))), num(v)] for t, v in zip(ids, vals, strict=False)][:TOP_TOKENS]
            info = self.meta.get("drivers", {})
            drivers["how"] = (
                f"each of the corpus's {info.get('n_candidates')} most common tokens put at position "
                f"{info.get('position')} of {info.get('n_contexts')} corpus documents, averaged; activation × the "
                "pole's sign"
            )
        card["drivers"] = drivers
        reads = []
        write = self._write_tokens(address)
        if write:
            reads.append(
                {
                    "key": "write_logit",
                    "side": "writes",
                    "lens": "logit",
                    "tokens": [t for t, _ in write],
                    "what": "the logit lens of its write (W_U · final norm): exact at the last layer, rough "
                    "before the last third",
                }
            )
        if self._z_affordable():
            write_z = self._write_tokens(address, z=True)
            if write_z:
                reads.append(
                    {
                        "key": "write_logit_z",
                        "side": "writes",
                        "lens": "logit",
                        "tokens": [t for t, _ in write_z],
                        "what": "the same, z-scored against the layer's poles (drops hub tokens)",
                    }
                )
        read = self._read_tokens(address)
        if read:
            reads.append(
                {
                    "key": "read_embed",
                    "side": "fires on",
                    "lens": "embedding",
                    "tokens": [t for t, _ in read],
                    "what": "its activation if the MLP input held only that token's embedding: exact at L0, "
                    "a rough stand-in for the S-lens later",
                }
            )
        card["reads"] = reads
        card["edges"] = self.edges(address)
        return card

    def _head_card(self, card: dict[str, Any], address: Address) -> dict[str, Any]:
        s = self.stats
        layer, h = address.layer, address.index
        group = max(1, self.meta["n_heads"] // max(1, self.meta.get("n_kv") or self.meta["n_heads"]))
        stats: dict[str, Any] = {
            "out_top": num(s["head_out_top"][layer, h]),
            "out_max": num(s["head_out_max"][layer, h]),
            "q50": num(s["head_q50"][layer, h]),
            "q90": num(s["head_q90"][layer, h]),
            "q99": num(s["head_q99"][layer, h]),
            "q999": num(s["head_q999"][layer, h]),
            "sink": num(s["head_attn"][layer, 0, h]),
            "prev": num(s["head_attn"][layer, 1, h]),
            "self": num(s["head_attn"][layer, 2, h]),
            "kv_head": h // group,
        }
        card["stats"] = stats
        src = s["head_src_pos"][layer, :, h]
        firings = self._firings("head_", layer, h, None)
        card["contexts"] = [self._context(d, p, v, stats, src=int(src[i])) for i, (d, p, v) in enumerate(firings[:8])]
        card["peak_tokens"] = self._counted([self._corpus[d][p] for d, p, _ in firings])
        card["src_tokens"] = self._counted([self._corpus[d][int(src[i])] for i, (d, _, _) in enumerate(firings)])
        card["reads"], card["copy"] = self._head_reads(layer, h)
        card["drivers"] = {"at": [], "keys": [], "how": "not computed for heads; see the OV and QK reads"}
        card["edges"] = {
            "in": [],
            "out": [],
            "source": "computed",
            "how": "direct weight edges are computed for neurons",
        }
        return card

    def _head_reads(self, layer: int, head: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        read: dict[str, Any] | None = None
        if self._heads is not None:
            read = self._heads.get(f"{layer}/{head}")
        elif self.parts is not None:
            if (layer, head) not in self._head_cache:
                from steerlab.atlas import compute

                counts = Counter(t for doc in self._corpus for t in doc)
                candidates = [t for t, _ in counts.most_common(HEAD_CANDIDATES)]
                self._head_cache[(layer, head)] = compute.head_reads(self.parts, layer, head, candidates)
            read = self._head_cache[(layer, head)]
        if read is None:
            return [], {}
        reads = [
            {
                "key": f"{'ov' if side == 'writes' else 'qk'}_{key}",
                "side": side,
                "lens": "weights",
                "what": text,
                "tokens": [show(self.text(int(t))) for t, _ in read[key][:TOP_TOKENS]],
            }
            for key, (side, text) in HEAD_READS.items()
            if read.get(key)
        ]
        copy = {
            "copy_score": num(read.get("copy_score")),
            "copy_median_z": num(read.get("copy_median_z")),
            "self_median_z": num(read.get("self_median_z")),
            "n_candidates": read.get("n_candidates"),
        }
        return reads, copy

    def edges(self, address: Address, limit: int = 16) -> dict[str, Any]:
        out: dict[str, Any] = {"source": "computed", "in": [], "out": []}
        if address.is_head:
            out["how"] = "direct weight edges are computed for neurons"
            return out
        if self.parts is None:
            out["how"] = "weight edges are computed when the model is loaded"
            return out
        from steerlab.atlas import compute

        found = compute.unit_edges(
            self.parts, address.layer, address.index, address.pole, k=limit, act_top=self.act_top
        )
        out["how"] = (
            "cos of the earlier pole's write with the later pole's read direction (ĝ ± û), from the weights; push = "
            "cos · act_top · ‖w‖ (not divided by the residual's RMS)"
        )
        for side in ("in", "out"):
            out[side] = [
                {
                    "address": Address.neuron(e["layer"], e["neuron"], e["pole"]).path(),
                    "cos": e["cos"],
                    "push": e.get("push", e["cos"]),
                }
                for e in found[side]
            ]
        return out

    # Overview and search

    def overview(self, slug: str, top: int = 40) -> dict[str, Any]:
        self._load()
        meta = self.meta
        s = self.stats
        layers = []
        for layer in range(meta["n_layers"]):
            act_top = s["act_top"][layer]  # [2, d]
            live = [(int(pi), int(n)) for pi, n in np.argwhere(act_top >= NEAR_SILENT)]
            scored = sorted(
                ((self.consistency(layer, pi, n), float(act_top[pi, n]), pi, n) for pi, n in live), reverse=True
            )[:top]
            units = [
                {
                    "address": Address.neuron(layer, n, 1 if pi == 0 else -1).path(),
                    "title": self._fires(self._peak_ids(layer, pi, n)[0]),
                    "score": round(c, 4),
                    "strength": round(c, 3),
                }
                for c, _, pi, n in scored
            ]
            heads = [
                {
                    "address": Address.head(layer, h).path(),
                    "title": self.title(Address.head(layer, h)),
                    "score": num(s["head_out_top"][layer, h]),
                    "strength": 0.5,
                }
                for h in range(meta["n_heads"])
            ]
            layers.append(
                {
                    "layer": layer,
                    "band": band(layer, meta["n_layers"]),
                    "units": units,
                    "heads": heads,
                    "n_units": meta["d_mlp"],
                }
            )
        return {
            "model": slug,
            "hf_id": meta.get("hf_id"),
            "source": "computed",
            "sizes": {"n_layers": meta["n_layers"], "d_mlp": meta["d_mlp"], "n_heads": meta["n_heads"]},
            "labels": {"by": LABEL_BY, "lent_from": None},
            "corpus": meta["corpus"],
            "built": meta.get("built"),
            "ranking": "consistency: the share of a pole's top firings on its most common token (computed; "
            "near-silent poles left out). Not a label confidence: there are no labels.",
            "layers": layers,
        }

    def search_documents(
        self, slug: str, reads: bool = False, progress: Callable[[int, int], None] | None = None
    ) -> Iterator[tuple[Address, str, str, float]]:
        """Every firing pole, as its peak and driver tokens (and with `weights.npz`, its write and read tokens)."""
        self._load()
        meta = self.meta
        s = self.stats
        for layer in range(meta["n_layers"]):
            if progress:
                progress(layer, meta["n_layers"])
            for pi in range(2):
                for n in np.argwhere(s["act_top"][layer, pi] >= NEAR_SILENT).reshape(-1).tolist():
                    peaks, _ = self._peak_ids(layer, pi, n)
                    tokens = [self.text(t) for t in set(peaks)]
                    if self._drivers is not None:
                        tokens += [self.text(int(t)) for t in self._drivers["ids"][layer, pi, :TOP_TOKENS, n]]
                    if reads and self._weights is not None:
                        for key in ("write_ids", "write_z_ids", "read_ids"):
                            tokens += [self.text(int(t)) for t in self._weights[key][layer, pi, n]]
                    address = Address.neuron(layer, n, 1 if pi == 0 else -1)
                    yield address, self._fires(peaks), " ".join(tokens), self.consistency(layer, pi, n)
