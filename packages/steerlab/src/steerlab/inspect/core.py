"""The compute behind `steerlab inspect`: how a model reaches its next word, layer by layer, and what one neuron does.

    inspector = Inspector.load("Qwen/Qwen2.5-0.5B-Instruct")
    run = inspector.run("The capital of France is")          # guesses, target, one row per layer, the leftover
    inspector.layer("The capital of France is", layer=20)     # heads and neurons pushing toward / against the target
    inspector.neuron("The capital of France is", 20, 1639)    # words it promotes, how it fires on each token
    inspector.find(" Paris")                                  # neurons whose output writes " Paris"
    inspector.edit("The capital of France is", [Edit(20, 1639, "off")])   # normal vs edited, side by side
    inspector.dose("The capital of France is", 20, 1639)      # ×0 … ×8 against 20 random neurons of the same layer

The split. The residual stream at the last layer is a sum: the token embedding, plus every attention layer's output,
plus every MLP's output. Each attention output is a sum over heads (`z_h @ W_O_h`); each MLP output is a sum over
neurons (`activation_n × W_down[:, n]`). The final RMSNorm divides the whole stream by one number (its root mean
square) and multiplies by a fixed weight, then the unembedding reads the target's logit off with a dot product. With
that one number FROZEN at its real value for this position, the logit is linear in the stream, so it splits exactly
into one term per component: embed + Σ heads + Σ neurons. `leftover` is the real logit minus that sum (float error
only). Without freezing it the split is not exact: removing a component would also change the norm.

The logit lens ("after this layer it would say ___") applies the final norm and the unembedding to the stream after
each layer, as if the model stopped there. At the last layer it is the model's real output.

Edits change a neuron's activation (the input of `mlp.down_proj`) at every position, prompt and generated alike: off
(0), flip (−a), scale (k·a) or set (v). With no edits, every number is bit-for-bit the plain model's.

Works on Qwen2/Qwen3/Llama-style Hugging Face decoders (`model.model.layers[i].self_attn.o_proj`,
`.mlp.down_proj`, `model.model.norm`, `model.lm_head`); torch is the only heavy import.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 64
"""Longest prompt the inspector accepts, in tokens: enough for a sentence or two, small enough to stay instant."""
DOSE_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
EditMode = Literal["off", "flip", "scale", "set"]
EDIT_MODES: tuple[str, ...] = ("off", "flip", "scale", "set")


@dataclass(frozen=True)
class Edit:
    """Change neuron `neuron` of layer `layer` at every position: off, flip, scale by `value`, or set to `value`."""

    layer: int
    neuron: int
    mode: EditMode = "off"
    value: float = 0.0

    def apply(self, activation: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return torch.zeros_like(activation)
        if self.mode == "flip":
            return -activation
        if self.mode == "scale":
            return activation * self.value
        if self.mode == "set":
            return torch.full_like(activation, self.value)
        raise ValueError(f"unknown edit mode {self.mode!r}; choose one of {EDIT_MODES}")

    def label(self) -> str:
        name = f"L{self.layer}.N{self.neuron}"
        return {"off": f"{name} off", "flip": f"{name} flipped", "scale": f"{name} ×{self.value:g}"}.get(
            self.mode, f"{name} = {self.value:g}"
        )


@dataclass
class Trace:
    """One forward pass, captured: the stream after the embedding and after each layer, each layer's per-head inputs
    to `o_proj` (`z`), each layer's neuron activations, the logits and (when asked) the attention patterns."""

    ids: list[int]
    embed: torch.Tensor  # [seq, d_model]
    resid: list[torch.Tensor]  # n_layers × [seq, d_model], after each layer
    z: list[torch.Tensor]  # n_layers × [seq, n_heads * d_head]
    acts: list[torch.Tensor]  # n_layers × [seq, d_mlp]
    logits: torch.Tensor  # [seq, vocab]
    attention: list[torch.Tensor] | None = None  # n_layers × [n_heads, seq, seq]
    edits: list[Edit] = field(default_factory=list)


def _first(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output  # type: ignore[no-any-return,unused-ignore]


class Inspector:
    """A model, its tokenizer and the views `steerlab inspect` shows. Not thread-safe: callers hold one lock."""

    def __init__(self, model: Any, tokenizer: Any, name: str = "") -> None:
        if getattr(model.config, "_attn_implementation", "eager") != "eager":
            model.set_attn_implementation("eager")  # sdpa returns no attention patterns; the layer view shows them
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.name = name or str(getattr(model.config, "name_or_path", "model"))
        config = model.config
        self.blocks = model.model.layers
        self.n_layers = len(self.blocks)
        self.n_heads = int(config.num_attention_heads)
        self.d_model = int(config.hidden_size)
        self.d_head = int(getattr(config, "head_dim", None) or self.d_model // self.n_heads)
        self.d_mlp = int(config.intermediate_size)
        self.W_U = model.lm_head.weight.detach()  # [vocab, d_model]
        self.w_norm = model.model.norm.weight.detach()  # [d_model]
        self.eps = float(getattr(model.model.norm, "variance_epsilon", getattr(config, "rms_norm_eps", 1e-6)))
        self.vocab = int(self.W_U.shape[0])
        self._down_norms: torch.Tensor | None = None

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL, *, local_only: bool = True) -> Inspector:
        """Load a Hugging Face model on the CPU in float32 (eager attention, so patterns can be read)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, attn_implementation="eager", local_files_only=local_only
        )
        return cls(model, tokenizer, name=model_id)

    # Tokens

    def tokenize(self, text: str) -> list[int]:
        ids: list[int] = self.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError("type at least one word")
        if len(ids) > MAX_TOKENS:
            raise ValueError(f"the prompt is {len(ids)} tokens; the inspector takes at most {MAX_TOKENS}")
        return ids

    def token_text(self, token_id: int) -> str:
        return str(self.tokenizer.decode([int(token_id)]))

    def target_id(self, word: str) -> int:
        """The token a typed word means: its first token, preferring the spaced form (" Paris") when that is shorter
        or equally short, since a next word usually starts with a space."""
        if not word:
            raise ValueError("the target word is empty")
        forms = [word] if word[0].isspace() else [" " + word, word]
        encoded = [self.tokenizer(form, add_special_tokens=False).input_ids for form in forms]
        best = min((ids for ids in encoded if ids), key=len, default=None)
        if best is None:
            raise ValueError(f"{word!r} has no tokens")
        return int(best[0])

    def _pos(self, ids: Sequence[int], pos: int | None) -> int:
        n = len(ids)
        pos = n - 1 if pos is None else pos
        if not -n <= pos < n:
            raise ValueError(f"position {pos} is outside the prompt's {n} tokens")
        return pos % n

    def _check_neuron(self, layer: int, neuron: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is outside 0..{self.n_layers - 1}")
        if not 0 <= neuron < self.d_mlp:
            raise ValueError(f"neuron {neuron} is outside 0..{self.d_mlp - 1}")

    # Hooks

    @contextmanager
    def edited(self, edits: Sequence[Edit]) -> Iterator[None]:
        """Apply `edits` to every forward pass inside the block (generation included); hooks are removed after."""
        handles = []
        by_layer: dict[int, list[Edit]] = {}
        for edit in edits:
            self._check_neuron(edit.layer, edit.neuron)
            by_layer.setdefault(edit.layer, []).append(edit)
        for layer, layer_edits in by_layer.items():
            handles.append(self.blocks[layer].mlp.down_proj.register_forward_pre_hook(_edit_hook(layer_edits)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def trace(self, ids: Sequence[int], edits: Sequence[Edit] = (), *, attention: bool = False) -> Trace:
        """One forward pass with every piece of the split captured (after any edits)."""
        resid: list[Any] = [None] * self.n_layers
        z: list[Any] = [None] * self.n_layers
        acts: list[Any] = [None] * self.n_layers
        embed: list[Any] = [None]
        handles = [self.model.model.embed_tokens.register_forward_hook(_store(embed, 0, output=True))]
        for i, block in enumerate(self.blocks):
            handles.append(block.register_forward_hook(_store(resid, i, output=True)))
            handles.append(block.self_attn.o_proj.register_forward_pre_hook(_store(z, i)))
        with torch.no_grad(), self.edited(edits):
            # Registered after the edit hooks, so the captured activations are the edited ones.
            for i, block in enumerate(self.blocks):
                handles.append(block.mlp.down_proj.register_forward_pre_hook(_store(acts, i)))
            try:
                out = self.model(torch.tensor([list(ids)]), output_attentions=attention, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
        patterns = [a[0].detach() for a in out.attentions] if attention and out.attentions else None
        return Trace(
            ids=list(ids),
            embed=embed[0],
            resid=resid,
            z=z,
            acts=acts,
            logits=out.logits[0].detach(),
            attention=patterns,
            edits=list(edits),
        )

    # The pieces of the split

    def _norm_scale(self, x: torch.Tensor) -> torch.Tensor:
        """The final RMSNorm's 1/rms for the stream `x` (the number the split freezes)."""
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def readout(self, trace: Trace, pos: int, target: int) -> torch.Tensor:
        """The vector whose dot product with any piece of the stream at `pos` is that piece's share of the target
        logit: (1/rms frozen at this position) × norm weight × the target's unembedding row."""
        scale = self._norm_scale(trace.resid[-1][pos])
        return scale * self.w_norm * self.W_U[target]  # type: ignore[no-any-return,unused-ignore]

    def head_outputs(self, trace: Trace, layer: int, pos: int) -> torch.Tensor:
        """Each head's write into the stream at `pos`: [n_heads, d_model]."""
        z = trace.z[layer][pos].view(self.n_heads, self.d_head)
        W_O = self.blocks[layer].self_attn.o_proj.weight.detach().view(self.d_model, self.n_heads, self.d_head)
        return torch.einsum("hd,mhd->hm", z, W_O)

    def _attn_bias(self, layer: int) -> torch.Tensor | None:
        bias = self.blocks[layer].self_attn.o_proj.bias
        return None if bias is None else bias.detach()

    def down(self, layer: int) -> torch.Tensor:
        """Layer `layer`'s neuron output directions as rows: [d_mlp, d_model]."""
        return self.blocks[layer].mlp.down_proj.weight.detach().T  # type: ignore[no-any-return,unused-ignore]

    def neuron_lens(self, layer: int, neuron: int) -> torch.Tensor:
        """What one unit of the neuron's activation does to every word's logit (before the norm's 1/rms): [vocab]."""
        return self.W_U @ (self.w_norm * self.down(layer)[neuron])  # type: ignore[no-any-return,unused-ignore]

    def words(self, logits: torch.Tensor, k: int, *, lowest: bool = False) -> list[str]:
        order = torch.topk(logits, k, largest=not lowest).indices
        return [self.token_text(int(i)) for i in order]

    # Views

    def resolve(self, text: str, pos: int | None) -> tuple[list[int], int]:
        ids = self.tokenize(text)
        return ids, self._pos(ids, pos)

    def _target(self, trace: Trace, pos: int, target: str | int | None) -> int:
        if target is None or target == "":
            return int(trace.logits[pos].argmax())
        if isinstance(target, int):
            if not 0 <= target < self.vocab:
                raise ValueError(f"token id {target} is outside the vocabulary")
            return target
        return self.target_id(target)

    def run(
        self,
        text: str,
        target: str | int | None = None,
        pos: int | None = None,
        edits: Sequence[Edit] = (),
        k: int = 8,
    ) -> dict[str, Any]:
        """Guesses, the target, and one row per layer: attention's and the neurons' share of the target logit, and
        what the logit lens would say after that layer. Plus the embed share, the sum, the real logit and the leftover.
        """
        ids, pos = self.resolve(text, pos)
        trace = self.trace(ids, edits)
        return self._run_view(trace, pos, self._target(trace, pos, target), k)

    def _run_view(self, trace: Trace, pos: int, target: int, k: int) -> dict[str, Any]:
        logits = trace.logits[pos]
        probs = torch.softmax(logits, -1)
        top = torch.topk(probs, k)
        u = self.readout(trace, pos, target)
        embed_share = float(trace.embed[pos] @ u)
        streams = torch.stack([trace.embed[pos], *[r[pos] for r in trace.resid]])  # [n_layers + 1, d_model]
        lens_logits = (streams * self._norm_scale(streams) * self.w_norm) @ self.W_U.T  # [n_layers + 1, vocab]
        lens_probs = torch.softmax(lens_logits, -1)
        rows = []
        total = embed_share
        for layer in range(self.n_layers + 1):
            row: dict[str, Any] = {"layer": layer - 1}  # -1 is the embedding row
            if layer > 0:
                i = layer - 1
                attn = float(self.head_outputs(trace, i, pos).sum(0) @ u)
                bias = self._attn_bias(i)
                if bias is not None:
                    attn += float(bias @ u)
                mlp = float(trace.acts[i][pos] @ (self.down(i) @ u))
                total += attn + mlp
                row.update(attn=attn, mlp=mlp)
            lens_top = int(lens_logits[layer].argmax())
            row.update(
                says=self.token_text(lens_top),
                says_prob=float(lens_probs[layer, lens_top]),
                target_rank=int((lens_logits[layer] > lens_logits[layer, target]).sum()) + 1,
                target_prob=float(lens_probs[layer, target]),
            )
            rows.append(row)
        real = float(logits[target])
        return {
            "model": self.name,
            "tokens": [self.token_text(i) for i in trace.ids],
            "pos": pos,
            "guesses": [
                {"id": int(i), "text": self.token_text(int(i)), "prob": float(p), "logit": float(logits[i])}
                for p, i in zip(top.values, top.indices, strict=True)
            ],
            "target": {
                "id": target,
                "text": self.token_text(target),
                "prob": float(probs[target]),
                "logit": real,
                "rank": int((logits > logits[target]).sum()) + 1,
            },
            "embed": embed_share,
            "layers": rows,
            "sum": total,
            "leftover": real - total,
            "norm_scale": float(self._norm_scale(trace.resid[-1][pos])),
            "edits": [edit.label() for edit in trace.edits],
        }

    def layer(
        self, text: str, layer: int, target: str | int | None = None, pos: int | None = None, k: int = 8
    ) -> dict[str, Any]:
        """Layer `layer`'s heads (all of them, by share) and its `k` neurons pushing most toward and against the
        target, each with its activation, how well its output lines up with the word, and the words it promotes."""
        self._check_neuron(layer, 0)
        ids, pos = self.resolve(text, pos)
        trace = self.trace(ids, attention=True)
        target_id = self._target(trace, pos, target)
        u = self.readout(trace, pos, target_id)
        outs = self.head_outputs(trace, layer, pos)  # [n_heads, d_model]
        head_logits = (outs * self._norm_scale(trace.resid[-1][pos]) * self.w_norm) @ self.W_U.T
        tokens = [self.token_text(i) for i in ids]
        heads: list[dict[str, Any]] = []
        for h in range(self.n_heads):
            attends = []
            if trace.attention is not None:
                pattern = trace.attention[layer][h, pos, : pos + 1]
                weights, where = torch.topk(pattern, min(2, pos + 1))
                for weight, j in zip(weights, where, strict=True):
                    attends.append({"pos": int(j), "text": tokens[int(j)], "weight": float(weight)})
            heads.append(
                {"head": h, "share": float(outs[h] @ u), "promotes": self.words(head_logits[h], 3), "attends": attends}
            )
        heads.sort(key=lambda row: -abs(float(row["share"])))
        align = self.down(layer) @ u  # [d_mlp]: share per unit of activation
        acts = trace.acts[layer][pos]
        shares = acts * align
        toward = torch.topk(shares, k).indices
        against = torch.topk(-shares, k).indices
        return {
            "layer": layer,
            "pos": pos,
            "target": {"id": target_id, "text": self.token_text(target_id)},
            "attn": float(outs.sum(0) @ u),
            "mlp": float(shares.sum()),
            "heads": heads,
            "toward": [self._neuron_row(layer, int(n), acts, align) for n in toward],
            "against": [self._neuron_row(layer, int(n), acts, align) for n in against],
        }

    def _neuron_row(self, layer: int, n: int, acts: torch.Tensor, align: torch.Tensor) -> dict[str, Any]:
        return {
            "id": f"L{layer}.N{n}",
            "layer": layer,
            "neuron": n,
            "act": float(acts[n]),
            "align": float(align[n]),
            "share": float(acts[n] * align[n]),
            "promotes": self.words(self.neuron_lens(layer, n), 3),
        }

    def neuron(
        self, text: str, layer: int, neuron: int, target: str | int | None = None, pos: int | None = None, k: int = 10
    ) -> dict[str, Any]:
        """One neuron: the words it promotes and suppresses, how strongly it fires on each token, its share here."""
        self._check_neuron(layer, neuron)
        ids, pos = self.resolve(text, pos)
        trace = self.trace(ids)
        target_id = self._target(trace, pos, target)
        u = self.readout(trace, pos, target_id)
        lens = self.neuron_lens(layer, neuron)
        acts = trace.acts[layer][:, neuron]
        all_shares = trace.acts[layer][pos] * (self.down(layer) @ u)
        share = float(all_shares[neuron])
        return {
            "id": f"L{layer}.N{neuron}",
            "layer": layer,
            "neuron": neuron,
            "pos": pos,
            "target": {"id": target_id, "text": self.token_text(target_id)},
            "promotes": self.words(lens, k),
            "suppresses": self.words(lens, k, lowest=True),
            "fires": [{"text": self.token_text(i), "act": float(a)} for i, a in zip(ids, acts, strict=True)],
            "act": float(acts[pos]),
            "align": float(self.down(layer)[neuron] @ u),
            "share": share,
            "rank_in_layer": int((all_shares.abs() > abs(share)).sum()) + 1,
        }

    def find(self, word: str | int, text: str | None = None, pos: int | None = None, k: int = 8) -> dict[str, Any]:
        """The neurons whose output direction writes `word` most: cosine of each neuron's output with the word's
        readout (norm weight × unembedding row), across every layer. `toward` write it when they fire positive,
        `against` when they fire negative (SiLU-gated neurons fire both ways). With `text`, each row also gets the
        neuron's activation at `pos` and its share of that word's logit there."""
        target = word if isinstance(word, int) else self.target_id(word)
        v = self.w_norm * self.W_U[target]
        v = v / v.norm()
        if self._down_norms is None:
            self._down_norms = torch.stack([self.down(i).norm(dim=-1) for i in range(self.n_layers)])
        cos = torch.stack([self.down(i) @ v for i in range(self.n_layers)]) / self._down_norms  # [n_layers, d_mlp]
        flat = cos.flatten()
        trace: Trace | None = None
        u: torch.Tensor | None = None
        at = 0
        if text:
            ids, at = self.resolve(text, pos)
            trace = self.trace(ids)
            u = self.readout(trace, at, target)

        def rows(indices: torch.Tensor) -> list[dict[str, Any]]:
            out = []
            for index in indices:
                layer, n = divmod(int(index), self.d_mlp)
                row: dict[str, Any] = {
                    "id": f"L{layer}.N{n}",
                    "layer": layer,
                    "neuron": n,
                    "cos": float(flat[index]),
                    "promotes": self.words(self.neuron_lens(layer, n), 3),
                }
                if trace is not None and u is not None:
                    act = float(trace.acts[layer][at, n])
                    row.update(act=act, share=act * float(self.down(layer)[n] @ u))
                out.append(row)
            return out

        return {
            "word": {"id": target, "text": self.token_text(target)},
            "toward": rows(torch.topk(flat, k).indices),
            "against": rows(torch.topk(-flat, k).indices),
        }

    def generate(self, ids: Sequence[int], edits: Sequence[Edit] = (), max_new_tokens: int = 12) -> list[int]:
        """Greedy continuation (repetition_penalty=1.0: Qwen's generation config sets 1.1, which greedy alone keeps)."""
        pad = self.tokenizer.pad_token_id
        pad = self.tokenizer.eos_token_id if pad is None else pad
        with torch.no_grad(), self.edited(edits):
            out = self.model.generate(
                torch.tensor([list(ids)]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad,
            )
        return [int(t) for t in out[0, len(ids) :]]

    def edit(
        self,
        text: str,
        edits: Sequence[Edit],
        target: str | int | None = None,
        pos: int | None = None,
        max_new_tokens: int = 12,
    ) -> dict[str, Any]:
        """The run view and a greedy continuation, normal and with `edits`, side by side. The target is fixed by the
        normal run, so both columns score the same word."""
        ids, pos = self.resolve(text, pos)
        normal_trace = self.trace(ids)
        target_id = self._target(normal_trace, pos, target)
        normal = self._run_view(normal_trace, pos, target_id, 8)
        edited = self._run_view(self.trace(ids, edits), pos, target_id, 8)
        context = ids[: pos + 1]
        return {
            "edits": [edit.label() for edit in edits],
            "normal": normal,
            "edited": edited,
            "normal_text": self.tokenizer.decode(self.generate(context, (), max_new_tokens)),
            "edited_text": self.tokenizer.decode(self.generate(context, edits, max_new_tokens)),
        }

    def dose(
        self,
        text: str,
        layer: int,
        neuron: int,
        target: str | int | None = None,
        pos: int | None = None,
        scales: Sequence[float] = DOSE_SCALES,
        n_random: int = 20,
        seed: int = 0,
    ) -> dict[str, Any]:
        """The target's probability as the neuron is scaled ×0 … ×8 at every position, and the same for `n_random`
        random neurons of the same layer (the control band: does this neuron matter more than any neuron would?)."""
        self._check_neuron(layer, neuron)
        ids, pos = self.resolve(text, pos)
        base = self.trace(ids)
        target_id = self._target(base, pos, target)
        others = [n for n in range(self.d_mlp) if n != neuron]
        controls = random.Random(seed).sample(others, min(n_random, len(others)))
        chosen = torch.tensor([neuron, *controls])
        batch = torch.tensor([list(ids)] * len(chosen))
        rows = torch.arange(len(chosen))
        curves: list[list[float]] = [[] for _ in range(len(chosen))]
        for scale in scales:
            handle = self.blocks[layer].mlp.down_proj.register_forward_pre_hook(_scale_hook(rows, chosen, scale))
            try:
                with torch.no_grad():
                    logits = self.model(batch, use_cache=False).logits[:, pos]
            finally:
                handle.remove()
            probs = torch.softmax(logits, -1)[:, target_id]
            for curve, p in zip(curves, probs, strict=True):
                curve.append(float(p))
        band = list(zip(*curves[1:], strict=True))
        return {
            "id": f"L{layer}.N{neuron}",
            "layer": layer,
            "neuron": neuron,
            "pos": pos,
            "target": {"id": target_id, "text": self.token_text(target_id)},
            "base_prob": float(torch.softmax(base.logits[pos], -1)[target_id]),
            "act": float(base.acts[layer][pos, neuron]),
            "scales": list(scales),
            "curve": curves[0],
            "random_ids": [f"L{layer}.N{n}" for n in controls],
            "random": curves[1:],
            "band_lo": [min(column) for column in band],
            "band_hi": [max(column) for column in band],
        }


def _store(slots: list[Any], i: int, *, output: bool = False) -> Callable[..., None]:
    """A hook that keeps batch row 0 of a module's input (pre-hook) or output (forward hook) in `slots[i]`."""

    def pre(module: Any, args: tuple[Any, ...]) -> None:
        slots[i] = args[0][0].detach().clone()

    def post(module: Any, args: tuple[Any, ...], out: Any) -> None:
        slots[i] = _first(out)[0].detach().clone()

    return post if output else pre


def _edit_hook(edits: Sequence[Edit]) -> Callable[..., tuple[Any, ...]]:
    def hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        acts = args[0].clone()
        for edit in edits:
            acts[..., edit.neuron] = edit.apply(acts[..., edit.neuron])
        return (acts, *args[1:])

    return hook


def _scale_hook(rows: torch.Tensor, neurons: torch.Tensor, scale: float) -> Callable[..., tuple[Any, ...]]:
    """Batch row r scales neuron `neurons[r]` (every position) by `scale`: one forward pass doses many neurons."""

    def hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        acts = args[0].clone()
        acts[rows, :, neurons] = acts[rows, :, neurons] * scale
        return (acts, *args[1:])

    return hook
