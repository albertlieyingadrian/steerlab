"""The bridge to circuit-tracer: a validation backend over a `ReplacementModel`, and reconstruction error.

Needs `torch` and `circuit-tracer` (install circuit-tracer from GitHub). Nothing else in circuitkit imports this
module, so the rest of the package works without either.

    from circuit_tracer import ReplacementModel
    from circuitkit.tracer import TracerBackend, reconstruction_error

    model = ReplacementModel.from_pretrained("Qwen/Qwen3-1.7B", "mwhanna/qwen3-1.7b-transcoders-lowl0")
    backend = TracerBackend(model)              # pass to circuitkit.validate.validate(...)
    fvu = reconstruction_error(model, prompts)  # per-layer FVU, for circuitkit.diff's validity check
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from circuitkit.validate.backend import ActiveFeature
from circuitkit.validate.types import FeatureNode, Intervention


class TracerBackend:
    """`InterventionBackend` over a circuit-tracer `ReplacementModel`.

    `freeze_attention=False` (the default) lets attention patterns respond to an intervention, so a feature's effect
    can travel through later layers as it would in the real model. Set it to True to match circuit-tracer's own
    default, which measures direct effects through frozen attention.
    """

    def __init__(self, model: Any, freeze_attention: bool = False) -> None:
        self.model = model
        self.freeze_attention = freeze_attention
        self._ids: dict[str, Any] = {}
        self._active: dict[str, list[ActiveFeature]] = {}

    def register(self, name: str, token_ids: Sequence[int]) -> str:
        """Use exact token ids for a prompt: pass `name` wherever circuitkit asks for the prompt."""
        import torch

        self._ids[name] = torch.tensor(list(token_ids), dtype=torch.long)
        return name

    def _tok(self, prompt: str) -> Any:
        if prompt not in self._ids:
            self._ids[prompt] = self.model.ensure_tokenized(prompt)
        return self._ids[prompt]

    def tokens(self, prompt: str) -> list[str]:
        ids = self._tok(prompt).tolist()
        return [str(t) for t in self.model.tokenizer.convert_ids_to_tokens(ids)]

    def token_id(self, token: str) -> int:
        ids = self.model.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens, not one")
        return int(ids[0])

    def logits(self, prompt: str, interventions: Sequence[Intervention] = ()) -> np.ndarray:
        import torch

        ids = self._tok(prompt)
        with torch.inference_mode():
            if interventions:
                out, _ = self.model.feature_intervention(
                    ids,
                    [(int(a), int(b), int(c), float(v)) for a, b, c, v in interventions],
                    freeze_attention=self.freeze_attention,
                    return_activations=False,
                )
            else:
                out = self.model(ids.unsqueeze(0))
        arr: np.ndarray = out.reshape(-1, out.shape[-1]).float().cpu().numpy()
        return arr

    def active_features(self, prompt: str) -> list[ActiveFeature]:
        if prompt not in self._active:
            _, acts = self.model.get_activations(self._tok(prompt), sparse=True)
            acts = acts.coalesce()
            idx = acts.indices().cpu().tolist()
            vals = acts.values().float().cpu().tolist()
            self._active[prompt] = [(idx[0][i], idx[1][i], idx[2][i], vals[i]) for i in range(len(vals))]
        return self._active[prompt]

    def decoder_norm(self, layer: int, feature: int) -> float:
        import torch

        tc = self.model.transcoders
        with torch.inference_mode():
            vec = tc._get_decoder_vectors(layer, torch.tensor([feature], device=self.model.cfg.device))
        return float(vec.float().norm())

    def describe(self) -> dict[str, str]:
        tc = self.model.transcoders
        return {
            "model": str(getattr(self.model.cfg, "model_name", "?")),
            "transcoders": str(getattr(tc, "scan_name", type(tc).__name__)),
            "device": str(self.model.cfg.device),
            "freeze_attention": str(self.freeze_attention),
        }


def nodes_for_units(
    backend: Any,
    prompt: str,
    units: Sequence[tuple[int, int]],
    positions: str | Sequence[int] = "all",
) -> list[FeatureNode]:
    """FeatureNodes for (layer, feature) units on `prompt`: at every position where they are active
    (`positions="all"`), only at the last position (`"last"`), or at the given positions. Position 0 is skipped."""
    act = backend.active_features(prompt)
    n = len(backend.tokens(prompt))
    want = set(units)
    keep = None if isinstance(positions, str) else set(positions)
    out = []
    for layer, pos, feat, a in act:
        if (layer, feat) not in want or pos == 0:
            continue
        if positions == "last" and pos != n - 1:
            continue
        if keep is not None and pos not in keep:
            continue
        out.append(FeatureNode(layer=layer, pos=pos, feature=feat, activation=a))
    return sorted(out, key=lambda x: (x.layer, x.feature, x.pos))


def reconstruction_error(model: Any, prompts: Sequence[str], skip_first: int = 1) -> list[float]:
    """Per-layer fraction of variance unexplained (FVU) of the transcoders' MLP-output reconstruction.

    FVU = sum ||mlp_out - recon||^2 / sum ||mlp_out - mean||^2 over every token of every prompt after the first
    `skip_first` positions (circuit-tracer zeroes position 0). 0 is perfect. Values that rise after fine-tuning mean
    the base-model transcoders describe the trained model worse.
    """
    import torch

    n_layers = int(model.cfg.n_layers)
    in_names = [f"blocks.{i}.{model.feature_input_hook}" for i in range(n_layers)]
    out_hook = getattr(model, "original_feature_output_hook", model.feature_output_hook)
    out_names = [f"blocks.{i}.{out_hook}" for i in range(n_layers)]
    wanted = set(in_names) | set(out_names)
    sse = np.zeros(n_layers)
    sq = np.zeros(n_layers)
    sums: list[Any] = [None] * n_layers
    count = 0
    for p in prompts:
        ids = model.ensure_tokenized(p)
        with torch.inference_mode():
            _, cache = model.run_with_cache(ids.unsqueeze(0), names_filter=lambda n: n in wanted)
            x = torch.stack([cache[n][0] for n in in_names])  # (layers, pos, d_model)
            y = torch.stack([cache[n][0] for n in out_names]).double()
            recon = model.transcoders.decode(model.transcoders.encode(x), x).double()
        y, recon = y[:, skip_first:], recon[:, skip_first:]
        count += int(y.shape[1])
        for i in range(n_layers):
            sse[i] += float(((y[i] - recon[i]) ** 2).sum())
            sq[i] += float((y[i] ** 2).sum())
            s = y[i].sum(0).cpu()
            sums[i] = s if sums[i] is None else sums[i] + s
    out = []
    for i in range(n_layers):
        sst = sq[i] - float((sums[i] ** 2).sum()) / max(count, 1)
        out.append(float(sse[i] / sst) if sst > 0 else float("nan"))
    return out
