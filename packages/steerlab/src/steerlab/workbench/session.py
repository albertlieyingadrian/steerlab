"""The Workbench: one loaded model you chat with under steering, with readings for every token it writes.

It is steerlab's steering engine with instruments bolted on. The steering is exactly `steerlab chat`'s: a
`ControlState` resolved by a `DirectionRegistry`, hooked in by a `SteeringSession`, applied to generated positions
only. What the Workbench adds, per generated token:

- the alternatives (top 5 next tokens with log-probabilities), so the loom can fork from any of them;
- probe readings: each active concept probe's z-score at its layer, next to its random-direction band;
- the top contributing units: the neurons and heads whose direct write pushed the chosen token's logit most
  (activation × write · readout, with the final norm's scale frozen at this position, like `steerlab inspect`);
- the steering strength: at each steered layer, the norm added over the median residual norm.

    bench = Workbench(model, tokenizer, "Qwen/Qwen2.5-0.5B-Instruct", cache_dir)
    bench.fit_bundle()                                        # first launch: fit and cache the bundled probes
    reply = bench.generate([{"role": "user", "content": "Hi"}], steer="4 probe:french.english")
    reply["text"], reply["tokens"][0]["probes"]["formal.casual"]

Steering is written three ways, all turned into control-file entries:
- control entries: `{"source": {"kind": "direction", "name": "french"}, "strength": 4, "layers": [14]}`;
- shorthand: `{"direction": "french", "strength": 4}`, `{"probe": "formal.casual", "strength": -2}`,
  `{"sae": {"release": ..., "layer": 14, "feature": 1234}, "strength": 3, "mode": "clamp"}`;
- an expression: `"0.5 formal.casual - 2 probe:sycophantic.blunt@10,12"` (a bare name is a probe when one has that
  name, else a loaded direction; `@` lists the layers).

Strength 1 on a probe direction adds one class gap (the distance between the two poles' means), the same unit as
steerlab's difference-of-means directions. Sampling at temperature 0 is greedy decoding, token for token
the same as `steerlab.decode.greedy_generate` (tested).

Not thread-safe: the caller holds the model lock for the whole reply.
"""

from __future__ import annotations

import importlib
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from steerlab.control import ControlEntry, ControlFile
from steerlab.workbench import edits as edit_ops
from steerlab.workbench.artifacts import ArtifactError, ArtifactStore
from steerlab.workbench.loom import LoomStore
from steerlab.workbench.probes import ConceptSpec, Probe, ProbeBank, default_layer, fit_concept, load_bundle

PROBE_PREFIX = "probe:"
MAX_NEW_TOKENS = 512
TOP_ALTERNATIVES = 5
TOP_UNITS = 8
TERM = re.compile(r"\s*([+-])?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)?\s*\*?\s*([A-Za-z_][\w.:/-]*)(?:@([\d,\s-]+))?")


class SteerError(ValueError):
    """A steering request that cannot be turned into control entries."""


@dataclass
class Sampling:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: tuple[str, ...] = ()


def parse_expression(text: str) -> list[dict[str, Any]]:
    """`"0.5 formal.casual - 2 sycophantic.blunt@10,12"` -> shorthand entries (name, strength, optional layers)."""
    out: list[dict[str, Any]] = []
    rest = text.strip()
    while rest:
        match = TERM.match(rest)
        if match is None or not match.group(0).strip():
            raise SteerError(f"cannot read the steering expression at {rest!r}")
        sign, number, name, layers = match.groups()
        strength = float(number) if number else 1.0
        if sign == "-":
            strength = -strength
        entry: dict[str, Any] = {"name": name, "strength": strength}
        if layers:
            entry["layers"] = _layer_list(layers)
        out.append(entry)
        rest = rest[match.end() :].strip()
    return out


def _layer_list(text: str) -> list[int]:
    layers: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    return layers


class Workbench:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        cache_dir: str | Path,
        *,
        probe_layer: int | None = None,
        on_message: Callable[[str], None] | None = None,
    ) -> None:
        from sandbagkit.probing.hooks import num_layers

        from steerlab.control import ControlState
        from steerlab.engine import DirectionRegistry

        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.cache_dir = Path(cache_dir).expanduser()
        self.n_layers = num_layers(model)
        self.probe_layer = default_layer(self.n_layers) if probe_layer is None else probe_layer
        self.messages: list[str] = []
        self._on_message = on_message
        self.registry = DirectionRegistry(model=model_id)
        self.state = ControlState(model=model_id, n_layers=self.n_layers, resolver=self.registry, on_message=self._say)
        self.artifacts = ArtifactStore(self.cache_dir / "artifacts")
        self.looms = LoomStore(self.cache_dir / "looms")
        self.bank = ProbeBank(self.artifacts, model_id)
        self.fit_status: dict[str, Any] = {"state": "idle", "concept": "", "done": 0, "total": 0, "error": None}
        self._cancel = threading.Event()
        self.bank.load_cached()
        self.refresh_directions()

    def _say(self, message: str) -> None:
        self.messages.append(message)
        del self.messages[:-50]
        if self._on_message is not None:
            self._on_message(message)

    def _model_type(self) -> str:
        return str(getattr(self.model.config, "model_type", "") or "")

    def _d_model(self) -> int:
        return int(self.model.config.hidden_size)

    # What can be steered

    def refresh_directions(self) -> None:
        """Register every probe (as `probe:<name>`) and every saved direction artifact of this model."""
        import torch

        from steerlab.saes.common import FeatureDirection

        for name, probe in self.bank.probes.items():
            self.registry.add_direction(
                PROBE_PREFIX + name,
                FeatureDirection(
                    unit=torch.as_tensor(probe.direction),
                    layer=probe.layer,
                    release="steerlab.workbench.probe",
                    feature=0,
                    label=f"probe {name}",
                    scale=probe.class_gap,
                ),
            )
        for meta in self.artifacts.list("direction"):
            if meta.get("model") not in (None, self.model_id):
                continue
            try:
                _, arrays = self.artifacts.load("direction", meta["name"])
            except ArtifactError:
                continue
            vector = arrays.get("vector", arrays.get("direction"))
            if vector is None or vector.shape[-1] != self._d_model():
                continue
            unit = torch.as_tensor(np.asarray(vector, dtype=np.float32).reshape(-1))
            norm = float(unit.norm())
            if norm == 0:
                continue
            self.registry.add_direction(
                meta["name"],
                FeatureDirection(
                    unit=unit / norm,
                    layer=int(meta.get("layer", self.probe_layer)),
                    release="steerlab.workbench.artifact",
                    feature=0,
                    label=meta["name"],
                    scale=float(meta["scale"]) if meta.get("scale") is not None else None,
                ),
            )

    def steerables(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "kind": "probe" if name.startswith(PROBE_PREFIX) else "direction",
                "layer": direction.layer,
                "scale": direction.scale,
                "label": direction.label,
            }
            for name, direction in sorted(self.registry.directions.items())
        ]

    def control_entries(self, steer: str | Sequence[Mapping[str, Any]] | None) -> list[ControlEntry]:
        """Turn any of the three steering forms into validated control entries."""
        if not steer:
            return []
        items: Sequence[Mapping[str, Any]] = parse_expression(steer) if isinstance(steer, str) else steer
        entries = []
        for item in items:
            raw = dict(item)
            if "source" not in raw:
                raw = self._shorthand(raw)
            try:
                entries.append(ControlEntry.model_validate(raw))
            except Exception as error:
                raise SteerError(f"bad steering entry {item!r}: {error}") from None
        return entries

    def _shorthand(self, item: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {"strength": float(item.get("strength", 1.0))}
        if item.get("layers") is not None:
            entry["layers"] = list(item["layers"])
        if item.get("mode") is not None:
            entry["mode"] = item["mode"]
        if "sae" in item:
            entry["source"] = {"kind": "sae", **dict(item["sae"])}
            return entry
        if "probe" in item:
            name = PROBE_PREFIX + str(item["probe"]).removeprefix(PROBE_PREFIX)
        elif "direction" in item:
            name = str(item["direction"])
        elif "name" in item:
            name = str(item["name"])
            if name not in self.registry.directions and PROBE_PREFIX + name in self.registry.directions:
                name = PROBE_PREFIX + name
        else:
            raise SteerError(f"a steering entry needs source, direction, probe, sae or name: {item!r}")
        if name not in self.registry.directions:
            known = ", ".join(sorted(self.registry.directions)) or "none"
            raise SteerError(f"nothing named {name!r} to steer with (loaded: {known})")
        entry["source"] = {"kind": "direction", "name": name}
        return entry

    def resolve_edits(self, edits: Sequence[Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]]:
        """Edit artifacts given inline (dicts with `type: edit`) or by name ("edit/<name>" or "<name>")."""
        metas: list[dict[str, Any]] = []
        arrays: list[dict[str, Any] | None] = []
        for item in edits or []:
            if isinstance(item, str):
                meta, arr = self.artifacts.load("edit", item.removeprefix("edit/"))
                metas.append(meta)
                arrays.append(arr)
            elif isinstance(item, Mapping):
                if item.get("type", "edit") != "edit":
                    raise edit_ops.EditError(f"expected an edit artifact, got type {item.get('type')!r}")
                metas.append(dict(item))
                arrays.append(None)
            else:
                raise edit_ops.EditError(f"an edit is an artifact dict or its name, not {type(item).__name__}")
        return metas, arrays

    # Probes

    def fit_bundle(self, bundle: Sequence[ConceptSpec] | None = None) -> list[str]:
        def progress(name: str, done: int, total: int) -> None:
            self.fit_status.update(state="fitting" if name else "ready", concept=name, done=done, total=total)

        self.fit_status.update(state="fitting", error=None)
        try:
            fitted = self.bank.ensure_bundle(
                self.model, self.tokenizer, layer=self.probe_layer, on_progress=progress, bundle=bundle
            )
        except Exception as error:
            self.fit_status.update(state="error", error=str(error))
            raise
        self.refresh_directions()
        return fitted

    def fit_custom(self, spec: ConceptSpec, layer: int | None = None) -> Probe:
        probe = fit_concept(
            self.model, self.tokenizer, spec, model_id=self.model_id, layer=self.probe_layer if layer is None else layer
        )
        self.bank.add(probe)
        self.refresh_directions()
        return probe

    def bundle_names(self) -> list[str]:
        return [spec.name for spec in load_bundle()]

    # Generation

    def cancel(self) -> None:
        self._cancel.set()

    def render(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        """Token ids of the conversation with the generation prompt, through the chat template when there is one."""
        from steerlab.decode import render_messages

        msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        if getattr(self.tokenizer, "chat_template", None):
            text = render_messages(self.tokenizer, msgs, model_type=self._model_type())
        else:
            text = "".join(f"{m['role'].title()}: {m['content']}\n" for m in msgs) + "Assistant:"
        ids: list[int] = self.tokenizer(text, add_special_tokens=False).input_ids
        return ids

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        steer: str | Sequence[Mapping[str, Any]] | None = None,
        edits: Sequence[Any] | None = None,
        probes: Sequence[str] | None = None,
        sampling: Sampling | None = None,
        units: bool = True,
        on_token: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """One reply to `messages`. Returns its text, per-token records and what produced it."""
        from steerlab.engine import SteeringSession

        sampling = sampling or Sampling()
        if not 1 <= sampling.max_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_tokens must be 1..{MAX_NEW_TOKENS}")
        entries = self.control_entries(steer)
        edit_metas, edit_arrays = self.resolve_edits(edits)
        probe_names = list(self.bank.probes) if probes is None else [p for p in probes if p in self.bank.probes]
        probe_layers = sorted({self.bank.probes[n].layer for n in probe_names})
        prompt_ids = self.render(messages)
        if not prompt_ids:
            raise ValueError("the conversation renders to no tokens")
        self.state.forget_messages()
        before = len(self.messages)
        self.state.apply(ControlFile(model=self.model_id, entries=entries))
        self._cancel.clear()
        started = time.time()
        try:
            with (
                edit_ops.applied(self.model, edit_metas, edit_arrays, tokenizer=self.tokenizer) as edit_labels,
                SteeringSession(self.model, self.tokenizer, self.state) as session,
                _Instruments(self, probe_layers, units=units) as instruments,
            ):
                applied = [entry.to_dict() for entry in self.state.applied]
                result = self._decode(session, instruments, prompt_ids, probe_names, sampling, on_token)
        finally:
            self.state.clear()
        result.update(
            prompt_tokens=len(prompt_ids),
            steer=applied,
            steer_request=[entry.model_dump() for entry in entries],
            edits=edit_labels,
            edit_names=[str(meta.get("name", "")) for meta in edit_metas],
            sampling={
                "max_tokens": sampling.max_tokens,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "seed": sampling.seed,
            },
            probes=probe_names,
            messages=self.messages[before:],
            seconds=round(time.time() - started, 3),
            model=self.model_id,
        )
        return result

    def _decode(
        self,
        session: Any,
        instruments: _Instruments,
        prompt_ids: list[int],
        probe_names: list[str],
        sampling: Sampling,
        on_token: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        """The decode loop of `steerlab.decode.greedy_generate`, with sampling and the instruments read each step."""
        import torch
        from transformers import DynamicCache

        from steerlab.decode import _eos_ids

        model = self.model
        eos = _eos_ids(session)
        cache = DynamicCache(config=model.config)
        device = session.device
        input_ids = torch.tensor([prompt_ids], device=device)
        session.prompt_length = len(prompt_ids)
        generator = None
        if sampling.temperature > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(sampling.seed) if sampling.seed is not None else time.time_ns() % 2**63)
        generated: list[int] = []
        tokens: list[dict[str, Any]] = []
        text = ""
        finish = "length"
        with torch.no_grad():
            for index in range(sampling.max_tokens):
                if self._cancel.is_set():
                    finish = "cancelled"
                    break
                session.position_offset = cache.get_seq_length()
                session.last_stats = {}
                output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
                logits = output.logits[0, -1].float().cpu()
                next_id = _choose(logits, sampling, generator)
                if next_id in eos:
                    finish = "stop"
                    break
                record = self._record(index, next_id, logits, session, instruments, probe_names)
                generated.append(next_id)
                full = self.tokenizer.decode(generated, skip_special_tokens=True)
                if not full.endswith("�"):  # an incomplete multi-byte character: wait for the next token
                    record["text"] = full[len(text) :]
                    text = full
                tokens.append(record)
                if on_token is not None:
                    on_token(record)
                hit = next((s for s in sampling.stop if s and s in text), None)
                if hit is not None:
                    text = text[: text.index(hit)]
                    finish = "stop"
                    break
                input_ids = torch.tensor([[next_id]], device=device)
        return {"text": text, "token_ids": generated, "tokens": tokens, "finish_reason": finish}

    def _record(
        self,
        index: int,
        token_id: int,
        logits: Any,
        session: Any,
        instruments: _Instruments,
        probe_names: list[str],
    ) -> dict[str, Any]:
        import torch

        logprobs = torch.log_softmax(logits, dim=-1)
        top = torch.topk(logprobs, min(TOP_ALTERNATIVES, logprobs.shape[0]))
        readings = self.bank.read(instruments.last_residuals(), probe_names)
        return {
            "i": index,
            "id": token_id,
            "text": "",
            "token": self.tokenizer.decode([token_id]),
            "logprob": round(float(logprobs[token_id]), 4),
            "alternatives": [
                {"id": int(i), "text": self.tokenizer.decode([int(i)]), "logprob": round(float(lp), 4)}
                for lp, i in zip(top.values, top.indices, strict=True)
            ],
            "steer": {
                str(layer): {
                    "added": round(added, 4),
                    "residual": round(resid, 4),
                    "ratio": round(added / resid, 4) if resid else None,
                }
                for layer, (added, resid) in session.last_stats.items()
            },
            "probes": {name: round(r["z"], 3) for name, r in readings.items()},
            "bands": {name: round(r["band"], 3) for name, r in readings.items()},
            "units": instruments.top_units(token_id) if instruments.units else [],
        }

    def read_text(
        self, text: str, *, prompt: str = "Say something.", probes: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Probe readings on every token of `text`, read as the assistant's reply to `prompt` (the Probes room)."""
        if not text.strip():
            raise ValueError("type some text to read")
        head = self.render([{"role": "user", "content": prompt}])
        ids: list[int] = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > MAX_NEW_TOKENS:
            raise ValueError(f"the text is {len(ids)} tokens; at most {MAX_NEW_TOKENS}")
        import torch

        names = list(self.bank.probes) if probes is None else [p for p in probes if p in self.bank.probes]
        layers = sorted({self.bank.probes[n].layer for n in names})
        with _Instruments(self, layers, units=False) as instruments, torch.no_grad():
            device = next(self.model.parameters()).device
            self.model(input_ids=torch.tensor([head + ids], device=device), use_cache=False)
            h = {
                layer: captured[0, len(head) :].float().cpu().numpy()
                for layer, captured in instruments.residuals.items()
            }
        readings = self.bank.read(h, names)
        return {
            "tokens": [self.tokenizer.decode([i]) for i in ids],
            "probes": {name: [round(z, 3) for z in r["z"]] for name, r in readings.items()},
            "bands": {name: [round(b, 3) for b in r["band"]] for name, r in readings.items()},
        }

    def replay(
        self,
        messages: Sequence[Mapping[str, str]],
        token_ids: Sequence[int],
        *,
        steer: str | Sequence[Mapping[str, Any]] | None = None,
        edits: Sequence[Any] | None = None,
        probes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Teacher-force a reply's tokens (no cache) and read the probes at each position, with or without steering.

        The Inspector's "replay without steering": the same tokens, read by the unsteered model.
        """
        import torch

        from steerlab.engine import SteeringSession

        if not token_ids:
            return {"probes": {}, "bands": {}}
        entries = self.control_entries(steer)
        edit_metas, edit_arrays = self.resolve_edits(edits)
        probe_names = list(self.bank.probes) if probes is None else [p for p in probes if p in self.bank.probes]
        layers = sorted({self.bank.probes[n].layer for n in probe_names})
        prompt_ids = self.render(messages)
        self.state.apply(ControlFile(model=self.model_id, entries=entries))
        try:
            with (
                edit_ops.applied(self.model, edit_metas, edit_arrays, tokenizer=self.tokenizer),
                SteeringSession(self.model, self.tokenizer, self.state) as session,
                _Instruments(self, layers, units=False) as instruments,
                torch.no_grad(),
            ):
                session.prompt_length = len(prompt_ids)
                session.position_offset = 0
                ids = torch.tensor([prompt_ids + list(token_ids)], device=session.device)
                self.model(input_ids=ids, use_cache=False)
                start = len(prompt_ids) - 1
                h = {
                    layer: captured[0, start : start + len(token_ids)].float().cpu().numpy()
                    for layer, captured in instruments.residuals.items()
                }
        finally:
            self.state.clear()
        readings = self.bank.read(h, probe_names)
        return {
            "probes": {name: [round(z, 3) for z in r["z"]] for name, r in readings.items()},
            "bands": {name: [round(b, 3) for b in r["band"]] for name, r in readings.items()},
        }


def _choose(logits: Any, sampling: Sampling, generator: Any) -> int:
    """Greedy at temperature 0 (argmax, as `greedy_generate`), else temperature + nucleus sampling."""
    import torch

    if sampling.temperature <= 0 or generator is None:
        return int(logits.argmax())
    probs = torch.softmax(logits / sampling.temperature, dim=-1)
    if sampling.top_p < 1.0:
        sorted_probs, order = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < sampling.top_p
        keep[0] = True
        probs = torch.zeros_like(probs).scatter(0, order[keep], sorted_probs[keep])
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


class _Instruments:
    """Hooks that read, never write: residuals at the probe layers, and (for units) every MLP's neuron activations,
    every attention layer's per-head inputs to `o_proj`, and the last layer's output for the final norm's scale.

    Entered after the SteeringSession, so the residuals they read are the steered ones.
    """

    def __init__(self, bench: Workbench, layers: Sequence[int], *, units: bool) -> None:
        self.bench = bench
        self.model = bench.model
        self.layers = sorted(set(layers))
        self.residuals: dict[int, Any] = {}
        self.acts: dict[int, Any] = {}
        self.z: dict[int, Any] = {}
        self._handles: list[Any] = []
        self._capture: Any = None
        self.units = units and _has_unit_layout(self.model)

    def __enter__(self) -> _Instruments:
        from sandbagkit.probing.hooks import capture_layers, decoder_layers

        wanted = list(self.layers)
        last = self.bench.n_layers - 1
        if self.units and last not in wanted:
            wanted.append(last)
        if wanted:
            self._capture = capture_layers(self.model, wanted)
            self.residuals = self._capture.__enter__()
        if self.units:
            for i, block in enumerate(decoder_layers(self.model)):
                self._handles.append(block.mlp.down_proj.register_forward_pre_hook(_keep(self.acts, i)))
                self._handles.append(block.self_attn.o_proj.register_forward_pre_hook(_keep(self.z, i)))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._capture is not None:
            self._capture.__exit__(None, None, None)
            self._capture = None

    def last_residuals(self) -> dict[int, Any]:
        return {
            layer: tensor[0, -1].float().cpu().numpy()
            for layer, tensor in self.residuals.items()
            if layer in self.layers
        }

    def top_units(self, token_id: int) -> list[dict[str, Any]]:
        """The units whose direct write moved `token_id`'s logit most at this position, by |contribution|."""
        import torch

        model = self.model
        stream = self.residuals.get(self.bench.n_layers - 1)
        if stream is None:
            return []
        x = stream[0, -1].float()
        norm = model.model.norm
        weight = norm.weight.detach().float()
        if self.bench._model_type().startswith("gemma"):
            weight = 1.0 + weight  # Gemma's RMSNorm scales by (1 + weight)
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        readout = torch.rsqrt(x.pow(2).mean() + eps) * weight * model.lm_head.weight[token_id].detach().float()
        rows: list[tuple[float, dict[str, Any]]] = []
        n_heads = int(model.config.num_attention_heads)
        for layer, block in enumerate(model.model.layers):
            act = self.acts.get(layer)
            if act is not None:
                a = act[0, -1].float()
                contrib = a * (readout @ block.mlp.down_proj.weight.detach().float())  # [d_mlp]
                for n in torch.topk(contrib.abs(), min(TOP_UNITS, contrib.shape[0])).indices.tolist():
                    unit = {"kind": "neuron", "layer": layer, "index": int(n), "pole": "+" if a[n] >= 0 else "-"}
                    unit["act"] = round(float(a[n]), 4)
                    rows.append((float(contrib[n]), unit))
            z = self.z.get(layer)
            if z is not None:
                per = (readout @ block.self_attn.o_proj.weight.detach().float()) * z[0, -1].float()
                heads = per.view(n_heads, -1).sum(-1)
                rows.extend((float(heads[h]), {"kind": "head", "layer": layer, "index": h}) for h in range(n_heads))
        rows.sort(key=lambda row: abs(row[0]), reverse=True)
        out = []
        for contribution, unit in rows[:TOP_UNITS]:
            unit["contribution"] = round(contribution, 4)
            unit["address"] = unit_address(unit)
            unit["href"] = unit_href(self.bench.model_id, unit)
            out.append(unit)
        return out


def model_slug(model_id: str) -> str:
    """The Units room's model slug (same rule as `steerlab.atlas.model_slug`): the id's last part, lowercased, every
    run of characters outside [a-z0-9.] turned into "-". Qwen/Qwen2.5-0.5B-Instruct -> qwen2.5-0.5b-instruct."""
    return re.sub(r"[^a-z0-9.]+", "-", model_id.rsplit("/", 1)[-1].lower()).strip("-")


def unit_href(model_id: str, unit: Mapping[str, Any]) -> str:
    """The Units room's URL of a unit: `/units/<slug>/L8/n344+` or `/units/<slug>/L3/h5` ("+" stays literal).

    Uses `steerlab.atlas.unit_href` when the atlas package is installed, so the two rooms never disagree.
    """
    try:  # imported by name: the atlas package (the Units room) is optional
        atlas_href = importlib.import_module("steerlab.atlas").unit_href
    except (ImportError, AttributeError):
        return f"/units/{model_slug(model_id)}/{unit_address(unit)}"
    if unit["kind"] == "head":
        return str(atlas_href(model_id, unit["layer"], unit["index"]))
    pole = -1 if unit.get("pole", "+") in ("-", "\u2212") else 1  # atlas takes the pole as +1 / -1
    return str(atlas_href(model_id, unit["layer"], unit["index"], pole))


def unit_address(unit: Mapping[str, Any]) -> str:
    """The Units room's address of a unit, relative to `/units/<model>/`: `L8/n344+` (a neuron pole) or `L3/h5`."""
    if unit["kind"] == "head":
        return f"L{unit['layer']}/h{unit['index']}"
    return f"L{unit['layer']}/n{unit['index']}{unit.get('pole', '+')}"


def _keep(store: dict[int, Any], i: int) -> Callable[..., None]:
    def pre(module: Any, args: tuple[Any, ...]) -> None:
        store[i] = args[0].detach()

    return pre


def _has_unit_layout(model: Any) -> bool:
    try:
        block = model.model.layers[0]
        return (
            hasattr(block.mlp, "down_proj")
            and hasattr(block.self_attn, "o_proj")
            and hasattr(model.model, "norm")
            and hasattr(model, "lm_head")
        )
    except (AttributeError, IndexError, TypeError):
        return False
