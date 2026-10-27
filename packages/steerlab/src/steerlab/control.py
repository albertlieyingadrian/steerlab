"""The control file: what is being steered right now, re-read every token.

Think of the control file as the mixing desk for a live show. Each entry is one fader: a *source* (an SAE feature named
by release, layer and ID, or a named difference-of-means direction), a *strength* (how far the fader is pushed, in
calibrated units once calibration exists; raw unit-vector multiples until then), and optionally the *layers* to apply
it at (default: the source's own layer). An SAE entry can also set `"mode": "clamp"`: instead of adding a fixed vector,
the engine reads the feature's activation every token and moves it to `strength` (the way Golden Gate Claude was made),
so the push is large where the feature is quiet and small where it already fires. The engine re-reads the file before
every generated token, so moving a fader changes the very next token.

    {"model": "Qwen/Qwen3-1.7B",
     "entries": [{"source": {"kind": "sae", "release": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
                             "layer": 14, "feature": 1234},
                  "strength": 2.0, "layers": [14]},
                 {"source": {"kind": "direction", "name": "chinese"}, "strength": -1.0}]}

Three rules protect a live session from a half-written file: a parse failure keeps the last valid state and is
reported once; a file naming another model rejects every entry with one message and generation continues unsteered;
an entry whose source is not loaded is skipped with a message naming the flag that loads it. Writers should write to
a temp file and rename it into place, so the engine never sees a partial file.

This module imports no torch: `steerlab schema control` and `steerlab control validate` run on the base install.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from steerlab import defaults

if TYPE_CHECKING:
    import torch

    from steerlab.saes.common import FeatureDirection

SCHEMA_PATH = Path(__file__).parent / "schemas" / "control.schema.json"
"""The published JSON Schema, generated from the models below (`steerlab schema control` prints the same)."""

MODEL_LAYERS: dict[str, int] = {defaults.QWEN_MODEL: 28, defaults.GEMMA_MODEL: 26}
"""Decoder layer counts of the two supported models, so `control validate` can check layer indexes without loading."""

WRITE_ADVICE = "Write a new version to a temp file and rename it over this one; the engine re-reads it every token."


class SAESource(BaseModel):
    """An SAE feature: `release` and `layer` pick the SAE, `feature` the decoder direction inside it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sae"]
    release: str = Field(min_length=1, description="Hub repo id of the SAE release, e.g. google/gemma-scope-2-1b-it.")
    layer: int = Field(ge=0, description="Decoder layer the SAE was trained on.")
    feature: int = Field(ge=0, description="Feature index inside that SAE.")

    def describe(self) -> str:
        return f"sae {self.release} L{self.layer} feature {self.feature}"


class DirectionSource(BaseModel):
    """A named direction, e.g. one of the repo's difference-of-means directions (sandbagging, refusal, chinese)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direction"]
    name: str = Field(min_length=1, description="Name of a loaded direction.")

    def describe(self) -> str:
        return f"direction {self.name!r}"


Source = Annotated[SAESource | DirectionSource, Field(discriminator="kind")]


class ControlEntry(BaseModel):
    """One fader: a source, how hard to push it, and where."""

    model_config = ConfigDict(extra="forbid")

    source: Source
    strength: float = Field(
        description="Multiplier on the source's unit direction (times its calibration scale once known). 0 = off."
    )
    layers: list[int] | None = Field(
        default=None,
        min_length=1,
        description="Decoder layers to add the direction at. Default: the source's own layer.",
    )
    mode: Literal["add", "clamp"] | None = Field(
        default=None,
        description=(
            "add (the default when absent): add strength × the unit direction. clamp (SAE features only, at the "
            "SAE's layer): set the feature's activation to strength every token, by adding (strength - activation) × "
            "its decoder vector."
        ),
    )


class ControlFile(BaseModel):
    """The whole file: the model it applies to, and the entries currently in force (none = unsteered)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "title": "steerlab control file",
            "description": "Live steering state, re-read by the engine before every generated token. " + WRITE_ADVICE,
        },
    )

    model: str = Field(min_length=1, description="The model this file steers; entries are ignored for any other.")
    entries: list[ControlEntry] = Field(default_factory=list)


class ControlValidationError(ValueError):
    """The control file is not valid JSON, or valid JSON that does not fit the schema."""


def parse_control(text: str) -> ControlFile:
    """Parse the text of a control file, turning JSON and schema errors into one ControlValidationError."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ControlValidationError(f"not valid JSON ({error.msg} at position {error.pos})") from None
    try:
        return ControlFile.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'file'}: {item['msg']}" for item in error.errors()
        )
        raise ControlValidationError(f"does not fit the control schema ({problems})") from None


def load_control(path: str | Path) -> ControlFile:
    return parse_control(Path(path).read_text())


def control_json_schema() -> dict[str, Any]:
    """The JSON Schema of `ControlFile`, with the draft marker so editors can validate against it."""
    schema = ControlFile.model_json_schema()
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}


def default_layers(entry: ControlEntry) -> list[int]:
    """Where an entry applies when it names no layers: the SAE's layer, or nothing knowable for a direction."""
    return [entry.source.layer] if isinstance(entry.source, SAESource) else []


def validate_layers(control: ControlFile, n_layers: int | None) -> list[str]:
    """Problems a schema cannot see: layer indexes outside the model. Empty list when everything fits."""
    if n_layers is None:
        return []
    problems: list[str] = []
    for index, entry in enumerate(control.entries):
        layers = entry.layers if entry.layers is not None else default_layers(entry)
        problems.extend(
            f"entry {index} ({entry.source.describe()}): layer {layer} doesn't exist, "
            f"this model has {n_layers} layers (valid: 0..{n_layers - 1})"
            for layer in layers
            if not 0 <= layer < n_layers
        )
    return problems


class SourceUnavailableError(LookupError):
    """The resolver has nothing loaded for this source. The message says how to load it."""


Resolver = Callable[[SAESource | DirectionSource], "FeatureDirection"]
"""Maps a source to a FeatureDirection, or raises SourceUnavailableError. `steerlab.engine.DirectionRegistry` is one."""


@dataclass
class Clamp:
    """One clamped SAE feature at one layer: everything the hook needs to read and reset its activation.

    activation(h) = h @ encoder + bias when above threshold, else 0; the hook adds (target - activation) × decoder.
    """

    encoder: torch.Tensor
    bias: float
    threshold: float
    decoder: torch.Tensor
    target: float


@dataclass
class AppliedEntry:
    """One entry as it is being applied: resolved label, the layers it lands on, and the raw norm it adds."""

    label: str
    source: dict[str, Any]
    strength: float
    scale: float
    layers: list[int]
    mode: str = "add"

    @property
    def norm(self) -> float:
        """The norm added per position by this entry alone: |strength × scale| times a unit vector. For a clamp it is
        the most it adds (where the feature is silent): |target| × the decoder vector's norm."""
        return abs(self.strength * self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "strength": self.strength,
            "scale": self.scale,
            "layers": list(self.layers),
            "norm": self.norm,
            "mode": self.mode,
        }

    def describe(self) -> str:
        if self.mode == "clamp":
            return f"{self.label} clamped to {self.strength} at layers={self.layers} (adds up to norm {self.norm:.3g})"
        return f"{self.label} strength={self.strength} layers={self.layers} (adds norm {self.norm:.3g})"


@dataclass
class ControlState:
    """The resolved steering state the hooks read: one summed vector per layer, plus what produced it.

    `apply` replaces the state from a parsed control file (or a dict / JSON text in that shape). `version` increases on
    every apply so the engine knows when to re-register hooks. Messages (skips, mismatches, parse failures) go to
    `on_message`, each distinct message once.
    """

    model: str
    n_layers: int
    resolver: Resolver
    on_message: Callable[[str], None] | None = None
    applied: list[AppliedEntry] = field(default_factory=list)
    deltas: dict[int, torch.Tensor] = field(default_factory=dict)
    clamps: dict[int, list[Clamp]] = field(default_factory=dict)
    version: int = 0
    _seen: set[str] = field(default_factory=set, repr=False)

    @property
    def layers(self) -> list[int]:
        """The layers with an entry in force (strength 0 still counts: the entry is live, its fader is just down)."""
        return sorted(set(self.deltas) | set(self.clamps))

    def say(self, message: str) -> None:
        """Report a message once; a repeat of the same text is dropped."""
        if message in self._seen:
            return
        self._seen.add(message)
        if self.on_message is not None:
            self.on_message(message)

    def apply(self, control: ControlFile | dict[str, Any] | str) -> None:
        """Replace the state. Raises ControlValidationError for text or a dict that does not fit the schema."""
        if isinstance(control, str):
            control = parse_control(control)
        elif isinstance(control, dict):
            control = parse_control(json.dumps(control))
        if control.model != self.model:
            self.say(
                f"control file names model {control.model!r} but this session runs {self.model!r}: "
                f"ignoring all {len(control.entries)} entries and generating unsteered"
            )
            self._set([], {})
            return
        applied: list[AppliedEntry] = []
        deltas: dict[int, torch.Tensor] = {}
        clamps: dict[int, list[Clamp]] = {}
        for index, entry in enumerate(control.entries):
            if entry.mode == "clamp":
                clamped = self._clamp(index, entry)
                if clamped is not None:
                    layer, clamp, label = clamped
                    clamps.setdefault(layer, []).append(clamp)
                    applied.append(
                        AppliedEntry(
                            label=label,
                            source=entry.source.model_dump(),
                            strength=entry.strength,
                            scale=float(clamp.decoder.norm()),
                            layers=[layer],
                            mode="clamp",
                        )
                    )
                continue
            try:
                direction = self.resolver(entry.source)
            except SourceUnavailableError as error:
                self.say(f"entry {index} ({entry.source.describe()}) skipped: {error}")
                continue
            layers = entry.layers if entry.layers is not None else [direction.layer]
            bad = [layer for layer in layers if not 0 <= layer < self.n_layers]
            if bad:
                self.say(
                    f"entry {index} ({entry.source.describe()}) skipped: layer {bad[0]} doesn't exist, "
                    f"this model has {self.n_layers} layers (valid: 0..{self.n_layers - 1})"
                )
                continue
            scale = direction.scale if direction.scale is not None else 1.0
            vector = (entry.strength * scale) * direction.unit
            for layer in layers:
                deltas[layer] = deltas[layer] + vector if layer in deltas else vector
            applied.append(
                AppliedEntry(
                    label=direction.label or entry.source.describe(),
                    source=entry.source.model_dump(),
                    strength=entry.strength,
                    scale=scale,
                    layers=list(layers),
                )
            )
        self._set(applied, deltas, clamps)

    def _clamp(self, index: int, entry: ControlEntry) -> tuple[int, Clamp, str] | None:
        """Resolve a clamp entry to (layer, Clamp, label), or report why it is skipped and return None."""
        source = entry.source
        prefix = f"entry {index} ({source.describe()}) skipped"
        if not isinstance(source, SAESource):
            self.say(f"{prefix}: clamp needs an SAE feature (its encoder reads the activation); use mode add")
            return None
        if entry.layers is not None and entry.layers != [source.layer]:
            self.say(f"{prefix}: a clamped feature can only sit at its SAE's layer {source.layer}")
            return None
        lookup = getattr(self.resolver, "clamp", None)
        if lookup is None:
            self.say(f"{prefix}: this resolver cannot clamp features")
            return None
        try:
            encoder, bias, threshold, decoder, label = lookup(source)
        except (SourceUnavailableError, NotImplementedError) as error:
            self.say(f"{prefix}: {error}")
            return None
        if not 0 <= source.layer < self.n_layers:
            self.say(f"{prefix}: layer {source.layer} doesn't exist, this model has {self.n_layers} layers")
            return None
        return source.layer, Clamp(encoder, bias, threshold, decoder, entry.strength), label

    def clear(self) -> None:
        self._set([], {})

    def forget_messages(self) -> None:
        """Let every message be reported again (a long-running server shows a repeated problem each time)."""
        self._seen.clear()

    def _set(
        self,
        applied: list[AppliedEntry],
        deltas: dict[int, torch.Tensor],
        clamps: dict[int, list[Clamp]] | None = None,
    ) -> None:
        self.applied = applied
        self.deltas = deltas
        self.clamps = clamps or {}
        self.version += 1


class ControlWatcher:
    """Re-reads a control file when its `(mtime_ns, size)` changes and applies it to a ControlState.

    `poll()` is called once per generated token. It returns True when the state changed. A file that fails to parse
    (truncated mid-write, invalid JSON, wrong shape), or whose entries the resolver cannot turn into vectors, leaves
    the state alone and is reported once per distinct failure. A path that does not exist yet is reported once too.
    """

    def __init__(self, path: str | Path, state: ControlState) -> None:
        self.path = Path(path)
        self.state = state
        self._stamp: tuple[int, int] | None = None
        self._seen_file = False

    def poll(self) -> bool:
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            if self._stamp is not None:
                self._stamp = None
                self.state.say(f"{self.path.name} disappeared; keeping the last valid state")
            elif not self._seen_file:
                self.state.say(f"{self.path} does not exist yet; generating unsteered until it appears")
            return False
        self._seen_file = True
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._stamp:
            return False
        self._stamp = stamp
        try:
            control = parse_control(self.path.read_text())
        except (OSError, UnicodeDecodeError, ControlValidationError) as error:
            self.state.say(f"{self.path.name}: {error}; keeping the last valid state")
            return False
        try:
            self.state.apply(control)
        except Exception as error:  # a crash inside the resolver must not kill the live session
            self.state.say(f"{self.path.name}: could not apply ({error}); keeping the last valid state")
            return False
        return True
