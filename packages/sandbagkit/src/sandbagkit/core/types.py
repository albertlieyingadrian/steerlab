"""Typed contracts shared by every detector, loader, backend and runner.

The flow is always the same:

    dataset loader ──► TrialDataset (many trials) ──► Detector.detect() ──► DetectorResult
                                                                                │
    runner: one CellRecord per detector × dataset × model ──► RunRecord (JSON, schema_version)

Detectors never take a single string. They take many trials, because every statistic needs a sample.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import string
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = "1.0"
"""Version of the RunRecord JSON format. Bump the major part on any breaking change."""

SUPPORTED_SCHEMA_MAJORS = frozenset({"1"})


class _Model(BaseModel):
    """Base for every contract: unknown fields are errors, so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class ContextType(StrEnum):
    """The situation a prompt is framed as. Sandbagging shows up as a gap between these."""

    NEUTRAL = "neutral"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"


class DetectorKind(StrEnum):
    """How a detector sees the model. The runner routes on this."""

    BLACK_BOX = "black_box"  # only prompts and outputs
    WHITE_BOX = "white_box"  # activations or weights, so a local model is required


class Verdict(StrEnum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    INSUFFICIENT_DATA = "insufficient_data"


class RunStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


# --- Provenance ------------------------------------------------------------------------------------------------


class DatasetInfo(_Model):
    """Where trials came from. Recorded in every result so a run can be reproduced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, description="e.g. 'cais/wmdp' or 'sandbagkit/contrast_pairs'")
    revision: str | None = Field(default=None, description="pinned dataset commit SHA or version")
    license: str = Field(default="unknown", description="SPDX id when known, else 'unknown'")
    split: str | None = None
    subset: str | None = None
    url: str | None = None


class ModelInfo(_Model):
    """What model produced the outputs. Backends return this from model_info()."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    provider: str = Field(default="unknown", description="'hf', 'anthropic', 'openai', 'fake', ...")
    revision: str | None = None
    dtype: str | None = None
    device: str | None = None
    num_parameters: int | None = Field(default=None, ge=0)
    supports_activations: bool = False


# --- Trials ----------------------------------------------------------------------------------------------------


class BaseTrial(_Model):
    """One question put to a model, plus (once run) how it answered.

    `item_id` groups trials that ask the same underlying question (e.g. the same item with shuffled options).
    `paraphrase_group` groups trials that ask it in different words.
    """

    trial_id: str = Field(min_length=1)
    prompt: str
    response: str | None = None
    correct: bool | None = None
    score: float | None = Field(default=None, description="graded score when correctness isn't binary")
    context: ContextType = ContextType.NEUTRAL
    difficulty: float | None = Field(default=None, ge=0.0, le=1.0, description="0 = easiest, 1 = hardest")
    item_id: str | None = None
    paraphrase_group: str | None = None
    source: DatasetInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trial(BaseTrial):
    """A free-text trial: a prompt and a response, optionally graded."""

    kind: Literal["text"] = "text"


class MultipleChoiceTrial(BaseTrial):
    """A multiple-choice trial. `answer` and `chosen` are indexes into `options` as presented."""

    kind: Literal["multiple_choice"] = "multiple_choice"
    options: list[str] = Field(min_length=2)
    answer: int
    chosen: int | None = None

    @model_validator(mode="after")
    def _check_indexes(self) -> Self:
        if not 0 <= self.answer < len(self.options):
            raise ValueError(f"answer index {self.answer} is outside the {len(self.options)} options")
        if self.chosen is not None:
            if not 0 <= self.chosen < len(self.options):
                raise ValueError(f"chosen index {self.chosen} is outside the {len(self.options)} options")
            is_right = self.chosen == self.answer
            if self.correct is None:
                self.correct = is_right
            elif self.correct != is_right:
                raise ValueError(f"correct={self.correct} contradicts chosen={self.chosen}, answer={self.answer}")
        return self

    @staticmethod
    def letter(index: int) -> str:
        """0 -> 'A', 1 -> 'B', ..."""
        return string.ascii_uppercase[index]

    @property
    def answer_option(self) -> str:
        return self.options[self.answer]

    @property
    def answer_letter(self) -> str:
        return self.letter(self.answer)

    @property
    def chosen_option(self) -> str | None:
        return None if self.chosen is None else self.options[self.chosen]


AnyTrial = Annotated[Trial | MultipleChoiceTrial, Field(discriminator="kind")]
"""Either trial kind. JSON round-trips back to the right class via the `kind` field."""


class TrialDataset(_Model):
    """An ordered collection of trials with provenance. This is what detectors consume."""

    name: str
    info: DatasetInfo | None = None
    trials: list[AnyTrial] = Field(default_factory=list)

    @field_validator("trials")
    @classmethod
    def _unique_ids(cls, trials: list[BaseTrial]) -> list[BaseTrial]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for trial in trials:
            if trial.trial_id in seen:
                duplicates.add(trial.trial_id)
            seen.add(trial.trial_id)
        if duplicates:
            raise ValueError(f"duplicate trial_id values: {sorted(duplicates)[:5]}")
        return trials

    def __len__(self) -> int:
        return len(self.trials)

    def by_context(self, context: ContextType | str) -> list[BaseTrial]:
        wanted = ContextType(context)
        return [t for t in self.trials if t.context is wanted]

    def contexts(self) -> set[ContextType]:
        return {t.context for t in self.trials}

    def by_item(self) -> dict[str, list[BaseTrial]]:
        """Trials grouped by item_id (trials without one are left out)."""
        return _group(self.trials, lambda t: t.item_id)

    def by_paraphrase_group(self) -> dict[str, list[BaseTrial]]:
        return _group(self.trials, lambda t: t.paraphrase_group)

    def multiple_choice(self) -> list[MultipleChoiceTrial]:
        return [t for t in self.trials if isinstance(t, MultipleChoiceTrial)]

    def accuracy(self, context: ContextType | str | None = None) -> tuple[int, int]:
        """(number correct, number graded), counting only trials where `correct` is known."""
        pool = self.trials if context is None else self.by_context(context)
        graded = [t.correct for t in pool if t.correct is not None]
        return sum(graded), len(graded)

    def filter(self, keep: Callable[[BaseTrial], bool]) -> TrialDataset:
        return TrialDataset(name=self.name, info=self.info, trials=[t for t in self.trials if keep(t)])


def _group(trials: Sequence[BaseTrial], key: Callable[[BaseTrial], str | None]) -> dict[str, list[BaseTrial]]:
    groups: dict[str, list[BaseTrial]] = defaultdict(list)
    for trial in trials:
        k = key(trial)
        if k is not None:
            groups[k].append(trial)
    return dict(groups)


# --- Results ---------------------------------------------------------------------------------------------------


class ConfidenceInterval(_Model):
    """A range [low, high] that the true value falls in with probability `level` (e.g. 0.95)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    low: float
    high: float
    level: float = Field(default=0.95, gt=0.0, lt=1.0)
    estimate: float | None = Field(default=None, description="the point estimate the interval surrounds")
    method: str | None = Field(default=None, description="'percentile bootstrap', 'wilson', 'beta posterior', ...")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")
        return self

    @property
    def width(self) -> float:
        return self.high - self.low


class DetectorResult(_Model):
    """What every detector returns.

    - `detected`: the yes/no verdict.
    - `confidence`: 0..1, how strongly the evidence supports the verdict's direction of concern.
    - `score`: the detector's raw statistic (its scale depends on the detector).
    - `n`: how many trials the verdict rests on. Too few gives `insufficient_data` instead of a guess.
    """

    detector: str
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    score: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    ci: ConfidenceInterval | None = None
    n: int = Field(ge=0)
    insufficient_data: bool = False
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _insufficient_is_not_a_verdict(self) -> Self:
        if self.insufficient_data and self.detected:
            raise ValueError("a result with insufficient_data cannot also be detected")
        return self

    @classmethod
    def insufficient(cls, detector: str, *, n: int, min_required: int, reason: str | None = None) -> DetectorResult:
        return cls(
            detector=detector,
            detected=False,
            confidence=0.0,
            n=n,
            insufficient_data=True,
            reason=reason or f"only {n} usable trials; need at least {min_required}",
            details={"min_required": min_required},
        )

    @property
    def verdict(self) -> Verdict:
        if self.insufficient_data:
            return Verdict.INSUFFICIENT_DATA
        return Verdict.DETECTED if self.detected else Verdict.NOT_DETECTED


# --- Run records ------------------------------------------------------------------------------------------------


class SchemaVersionError(ValueError):
    """A results file was written by an incompatible version of the schema."""


class EnvironmentInfo(_Model):
    python: str
    platform: str
    sandbagkit_version: str
    packages: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def capture(cls, packages: Sequence[str] = ("numpy", "scipy", "pydantic", "torch", "transformers")) -> Self:
        """Record the interpreter, OS and installed versions of `packages` (missing ones are skipped)."""
        versions: dict[str, str] = {}
        for name in packages:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
        try:
            own = importlib.metadata.version("sandbagkit")
        except importlib.metadata.PackageNotFoundError:
            own = "unknown"
        return cls(
            python=platform.python_version(), platform=platform.platform(), sandbagkit_version=own, packages=versions
        )


class CellRecord(_Model):
    """One cell of a run matrix: one detector on one dataset with one model."""

    detector: str
    detector_config: dict[str, Any] = Field(default_factory=dict)
    model: ModelInfo
    dataset: DatasetInfo
    status: RunStatus = RunStatus.OK
    result: DetectorResult | None = None
    error: str | None = None
    duration_s: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _status_matches_payload(self) -> Self:
        if self.status is RunStatus.OK and self.result is None:
            raise ValueError("a cell with status 'ok' needs a result")
        if self.status is RunStatus.FAILED and not self.error:
            raise ValueError("a cell with status 'failed' needs an error message")
        return self


def _check_schema_version(version: str) -> None:
    major = version.split(".", 1)[0]
    if major not in SUPPORTED_SCHEMA_MAJORS:
        raise SchemaVersionError(
            f"results file has schema_version '{version}', but this sandbagkit reads "
            f"{sorted(SUPPORTED_SCHEMA_MAJORS)}.x (current {SCHEMA_VERSION}). Upgrade sandbagkit or re-run."
        )


class RunRecord(_Model):
    """A whole run, saved as one JSON file. Reports are rendered from this and nothing else."""

    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    runner: str = Field(default="sandbagkit", description="'sandbagkit', 'inspect' or 'sandbagrig'")
    config: dict[str, Any] = Field(default_factory=dict)
    environment: EnvironmentInfo | None = None
    cells: list[CellRecord] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _supported(cls, version: str) -> str:
        _check_schema_version(version)
        return version

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str | bytes) -> RunRecord:
        """Parse a results file, checking schema_version first so the error says what's wrong."""
        data = json.loads(text)
        if not isinstance(data, dict) or "schema_version" not in data:
            raise SchemaVersionError("not a sandbagkit results file: no schema_version field")
        _check_schema_version(str(data["schema_version"]))
        try:
            return cls.model_validate(data)
        except ValidationError as error:
            raise ValueError(f"invalid results file:\n{error}") from error

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
