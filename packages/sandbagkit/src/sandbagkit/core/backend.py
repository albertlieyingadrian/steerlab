"""One protocol for every model: local Hugging Face, Anthropic, OpenAI, or a scripted fake.

The reference toolkit had three incompatible "model API" interfaces. Here there is one. A Protocol is
structural typing (like a TypeScript interface): any class with these four methods IS a ModelBackend; it never
has to inherit from anything.

    backend.generate(prompt)                  -> Generation      one answer
    backend.generate_with_cot(prompt, cot)    -> Generation      answer AFTER a given (possibly edited) reasoning
    backend.generate_batch(prompts)           -> list[Generation]
    backend.model_info()                      -> ModelInfo       recorded in every result
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from sandbagkit.core.types import ModelInfo


class GenerationConfig(BaseModel):
    """Decoding settings. The defaults are deterministic: greedy decoding with a fixed seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 0
    system_prompt: str | None = None
    stop: tuple[str, ...] = ()


class Generation(BaseModel):
    """One model output."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model: str = ""
    prompt: str | None = None
    cot: str | None = Field(default=None, description="the reasoning the answer was conditioned on, if any")
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict, description="token counts, e.g. input_tokens/output_tokens")
    latency_ms: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    def generate(self, prompt: str, *, config: GenerationConfig | None = None) -> Generation:
        """Answer one prompt."""
        ...

    def generate_with_cot(self, prompt: str, cot: str, *, config: GenerationConfig | None = None) -> Generation:
        """Answer `prompt` as if the model had already reasoned `cot`. Counterfactual CoT edits `cot`."""
        ...

    def generate_batch(self, prompts: Sequence[str], *, config: GenerationConfig | None = None) -> list[Generation]:
        """Answer many prompts; the output order matches the input order."""
        ...

    def model_info(self) -> ModelInfo: ...


@dataclass(frozen=True)
class BackendCall:
    """One recorded call to a FakeBackend, so tests can assert on exactly what was asked."""

    method: str
    prompt: str
    cot: str | None = None
    config: GenerationConfig | None = None


class FakeBackend:
    """A scripted, offline backend for tests and lessons.

    - `responses`: a dict from prompt to answer, or a function prompt -> answer. Unknown prompts get `default`.
    - `cot_responses`: a function (prompt, cot) -> answer for generate_with_cot. Without it, the CoT is ignored.
    """

    def __init__(
        self,
        responses: Mapping[str, str] | Callable[[str], str] | None = None,
        *,
        default: str = "",
        cot_responses: Callable[[str, str], str] | None = None,
        name: str = "fake",
    ) -> None:
        self._responses = responses
        self._default = default
        self._cot_responses = cot_responses
        self.name = name
        self.calls: list[BackendCall] = []

    def _answer(self, prompt: str) -> str:
        if self._responses is None:
            return self._default
        if callable(self._responses):
            return self._responses(prompt)
        return self._responses.get(prompt, self._default)

    def generate(self, prompt: str, *, config: GenerationConfig | None = None) -> Generation:
        self.calls.append(BackendCall("generate", prompt, config=config))
        return Generation(text=self._answer(prompt), model=self.name, prompt=prompt, finish_reason="stop")

    def generate_with_cot(self, prompt: str, cot: str, *, config: GenerationConfig | None = None) -> Generation:
        self.calls.append(BackendCall("generate_with_cot", prompt, cot=cot, config=config))
        text = self._cot_responses(prompt, cot) if self._cot_responses else self._answer(prompt)
        return Generation(text=text, model=self.name, prompt=prompt, cot=cot, finish_reason="stop")

    def generate_batch(self, prompts: Sequence[str], *, config: GenerationConfig | None = None) -> list[Generation]:
        return [self.generate(prompt, config=config) for prompt in prompts]

    def model_info(self) -> ModelInfo:
        return ModelInfo(name=self.name, provider="fake")
