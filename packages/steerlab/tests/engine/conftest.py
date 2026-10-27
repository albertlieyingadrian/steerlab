"""Fixtures for the engine tests: the two tiny models (Qwen3-shaped and Gemma-3-shaped) and fixed random directions.

Everything runs on CPU in float32 in milliseconds; no download. The character tokenizer comes from sandbagkit's tiny
backend, so a prompt is a handful of single-character tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sandbagkit.backends.tiny import tiny_backend  # noqa: E402

from steerlab.control import ControlState  # noqa: E402
from steerlab.engine import DirectionRegistry, SteeringSession  # noqa: E402
from steerlab.saes.common import FeatureDirection  # noqa: E402

VARIANTS = ("qwen3", "gemma3_text")


@dataclass
class Tiny:
    """One tiny model with everything a steering test needs."""

    name: str
    model: Any
    tokenizer: Any
    registry: DirectionRegistry
    messages: list[str]

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    def state(self) -> ControlState:
        return ControlState(self.name, self.n_layers, self.registry, on_message=self.messages.append)

    def session(self, state: ControlState | None = None) -> SteeringSession:
        return SteeringSession(self.model, self.tokenizer, state or self.state())

    def prompt_ids(self, text: str = "Say hi") -> torch.Tensor:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        ids: torch.Tensor = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids

    def control(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        return {"model": self.name, "entries": entries}


def unit_direction(d_model: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return v / v.norm()


def make_tiny(variant: str) -> Tiny:
    backend = tiny_backend(seed=0, variant=variant)
    model, tokenizer = backend.model, backend.tokenizer
    d = int(model.config.hidden_size)
    registry = DirectionRegistry()
    registry.add_direction("random", FeatureDirection(unit_direction(d, 1), layer=1, release="test", feature=0))
    registry.add_direction(
        "random0", FeatureDirection(unit_direction(d, 2), layer=0, release="test", feature=1, label="layer-0 noise")
    )
    return Tiny(backend.name, model, tokenizer, registry, [])


@pytest.fixture(scope="session")
def tiny_qwen3() -> Tiny:
    return make_tiny("qwen3")


@pytest.fixture(scope="session")
def tiny_gemma3() -> Tiny:
    return make_tiny("gemma3_text")


@pytest.fixture(params=VARIANTS, scope="session")
def tiny(request: pytest.FixtureRequest) -> Tiny:
    return make_tiny(request.param)


def direction_entry(name: str, strength: float, layers: list[int] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"source": {"kind": "direction", "name": name}, "strength": strength}
    if layers is not None:
        entry["layers"] = layers
    return entry
