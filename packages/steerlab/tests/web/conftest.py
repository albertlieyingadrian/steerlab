"""Fixtures for the web tests: a worker running the tiny Qwen3-shaped model, with one direction and a small SAE.

The loader builds a fresh tiny model each time (milliseconds, no download), so swaps and unloads are real. A visitor's
events land in a `Visitor`, and every event the worker publishes is recorded by `RecordingBroker`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sandbagkit.backends.tiny import tiny_backend  # noqa: E402

from steerlab.control import ControlState  # noqa: E402
from steerlab.engine import DirectionRegistry  # noqa: E402
from steerlab.loading import CatalogEntry, SAEInfo  # noqa: E402
from steerlab.saes.common import FeatureDirection, LoadedSAE  # noqa: E402
from steerlab.web.broker import Broker, Event  # noqa: E402
from steerlab.web.worker import Worker  # noqa: E402

SAE_RELEASE = "test/tiny-sae"
SYSTEM = "Reply in English only."


def unit_direction(d_model: int, seed: int) -> torch.Tensor:
    vector = torch.randn(d_model, generator=torch.Generator().manual_seed(seed))
    return vector / vector.norm()


class TinySAE(LoadedSAE):
    """Six features at layer 1; only the decoder is ever used for steering."""

    def __init__(self, d_model: int) -> None:
        self.release = SAE_RELEASE
        self.layer = 1
        self.W = torch.stack([unit_direction(d_model, 20 + feature) for feature in range(6)])

    @property
    def d_model(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.W.shape[0])

    @property
    def device(self) -> torch.device:
        return self.W.device

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decoder_vector(self, feature: int) -> torch.Tensor:
        return self.W[feature]


CATALOG = [
    CatalogEntry("tiny", "tiny/qwen3", ["random=unused"], SAEInfo(SAE_RELEASE, 1)),
    CatalogEntry("plain", "tiny/plain", [], None),
    CatalogEntry("broken", "tiny/broken", [], None),
]


def tiny_session(model_id: str, on_message: Callable[[str], None], *, directions: bool, sae: bool) -> tuple[Any, Any]:
    backend = tiny_backend(seed=0, variant="qwen3")
    d_model = int(backend.model.config.hidden_size)
    registry = DirectionRegistry(model=model_id)
    if directions:
        direction = FeatureDirection(unit_direction(d_model, 1), layer=1, release="test", feature=0, label="Random")
        registry.add_direction("random", direction)
    if sae:
        registry.add_sae(TinySAE(d_model))
    state = ControlState(model_id, int(backend.model.config.num_hidden_layers), registry, on_message=on_message)
    return backend, state


def tiny_load(entry: CatalogEntry, on_message: Callable[[str], None]) -> tuple[Any, Any]:
    if entry.key == "broken":
        raise RuntimeError("weights are corrupt (test)")
    return tiny_session(entry.model_id, on_message, directions=bool(entry.direction_specs), sae=entry.sae is not None)


class RecordingBroker(Broker):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []
        self._events_lock = threading.Lock()

    def publish(self, event: Event) -> None:
        with self._events_lock:
            self.events.append(event)
        super().publish(event)

    def of(self, kind: str) -> list[Event]:
        with self._events_lock:
            return [event for event in self.events if event["type"] == kind]


@dataclass
class Visitor:
    """Collects one request's events; `done` is set at its `end`."""

    events: list[Event] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    on_event: Callable[[Event], None] | None = None

    def __call__(self, event: Event) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)
        if event["type"] == "end":
            self.done.set()

    def wait(self, timeout: float = 10) -> Visitor:
        assert self.done.wait(timeout), f"no end event; got {self.events}"
        return self

    @property
    def text(self) -> str:
        return "".join(event["text"] for event in self.events if event["type"] == "token")

    @property
    def end(self) -> Event:
        return next(event for event in self.events if event["type"] == "end")


def wait_until(predicate: Callable[[], bool], timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


@pytest.fixture
def unloaded() -> list[str]:
    return []


@pytest.fixture
def make_worker(unloaded: list[str]) -> Any:
    workers: list[Worker] = []

    def build(
        *,
        model: str = "tiny",
        max_new_tokens: int = 12,
        catalog: list[CatalogEntry] | None = None,
        presets: Any = None,
        prompt_for: Any = None,
    ) -> Worker:
        worker = Worker(
            catalog=lambda: list(catalog or CATALOG),
            load=tiny_load,
            unload=lambda backend, state: unloaded.append(state.model),
            system_prompt=SYSTEM,
            broker=RecordingBroker(),
            initial_model=model,
            max_new_tokens=max_new_tokens,
            feature_label=lambda model_id, release, layer, feature: "Bridges" if feature == 3 else None,
            presets=presets,
            prompt_for=prompt_for,
        )
        worker.start()
        wait_until(lambda: worker.state()["status"] in ("ready", "failed"))
        workers.append(worker)
        return worker

    yield build
    for worker in workers:
        worker.stop()
