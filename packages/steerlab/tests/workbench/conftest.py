"""Fixtures for the Workbench tests: sandbagkit's tiny random Qwen2 model, a Workbench around it, three probes.

Two layers, 32-wide stream, a character tokenizer with a chat template: no download, milliseconds per reply. The
words are random, so the tests check plumbing (shapes, determinism, round trips, schema), never what a probe means.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from sandbagkit.backends.tiny import tiny_backend  # noqa: E402

from steerlab.workbench.probes import load_bundle  # noqa: E402
from steerlab.workbench.session import Workbench  # noqa: E402

FIT = ("sandbagging", "formal.casual", "french.english")


@pytest.fixture
def tiny() -> object:
    return tiny_backend(seed=0)


@pytest.fixture
def bench(tiny: object, tmp_path: Path) -> Workbench:
    b = Workbench(tiny.model, tiny.tokenizer, "tiny-random-qwen2", tmp_path)  # type: ignore[attr-defined]
    b.fit_bundle([spec for spec in load_bundle() if spec.name in FIT])
    return b
