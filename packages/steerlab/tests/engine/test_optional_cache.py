"""The feature cache is optional: without it, a feature that is not loaded names the flag that loads it."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

pytest.importorskip("torch")

from steerlab import engine
from steerlab.control import SAESource, SourceUnavailableError
from steerlab.engine import DirectionRegistry

FEATURE = SAESource(kind="sae", release="test/sae-release", layer=1, feature=3)


def _missing(name: str) -> object:
    real = importlib.import_module

    def import_module(module: str, package: str | None = None) -> ModuleType:
        if module == "steerlab.features.cache":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(module, package)

    return import_module


def test_without_the_feature_cache_an_unloaded_feature_names_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine.importlib, "import_module", _missing("steerlab.features"))
    with pytest.raises(SourceUnavailableError, match="--sae-layer 1"):
        DirectionRegistry(model="some/model")(FEATURE)


def test_a_missing_dependency_inside_the_feature_cache_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine.importlib, "import_module", _missing("some_dependency"))
    with pytest.raises(ModuleNotFoundError, match="some_dependency"):
        DirectionRegistry(model="some/model")(FEATURE)
