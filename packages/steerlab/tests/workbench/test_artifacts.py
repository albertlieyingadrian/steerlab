"""Artifacts: save, list, load, export to a zip and import back, bit for bit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from steerlab.workbench.artifacts import ArtifactError, ArtifactStore, slug


def test_round_trip_through_a_bundle(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path / "a")
    vector = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    a.save({"type": "direction", "name": "french", "model": "m", "layer": 1, "scale": 2.5}, {"vector": vector})
    a.save(
        {
            "type": "edit",
            "name": "flip",
            "model": "m",
            "mode": "reflect",
            "axis": [1.0, 0.0],
            "units": [{"kind": "neuron", "layer": 0, "index": 3}],
        }
    )
    blob = a.export(["direction/french", "edit/flip"])
    b = ArtifactStore(tmp_path / "b")
    assert sorted(b.import_bundle(blob)) == ["direction/french", "edit/flip"]
    meta, arrays = b.load("direction", "french")
    assert meta["scale"] == 2.5 and meta["arrays"] == ["vector"]
    assert np.array_equal(arrays["vector"], vector) and arrays["vector"].dtype == np.float32
    edit, none = b.load("edit", "flip")
    assert none == {} and edit["units"][0]["index"] == 3
    assert {m["name"] for m in b.list()} == {"french", "flip"} and [m["name"] for m in b.list("edit")] == ["flip"]


def test_import_one_json_and_delete(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    key = store.import_files(json.dumps({"type": "lens", "name": "logit", "model": "m"}).encode())
    assert key == "lens/logit" and store.exists("lens", "logit")
    store.delete("lens", "logit")
    assert not store.exists("lens", "logit")


def test_bad_artifacts_are_refused(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError):
        store.save({"type": "picture", "name": "x"})
    with pytest.raises(ArtifactError):
        store.save({"type": "edit", "name": "../escape"})
    with pytest.raises(ArtifactError):
        store.import_bundle(b"not a zip")
    with pytest.raises(ArtifactError):
        store.load("edit", "missing")
    assert slug("Polite / formal #2") == "Polite-formal-2"
