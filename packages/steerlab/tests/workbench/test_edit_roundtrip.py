"""Save in the Edit room, apply in the Workbench: one artifact store, one schema, one edit implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")
pytest.importorskip("steerlab.edits.reflection")

from fastapi.testclient import TestClient  # noqa: E402
from sandbagkit.backends.tiny import char_tokenizer, tiny_model  # noqa: E402

from steerlab.inspect.app import create_app  # noqa: E402
from steerlab.inspect.core import Inspector  # noqa: E402
from steerlab.workbench import edits as edit_ops  # noqa: E402
from steerlab.workbench.artifacts import ArtifactStore  # noqa: E402

BASE = "http://127.0.0.1:8770"
MODEL = "tiny-random-qwen2"
AXIS = {"kind": "pairs", "method": "activations", "pairs": [["cat", "dog"]]}
IDS = [[5, 9, 11, 30, 41]]


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[tuple[TestClient, Inspector, Path]]:
    tokenizer = char_tokenizer()
    model = tiny_model(len(tokenizer), seed=0)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    inspector = Inspector(model, tokenizer, name=MODEL)
    with TestClient(create_app(inspector, cache_dir=tmp_path), base_url=BASE) as client:
        yield client, inspector, tmp_path


def logits(model: Any) -> Any:
    with torch.no_grad():
        return model(torch.tensor(IDS), use_cache=False).logits


def test_save_in_edit_apply_in_workbench(setup: tuple[TestClient, Inspector, Path]) -> None:
    client, inspector, cache = setup
    model = inspector.model
    before = logits(model)

    units = {"per_layer": 4, "kind": "both", "heads_per_layer": 1}
    record = client.post("/api/edit/stack", json={"axis": AXIS, "units": units}).json()["stack"][0]
    saved = client.post(f"/api/edit/stack/{record['id']}/save", json={"name": "cat-dog"}).json()
    assert saved == {"saved": "edit/cat-dog", "arrays": ["reads", "writes"]}
    assert [row["name"] for row in client.get("/api/edit/saved").json()["saved"]] == ["cat-dog"]

    # the Workbench lists it from the same store
    listed = client.get("/api/workbench/artifacts", params={"type": "edit"}).json()["artifacts"]
    assert [row["name"] for row in listed] == ["cat-dog"] and listed[0]["mode"] == "reflect"

    # what the Edit room's list does to the model ...
    with client.app.state.edits.applied(model, inspector.tokenizer):  # type: ignore[attr-defined]
        expected = logits(model)
    assert not torch.equal(expected, before)
    client.post(f"/api/edit/stack/{record['id']}/toggle")  # off, so only the Workbench applies it below

    # ... the Workbench applier does too, from the stored files, with no tokenizer
    meta, arrays = ArtifactStore(cache / "artifacts").load("edit", "cat-dog")
    with edit_ops.applied(model, [meta], [arrays]) as labels:
        assert torch.equal(logits(model), expected)
    assert labels == ["cat-dog: reflect k=2 on 10 units"]
    assert torch.equal(logits(model), before)

    # and by name in a chat completion
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 3,
        "temperature": 0,
        "edits": ["cat-dog"],
    }
    out = client.post("/v1/chat/completions", json=body).json()
    assert out["steerlab"]["edits"] == ["cat-dog: reflect k=2 on 10 units"]
    assert torch.equal(logits(model), before)

    # and back into the Edit room's list
    assert len(client.post("/api/edit/load", json={"name": "cat-dog"}).json()["stack"]) == 2
    assert client.post("/api/edit/load", json={"name": "nope"}).status_code == 400
