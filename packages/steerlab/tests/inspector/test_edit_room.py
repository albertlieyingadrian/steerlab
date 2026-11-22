"""The Edit room's routes on the tiny model: units, the four-column comparison, the shared list, artifacts, restore."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")
pytest.importorskip("steerlab.edits.reflection")

from fastapi.testclient import TestClient  # noqa: E402

from steerlab.inspect.app import create_app  # noqa: E402
from steerlab.inspect.core import Inspector  # noqa: E402

BASE = "http://127.0.0.1:8770"
AXIS = {"kind": "pairs", "method": "activations", "pairs": [["cat", "dog"]]}


@pytest.fixture
def client(inspector: Inspector, tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(inspector, cache_dir=tmp_path), base_url=BASE) as test_client:
        yield test_client


def logits(inspector: Inspector) -> Any:
    with torch.no_grad():
        return inspector.model(torch.tensor([[5, 9, 11, 30]]), use_cache=False).logits


def test_the_room_is_listed_and_served(client: TestClient) -> None:
    rooms = {r["id"]: r for r in client.get("/api/rooms").json()["rooms"]}
    assert rooms["edit"]["key"] == "e" and rooms["edit"]["js"] and rooms["edit"]["css"]
    assert 'data-room="edit"' in client.get("/edit").text
    for asset in ("rooms/edit.js", "rooms/edit.css"):
        assert client.get(f"/static/{asset}").status_code == 200
    info = client.get("/api/edit/info").json()
    assert info["n_layers"] == 2 and info["exact"] is True and info["stack"] == []
    assert info["example"]["k"] == 2.0


def test_units_by_loading_and_by_id(client: TestClient) -> None:
    top = client.post("/api/edit/units", json={"axis": AXIS, "units": {"per_layer": 5}}).json()
    assert top["count"] == {"neurons": 10, "heads": 0}
    assert all(row["vs_layer"] >= 1 for row in top["layers"])  # chosen by loading, so above the layer's median
    ids = client.post("/api/edit/units", json={"axis": AXIS, "units": {"selection": "ids", "ids": ["L1.N3", "L0/h2"]}})
    assert ids.json()["ids"] == ["L1.N3", "L0.H2"]
    assert ids.json()["links"] == [
        {"id": "L1.N3", "href": "/units/tiny-random-qwen2/L1/n3+"},
        {"id": "L0.H2", "href": "/units/tiny-random-qwen2/L0/h2"},
    ]
    basket = {"selection": "ids", "ids": ["/units/tiny-random-qwen2/L1/n4+", "L0/n7-"]}
    assert client.post("/api/edit/units", json={"axis": AXIS, "units": basket}).json()["ids"] == ["L0.N7", "L1.N4"]
    other = {"selection": "ids", "ids": ["/units/qwen3-8b/L1/n4+"]}
    refused_model = client.post("/api/edit/units", json={"axis": AXIS, "units": other})
    assert refused_model.status_code == 400 and "qwen3-8b" in refused_model.json()["detail"]
    spec = {"kind": "both", "per_layer": 2, "heads_per_layer": 1}
    both = client.post("/api/edit/units", json={"axis": AXIS, "units": spec})
    assert both.json()["count"] == {"neurons": 4, "heads": 2}
    bad_units = {"selection": "ids", "ids": ["L9.N0"]}
    refused = client.post("/api/edit/compare", json={"axis": AXIS, "units": bad_units, "prompts": ["hi"]})
    assert refused.status_code == 400 and "outside" in refused.json()["detail"]


def test_compare_returns_four_columns_and_leaves_the_model_clean(client: TestClient, inspector: Inspector) -> None:
    before = logits(inspector)
    body = {
        "axis": AXIS,
        "units": {"per_layer": 8},
        "k": 2.0,
        "prompts": ["hi", "cat"],
        "max_new_tokens": 4,
        "checks": [{"kind": "words", "source": ["c"], "target": ["d"]}, {"kind": "probe"}],
    }
    data = client.post("/api/edit/compare", json=body).json()
    assert data["order"] == ["before", "after", "random_axis", "random_units"]
    assert data["restored"] is True
    assert data["artifact"]["type"] == "edit" and data["artifact"]["model"] == "tiny-random-qwen2"
    assert data["artifact"]["units"][0]["kind"] == "neuron" and data["artifact"]["k"] == 2.0
    assert torch.equal(logits(inspector), before)
    sampled = client.post("/api/edit/compare", json={**body, "sample": True, "seed": 4, "controls": False}).json()
    assert sampled["order"] == ["before", "after"] and sampled["sample"] is True
    assert client.post("/api/edit/restore").json() == {"restored": True, "max_abs_diff": 0.0, "exact": True}


def test_the_shared_list_and_artifacts(client: TestClient, inspector: Inspector) -> None:
    before = logits(inspector)
    stack = client.post("/api/edit/stack", json={"axis": AXIS, "units": {"per_layer": 4}, "k": -1.0}).json()["stack"]
    assert len(stack) == 1 and stack[0]["kind"] == "reflect" and stack[0]["count"]["neurons"] == 8
    trace = [{"layer": 1, "neuron": 7, "mode": "scale", "value": 3.0}]
    stack = client.post("/api/edit/stack/neurons", json={"edits": trace}).json()["stack"]
    assert [r["kind"] for r in stack] == ["reflect", "neuron"] and stack[1]["label"] == "L1.N7 ×3"
    assert torch.equal(logits(inspector), before)  # the list is applied per request, never left on

    data = client.post("/api/edit/compare", json={"axis": AXIS, "prompts": ["hi"], "max_new_tokens": 3}).json()
    assert len(data["base"]) == 2 and data["restored"]

    rid = stack[0]["id"]
    art = client.get(f"/api/edit/stack/{rid}/artifact").json()
    assert art["type"] == "edit" and art["axis_source"] == AXIS and isinstance(art["created"], float)
    client.post(f"/api/edit/stack/{rid}/toggle")
    assert client.get("/api/edit/stack").json()["stack"][0]["enabled"] is False
    assert len(client.post("/api/edit/import", json={"artifact": art}).json()["stack"]) == 3
    wrong = client.post("/api/edit/import", json={"artifact": {**art, "model": "other/model"}})
    assert wrong.status_code == 400 and "other/model" in wrong.json()["detail"]
    client.delete(f"/api/edit/stack/{rid}")
    assert len(client.get("/api/edit/stack").json()["stack"]) == 2
    assert client.delete("/api/edit/stack").json()["stack"] == []
    assert client.post("/api/edit/stack/neurons", json={"edits": [{"layer": 5, "neuron": 0}]}).status_code == 400
    assert torch.equal(logits(inspector), before)


def test_trace_runs_under_the_shared_list(client: TestClient, inspector: Inspector) -> None:
    plain = client.post("/api/run", json={"text": "hello wor"}).json()
    assert "shared_edits" not in plain
    client.post("/api/edit/stack/neurons", json={"edits": [{"layer": 1, "neuron": 5, "mode": "scale", "value": 6.0}]})
    edited = client.post("/api/run", json={"text": "hello wor"}).json()
    assert edited["shared_edits"] == ["L1.N5 ×6"]
    assert edited["guesses"] != plain["guesses"] or edited["layers"] != plain["layers"]
    layer = client.post("/api/layer", json={"text": "hello wor", "layer": 1}).json()
    assert layer["shared_edits"] == ["L1.N5 ×6"]
    rid = client.get("/api/edit/stack").json()["stack"][0]["id"]
    client.post(f"/api/edit/stack/{rid}/toggle")
    assert client.post("/api/run", json={"text": "hello wor"}).json() == plain  # off again: identical
