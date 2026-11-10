"""The explorer's HTTP routes on the tiny model: every view answers, bad requests say why, unknown hosts are refused."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from steerlab.inspect.app import create_app
from steerlab.inspect.core import Inspector

from .conftest import PROMPT

BASE = "http://127.0.0.1:8770"


@pytest.fixture
def client(inspector: Inspector) -> Iterator[TestClient]:
    with TestClient(create_app(inspector), base_url=BASE) as test_client:
        yield test_client


def test_the_page_and_its_assets_are_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "Layer explorer" in page.text and 'src="static/inspect.js"' in page.text
    assert client.get("/static/inspect.js").status_code == 200
    assert client.get("/static/inspect.css").status_code == 200


def test_info_lists_the_model_and_presets(client: TestClient) -> None:
    info = client.get("/api/info").json()
    assert info["model"] == "tiny-random-qwen2" and info["n_layers"] == 2
    assert {"france", "golden-gate", "maths", "sandbag"} <= {p["id"] for p in info["presets"]}


def test_every_view_answers(client: TestClient) -> None:
    run = client.post("/api/run", json={"text": PROMPT}).json()
    target = run["target"]["id"]
    body = {"text": PROMPT, "target": target}
    assert client.post("/api/layer", json={**body, "layer": 1}).json()["target"]["id"] == target
    assert client.post("/api/neuron", json={**body, "layer": 1, "neuron": 4}).json()["id"] == "L1.N4"
    assert client.post("/api/find", json={"word": "e", "text": PROMPT}).json()["toward"]
    edit = {**body, "edits": [{"layer": 1, "neuron": 4, "mode": "flip"}], "max_new_tokens": 3}
    assert client.post("/api/edit", json=edit).json()["edits"] == ["L1.N4 flipped"]
    dose = client.post("/api/dose", json={**body, "layer": 0, "neuron": 2}).json()
    assert len(dose["curve"]) == len(dose["scales"])


def test_a_typed_target_word_and_a_position_are_used(client: TestClient) -> None:
    run = client.post("/api/run", json={"text": PROMPT, "target": "o", "pos": 3}).json()
    assert run["target"]["text"] == "o" and run["pos"] == 3


def test_bad_requests_say_why(client: TestClient) -> None:
    layer = client.post("/api/layer", json={"text": PROMPT, "layer": 9})
    assert layer.status_code == 400 and "layer 9" in layer.json()["detail"]
    assert client.post("/api/run", json={"text": ""}).status_code == 422
    mode = {"text": PROMPT, "edits": [{"layer": 0, "neuron": 0, "mode": "melt"}]}
    assert client.post("/api/edit", json=mode).status_code == 422
    assert client.post("/api/edit", json={"text": PROMPT, "edits": []}).status_code == 422


def test_an_unknown_host_is_refused(inspector: Inspector) -> None:
    with TestClient(create_app(inspector), base_url="http://evil.example") as other:
        assert other.get("/api/info").status_code == 400
