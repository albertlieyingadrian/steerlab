"""The Workbench room over HTTP on the tiny model: the loom flow, probes, artifacts and the OpenAI-compatible API."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from sandbagkit.backends.tiny import char_tokenizer, tiny_model

from steerlab.inspect.app import create_app
from steerlab.inspect.core import Inspector

BASE = "http://127.0.0.1:8770"
MODEL = "tiny-random-qwen2"


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    tokenizer = char_tokenizer()
    model = tiny_model(len(tokenizer), seed=0)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    inspector = Inspector(model, tokenizer, name=MODEL)
    cache: Path = tmp_path_factory.mktemp("cache")
    with TestClient(create_app(inspector, cache_dir=cache), base_url=BASE) as test_client:
        assert test_client.post("/api/workbench/probes/refit").status_code == 200
        yield test_client


def ndjson(response) -> list[dict]:  # type: ignore[no-untyped-def]
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_the_room_is_listed_and_its_pages_serve(client: TestClient) -> None:
    rooms = client.get("/api/rooms").json()["rooms"]
    assert any(r["id"] == "workbench" and r["js"] and r["css"] for r in rooms)
    for path in ("/workbench", "/workbench/probes", "/workbench/artifacts"):
        assert client.get(path).status_code == 200
    state = client.get("/api/workbench/state").json()
    assert state["model"] == MODEL and len(state["probes"]) == 16 and state["fit"]["state"] == "ready"


def test_chat_on_the_loom_streams_tokens_then_saves_the_reply(client: TestClient) -> None:
    loom = client.post("/api/workbench/looms", json={"title": ""}).json()
    turn = client.post(f"/api/workbench/looms/{loom['id']}/turn", json={"content": "hi", "parent": None}).json()
    user = turn["result"]
    body = {"loom": loom["id"], "parent": user, "steer": "2 formal.casual", "sampling": {"max_tokens": 5}}
    events = ndjson(client.post("/api/workbench/generate", json=body))
    assert [e["type"] for e in events][-1] == "done" and events[0]["type"] == "token"
    done = events[-1]
    node = done["node"]
    assert node["parent"] == user and node["meta"]["steer"][0]["strength"] == 2
    assert len(node["meta"]["tokens"]) == len(events) - 1
    # a reroll is a sibling; compare puts the two replies side by side
    plain = ndjson(client.post("/api/workbench/generate", json={**body, "steer": None}))[-1]["node"]
    tree = client.get(f"/api/workbench/looms/{loom['id']}").json()
    assert tree["nodes"][user]["children"] == [node["id"], plain["id"]]
    view = client.get(f"/api/workbench/looms/{loom['id']}/compare", params={"a": node["id"], "b": plain["id"]}).json()
    assert view["ancestor"] == user and len(view["a"]) == len(view["b"]) == 1
    replay = client.post("/api/workbench/replay", json={"loom": loom["id"], "node": node["id"]}).json()
    assert len(replay["probes"]["formal.casual"]) == len(node["meta"]["token_ids"])
    bad = ndjson(client.post("/api/workbench/generate", json={**body, "parent": node["id"]}))
    assert bad[-1]["type"] == "error" and "user" in bad[-1]["detail"]


def test_probes_fit_and_read(client: TestClient) -> None:
    fitted = client.post(
        "/api/workbench/probes/fit",
        json={"name": "shouting", "positive": ["HELLO", "STOP NOW"], "negative": ["hello", "stop now"]},
    ).json()
    assert fitted["name"] == "shouting" and "random_d_prime" in fitted
    read = client.post("/api/workbench/probes/read", json={"text": "HELLO", "probes": ["shouting"]}).json()
    assert len(read["probes"]["shouting"]) == len(read["tokens"]) == 5
    assert (
        client.post(
            "/api/workbench/probes/fit", json={"name": "x", "positive": ["a"], "negative": ["b", "c"]}
        ).status_code
        == 422
    )


def test_artifacts_save_export_import(client: TestClient) -> None:
    saved = client.post("/api/workbench/artifacts/from-probe", json={"probe": "french.english", "name": "fr"}).json()
    assert saved["type"] == "direction" and saved["arrays"] == ["vector"]
    assert any(s["name"] == "fr" for s in client.get("/api/workbench/state").json()["steerables"])
    blob = client.post("/api/workbench/artifacts/export", json={"keys": ["direction/fr"]})
    assert blob.headers["content-type"] == "application/zip"
    assert sorted(zipfile.ZipFile(io.BytesIO(blob.content)).namelist()) == ["direction/fr.json", "direction/fr.npz"]
    client.delete("/api/workbench/artifacts/direction/fr")
    assert client.post("/api/workbench/artifacts/import", content=blob.content).json() == {"imported": ["direction/fr"]}
    edit = {
        "type": "edit",
        "name": "e1",
        "mode": "reflect",
        "axis": [1.0] + [0.0] * 31,
        "units": [{"kind": "neuron", "layer": 0, "index": 0}],
    }
    assert client.post("/api/workbench/artifacts/import", content=json.dumps(edit)).json()["imported"] == ["edit/e1"]
    listed = client.get("/api/workbench/artifacts", params={"type": "edit"}).json()["artifacts"]
    assert listed[0]["name"] == "e1" and listed[0]["axis"] == [1.0] + [0.0] * 31


def test_openai_chat_completion_schema(client: TestClient) -> None:
    assert client.get("/v1/models").json()["data"][0]["id"] == MODEL
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 2,
        "steer": [{"probe": "formal.casual", "strength": 2}],
        "probes": ["formal.casual"],
        "edits": ["e1"],
    }
    out = client.post("/v1/chat/completions", json=body).json()
    assert out["object"] == "chat.completion" and out["id"].startswith("chatcmpl-")
    choice = out["choices"][0]
    assert choice["message"]["role"] == "assistant" and choice["finish_reason"] in ("stop", "length")
    assert out["usage"]["total_tokens"] == out["usage"]["prompt_tokens"] + out["usage"]["completion_tokens"]
    assert len(choice["logprobs"]["content"][0]["top_logprobs"]) == 2
    extras = out["steerlab"]
    assert extras["steer"][0]["strength"] == 2 and extras["edits"] and "formal.casual" in extras["probes"]
    assert len(extras["probes"]["formal.casual"]["z"]) == out["usage"]["completion_tokens"]


def test_openai_stream_and_errors(client: TestClient) -> None:
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "max_tokens": 3,
        "temperature": 0,
        "stream": True,
    }
    lines = [line[6:] for line in client.post("/v1/chat/completions", json=body).text.split("\n\n") if line]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant" and chunks[-1]["choices"][0]["finish_reason"]
    plain = client.post("/v1/chat/completions", json={**body, "stream": False}).json()["choices"][0]
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == plain["message"]["content"]
    wrong = client.post("/v1/chat/completions", json={**body, "model": "gpt-4o", "stream": False})
    assert wrong.status_code == 404 and wrong.json()["error"]["code"] == "model_not_found"
    unknown = client.post("/v1/chat/completions", json={**body, "stream": False, "steer": "1 nope"})
    assert unknown.status_code == 400 and "loaded" in unknown.json()["error"]["message"]
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422


def test_unknown_hosts_are_refused(client: TestClient) -> None:
    assert client.get("/v1/models", headers={"host": "evil.example"}).status_code == 400
