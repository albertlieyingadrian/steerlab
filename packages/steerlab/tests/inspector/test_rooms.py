"""The room plug-in contract: discovery, /api/rooms, deep-linked pages and the Model room, on the tiny model."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from steerlab.inspect import rooms
from steerlab.inspect.app import create_app
from steerlab.inspect.core import Inspector
from steerlab.inspect.rooms import model as model_room

BASE = "http://127.0.0.1:8770"


@pytest.fixture
def client(inspector: Inspector, tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(inspector, cache_dir=tmp_path), base_url=BASE) as test_client:
        yield test_client


def test_rooms_are_listed_in_nav_order(client: TestClient) -> None:
    data = client.get("/api/rooms").json()
    ids = [room["id"] for room in data["rooms"]]
    assert ids[0] == "trace" and "model" in ids
    orders = [room["order"] for room in data["rooms"]]
    assert orders == sorted(orders)
    model = next(room for room in data["rooms"] if room["id"] == "model")
    assert model["key"] == "m" and model["js"] and model["css"]
    assert data["model"] == "tiny-random-qwen2" and data["skipped"] == []


def test_every_room_url_serves_the_page_with_a_base(client: TestClient) -> None:
    for path in ("/", "/trace", "/model", "/trace/anything?preset=france"):
        page = client.get(path)
        assert page.status_code == 200, path
        assert '<base href="/">' in page.text and 'src="static/shell.js"' in page.text
    assert 'data-room="model"' in client.get("/model").text
    assert 'data-room="trace"' in client.get("/").text
    assert client.get("/no-such-room").status_code == 404
    for asset in ("shell.js", "rooms/model.js", "rooms/model.css", "rooms/trace.js"):
        assert client.get(f"/static/{asset}").status_code == 200, asset


def test_the_model_room_describes_the_model(client: TestClient) -> None:
    info = client.get("/api/model/info").json()
    assert info["model"] == "tiny-random-qwen2" and info["n_layers"] == 2 and info["n_heads"] == 4
    assert info["d_mlp"] == 64 and info["neurons"] == 128 and info["mlp"] == "SwiGLU" and info["poles"] == 2
    assert info["units"] == 2 * 64 * 2 + 2 * 4
    assert info["lenses"]["logit"]["status"] == "ready"
    assert info["catalog"]["covered"] is False and info["saes"] == []
    assert info["rooms"][0]["id"] == "trace"
    assert "loaded" in client.get("/api/model/cached").json()


def test_known_models_have_catalog_and_saes() -> None:
    assert "Qwen/Qwen3-8B" in model_room.CATALOG_MODELS
    assert [s["layers"] for s in model_room.saes_for("Qwen/Qwen3-8B")] == [[9, 18, 27]]
    assert model_room.saes_for("meta-llama/Llama-3.1-8B-Instruct")[0]["layers"] == [19]
    assert model_room.saes_for("Qwen/Qwen2.5-0.5B-Instruct") == []


def test_cached_models_reads_directory_names(tmp_path: Path) -> None:
    for name in ("models--Qwen--Qwen3-8B", "datasets--x--y", "models--gpt2"):
        (tmp_path / name).mkdir()
    assert model_room.cached_models(tmp_path) == ["Qwen/Qwen3-8B", "gpt2"]


def test_discovery_skips_broken_and_invalid_rooms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "fake_rooms"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    register = "def register(app, ctx): pass\n"
    (pkg / "good.py").write_text('ROOM = {"id": "good", "title": "Good", "order": 5, "key": "x"}\n' + register)
    (pkg / "late.py").write_text('ROOM = {"id": "late", "title": "Late", "order": 1, "key": "y"}\n' + register)
    (pkg / "broken.py").write_text("import no_such_module_anywhere\n")
    (pkg / "invalid.py").write_text('ROOM = {"id": "invalid"}\n')
    (pkg / "_private.py").write_text("raise RuntimeError('never imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    found, skipped = rooms.discover("fake_rooms")
    assert [room.id for room in found] == ["late", "good"]
    assert {s["id"] for s in skipped} == {"broken", "invalid"}
    assert "ModuleNotFoundError" in next(s["error"] for s in skipped if s["id"] == "broken")
    for name in [m for m in sys.modules if m.startswith("fake_rooms")]:
        del sys.modules[name]


def test_room_context_view_turns_value_errors_into_400(inspector: Inspector, tmp_path: Path) -> None:
    from fastapi import HTTPException

    ctx = rooms.RoomContext(inspector, cache_dir=tmp_path)
    assert ctx.model_id == "tiny-random-qwen2" and ctx.view(lambda: 3) == 3
    with pytest.raises(HTTPException) as caught:
        ctx.view(lambda: (_ for _ in ()).throw(ValueError("bad layer")))
    assert caught.value.status_code == 400 and caught.value.detail == "bad layer"
    assert ctx.room_cache("lens").is_dir() and "tiny-random-qwen2" in str(ctx.room_cache("lens"))
