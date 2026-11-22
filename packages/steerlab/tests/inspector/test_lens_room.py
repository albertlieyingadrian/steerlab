"""The Lens room on the tiny model: lens status, the layer × position grid, a cell's reads and units, loading a lens."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import torch
from fastapi.testclient import TestClient

from steerlab.inspect.app import create_app
from steerlab.inspect.core import Inspector
from steerlab.lens import jacobian

from .conftest import PROMPT

BASE = "http://127.0.0.1:8770"


@pytest.fixture
def client(inspector: Inspector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("STEERLAB_LENS_DIR", str(tmp_path / "lenses"))
    for p in inspector.model.parameters():
        p.requires_grad_(False)
    docs = [inspector.tokenize("hello world, hello there, said the cat"), inspector.tokenize("abc abc abc abc abc ab")]
    acc = jacobian.fit(inspector.model, docs, "R", dim_batch=16, skip_first=2)
    jacobian.save(acc.blob(inspector.name), jacobian.own_path(inspector.name, "R"))
    with TestClient(create_app(inspector, cache_dir=tmp_path), base_url=BASE) as test_client:
        yield test_client


def test_lens_info_lists_every_lens(client: TestClient) -> None:
    info = client.get("/api/lens/info").json()
    assert info["slug"] == "tiny-random-qwen2" and info["n_layers"] == 2
    assert info["lenses"]["logit"]["status"] == "loaded" and info["lenses"]["R"]["status"] == "loaded"
    assert info["lenses"]["J"]["status"] == info["lenses"]["S"]["status"] == "needs a fit"
    assert info["loaded"] == ["logit", "R"]
    assert "lens" in [room["id"] for room in client.get("/api/rooms").json()["rooms"]]
    assert client.get("/lens?text=hi&lens=R").status_code == 200


def test_the_grid_has_a_cell_per_layer_and_position(client: TestClient, inspector: Inspector) -> None:
    n = len(inspector.tokenize(PROMPT))
    for lens, z in (("logit", False), ("R", False), ("R", True)):
        grid = client.post("/api/lens/grid", json={"text": PROMPT, "lens": lens, "z": z}).json()
        assert len(grid["layers"]) == 2 and all(len(row["cells"]) == n for row in grid["layers"])
        assert grid["value"] == ("z" if z else "p") and len(grid["output"]) == n
    logit = client.post("/api/lens/grid", json={"text": PROMPT, "lens": "logit"}).json()
    assert [c["id"] for c in logit["layers"][-1]["cells"]] == [c["id"] for c in logit["output"]]


def test_a_cell_reads_every_lens_and_names_units(client: TestClient) -> None:
    response = client.post("/api/lens/cell", json={"text": PROMPT, "lens": "R", "layer": 0, "pos": 3, "k": 5})
    assert response.status_code == 200, response.text
    cell = response.json()
    assert set(cell["reads"]) == {"logit", "R"} and len(cell["reads"]["R"]["z"]) == 5
    assert cell["top"] == cell["reads"]["R"]["raw"][0]
    toward = cell["neurons"]["toward"]
    assert len(toward) == 8 and toward[0]["push"] >= toward[-1]["push"]
    assert toward[0]["href"].startswith("/units/tiny-random-qwen2/L0/n") and toward[0]["href"][-1] in "+-"
    assert cell["heads"][0]["href"].startswith("/units/tiny-random-qwen2/L0/h")


def test_neuron_pushes_sum_to_the_mlp_share(client: TestClient, inspector: Inspector) -> None:
    """Through the logit lens, Σ_n act_n × (g ⊙ W_U[v]) · W_down[:, n] is the MLP output's linear score for v."""
    cell = client.post("/api/lens/cell", json={"text": PROMPT, "lens": "logit", "layer": 1, "pos": 2}).json()
    ids = inspector.tokenize(PROMPT)
    trace = inspector.trace(ids)
    v = cell["top"]["id"]
    u = inspector.W_U[v] * inspector.w_norm
    total = float((trace.acts[1][2] * (inspector.down(1) @ u)).sum())
    mlp_out = trace.acts[1][2] @ inspector.down(1)
    assert abs(total - float(mlp_out @ u)) < 1e-4
    assert torch.isfinite(torch.tensor(total))


def test_bad_lens_requests_say_why(client: TestClient) -> None:
    load = client.post("/api/lens/load", json={"lens": "S"})
    assert load.status_code == 400 and "no S-lens" in load.json()["detail"]
    grid = client.post("/api/lens/grid", json={"text": PROMPT, "lens": "J"})
    assert grid.status_code == 400 and "not loaded" in grid.json()["detail"]
    assert client.post("/api/lens/cell", json={"text": PROMPT, "lens": "R", "layer": 7, "pos": 0}).status_code == 400
    assert client.post("/api/lens/grid", json={"text": PROMPT, "lens": "X"}).status_code == 422


def test_unit_links_match_the_atlas_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    from steerlab.inspect.rooms import lens

    with_atlas = [lens.unit_href("Qwen/Qwen3-8B", 8, 344, 1), lens.unit_href("Qwen/Qwen3-8B", 20, 17)]
    real = lens.importlib.import_module

    def no_atlas(name: str) -> object:
        if name.startswith("steerlab.atlas"):
            raise ImportError(name)
        return real(name)

    monkeypatch.setattr(lens.importlib, "import_module", no_atlas)
    without = [lens.unit_href("Qwen/Qwen3-8B", 8, 344, 1), lens.unit_href("Qwen/Qwen3-8B", 20, 17)]
    assert with_atlas == without == ["/units/qwen3-8b/L8/n344+", "/units/qwen3-8b/L20/h17"]
    assert lens.unit_href("Qwen/Qwen3-8B", 8, 344, -1).endswith("n344-")
