"""Computed atlases on sandbagkit's tiny random Qwen2 (2 layers, 64 neurons, 4 heads, a character tokenizer).

The words are random, so these check the arithmetic against a direct recomputation (top firings, quantiles, exact
drivers, edges, per-token traces) and the plumbing (files, cards, the room's routes), never what a unit means.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from sandbagkit.backends.tiny import char_tokenizer, tiny_model  # noqa: E402

from steerlab.atlas import compute  # noqa: E402
from steerlab.atlas.address import Address  # noqa: E402
from steerlab.atlas.computed import ComputedAtlas  # noqa: E402
from steerlab.atlas.corpus import Corpus, build_corpus  # noqa: E402

TEXTS = {
    "web": ["the quick brown fox jumps over the lazy dog and keeps running far away " * 3] * 4,
    "code": ["def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n" * 4] * 3,
}


@pytest.fixture(scope="module")
def parts() -> compute.Parts:
    tokenizer = char_tokenizer()
    model = tiny_model(len(tokenizer), seed=0)
    model.set_attn_implementation("eager")
    return compute.Parts(model, tokenizer)


@pytest.fixture(scope="module")
def corpus(parts: compute.Parts) -> Corpus:
    return build_corpus(parts.tokenizer, n_tokens=1500, max_len=48, texts=TEXTS)


@pytest.fixture(scope="module")
def stats(parts: compute.Parts, corpus: Corpus) -> dict[str, Any]:
    return compute.corpus_stats(parts, corpus.docs, k=6, batch_size=4)


def layer_acts(parts: compute.Parts, ids: list[int], layer: int) -> Any:
    """One document's activations at `layer` from its own forward pass (no padding): the reference."""
    acts, _, handles = compute._capture(parts, [layer])
    try:
        parts.model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return acts[layer][0]


def test_corpus_takes_each_slice(corpus: Corpus) -> None:
    assert set(corpus.sources) == {"web", "code"} and all(8 <= len(d) <= 48 for d in corpus.docs)
    assert "documents" in corpus.describe() and Corpus.from_json(corpus.to_json()).docs == corpus.docs


def test_top_firings_match_a_direct_recomputation(parts: compute.Parts, corpus: Corpus, stats: dict[str, Any]) -> None:
    layer, n = 1, 5
    acts = [layer_acts(parts, doc, layer) for doc in corpus.docs]
    for pi, sign in enumerate((1.0, -1.0)):
        best = max(float((a[1:, n] * sign).max()) for a in acts)
        assert stats["top_val"][layer, pi, 0, n].item() == pytest.approx(max(best, 0.0), rel=1e-4, abs=1e-6)
        doc, pos = stats["top_doc"][layer, pi, 0, n].item(), stats["top_pos"][layer, pi, 0, n].item()
        assert pos >= 1 and (acts[doc][pos, n] * sign).item() == pytest.approx(best, rel=1e-4, abs=1e-6)


def test_quantiles_land_within_a_bin_of_the_exact_ones(
    parts: compute.Parts, corpus: Corpus, stats: dict[str, Any]
) -> None:
    layer, n = 0, 3
    values = torch.cat([layer_acts(parts, doc, layer)[1:, n] for doc in corpus.docs])
    assert stats["n_positions"] == values.numel()
    for key, q in (("q50", 0.5), ("q90", 0.9), ("q99", 0.99)):
        exact = torch.quantile(values, q).item()
        got = stats[key][layer, 0, n].item()
        if exact > 1e-4:
            assert exact / 2**0.25 <= got <= exact * 2**0.25, (key, got, exact)


def test_token_drivers_are_exact(parts: compute.Parts, corpus: Corpus) -> None:
    contexts = corpus.docs[:2]
    found = compute.token_drivers(parts, contexts, list(range(10, 40)), position=6, k=3, layers=[0, 1])
    n = 7
    token = found["ids"][1, 1, 0, n].item()  # layer 1, the − pole, its top driver
    mean = sum(-layer_acts(parts, [*ctx[:6], token], 1)[-1, n].item() for ctx in contexts) / 2
    assert found["vals"][1, 1, 0, n].item() == pytest.approx(mean, rel=1e-4, abs=1e-6)
    assert found["position"] == 6 and found["n_candidates"] == 30


def test_unit_trace_matches_the_hook(parts: compute.Parts, corpus: Corpus) -> None:
    doc = corpus.docs[0][:12]
    ref = layer_acts(parts, doc, 1)[:, 9]
    assert compute.unit_trace(parts, doc, 1, "neuron", 9, -1) == pytest.approx((-ref).tolist(), rel=1e-5, abs=1e-6)
    heads = compute.unit_trace(parts, doc, 0, "head", 2)
    assert len(heads) == len(doc) and all(v >= 0 for v in heads)


def test_weight_reads(parts: compute.Parts) -> None:
    logits = compute.write_logits(parts, 1, 4, -1)
    assert logits.shape == (parts.vocab,) and torch.allclose(logits, -compute.write_logits(parts, 1, 4, 1))
    rms = compute.layer_write_rms(parts, 1, chunk=16)
    W = parts.final_norm()[:, None] * parts.down(1)
    assert torch.allclose(rms, (parts.W_U.float() @ W).pow(2).mean(-1).sqrt(), rtol=1e-4)
    reads = compute.head_reads(parts, 1, 3, list(range(parts.vocab)), k=4)
    assert -1.0 <= reads["copy_score"] <= 1.0 and len(reads["copies"]) == 4 and reads["kv_head"] == 1
    edges = compute.unit_edges(parts, 0, 2, 1, k=3)
    top = edges["out"][0]
    r = compute.read_directions(parts, 1)[0 if top["pole"] > 0 else 1, top["neuron"]]
    w = parts.down(0)[:, 2]
    assert float(r @ w / w.norm()) == pytest.approx(top["cos"], rel=1e-4)
    assert edges["in"] == []  # nothing comes before layer 0
    assert all(e["cos"] >= 0 for e in compute.unit_edges(parts, 1, 2, -1, k=3)["in"])


def test_a_computed_atlas_round_trips(parts: compute.Parts, corpus: Corpus, tmp_path: Path) -> None:
    atlas = ComputedAtlas(tmp_path / "tiny", parts=parts, hf_id="tiny/model")
    stages: list[str] = []
    meta = atlas.build(
        corpus=corpus, k=6, n_candidates=40, position=6, weights=True, progress=lambda stage, done: stages.append(stage)
    )
    assert {"activations", "drivers", "weights", "done"} <= set(stages)
    assert meta["n_layers"] == 2 and meta["d_mlp"] == 64 and 0 < meta["drivers"]["n_candidates"] <= 40
    for name in ("meta.json", "corpus.json", "tokens.json", "stats.npz", "drivers.npz", "weights.npz", "heads.json"):
        assert (tmp_path / "tiny" / name).exists()
    card = atlas.unit("tiny", Address.parse("L1/n5+"))
    assert card["source"] == "computed" and card["label"]["source"] == "computed"
    assert card["label"]["fires_conf"] is None and card["drivers"]["at"]
    assert card["contexts"] and card["contexts"][0]["pct"] is not None
    assert {r["key"] for r in card["reads"]} == {"write_logit", "write_logit_z", "read_embed"}
    assert len(card["edges"]["in"]) == 16 and card["edges"]["out"] == []
    head = atlas.unit("tiny", Address.parse("L0/h1"))
    assert head["kind"] == "head" and head["reads"] and head["copy"]["copy_score"] is not None
    ov = atlas.overview("tiny", top=5)
    assert len(ov["layers"]) == 2 and ov["layers"][0]["units"] and len(ov["layers"][0]["heads"]) == 4
    docs = list(atlas.search_documents("tiny", reads=True))
    assert docs and all(isinstance(a, Address) for a, *_ in docs)
    # A reader without the model (a Modal-built atlas on a laptop) renders the same card from the files alone.
    offline = ComputedAtlas(tmp_path / "tiny").unit("tiny", Address.parse("L1/n5+"))
    assert offline["contexts"] == card["contexts"]
    assert {r["key"] for r in offline["reads"]} == {"write_logit", "write_logit_z", "read_embed"}
    assert offline["edges"]["in"] == [] and "loaded" in offline["edges"]["how"]


# The room, end to end on the tiny model


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from steerlab.atlas.catalog import Catalog, LocalSource
    from steerlab.inspect.app import create_app
    from steerlab.inspect.core import Inspector

    tokenizer = char_tokenizer()
    inspector = Inspector(tiny_model(len(tokenizer), seed=0), tokenizer, name="tiny/random-qwen2")
    app = create_app(inspector, cache_dir=tmp_path_factory.mktemp("cache"))
    app.state.atlas._catalog = Catalog(LocalSource(tmp_path_factory.mktemp("empty")), models={})  # no network
    with TestClient(app, base_url="http://127.0.0.1:8770") as test_client:
        yield test_client


def test_the_room_is_listed_and_its_page_served(client: Any) -> None:
    rooms = client.get("/api/rooms").json()["rooms"]
    assert any(r["id"] == "units" and r["js"] and r["css"] for r in rooms)
    assert client.get("/units/random-qwen2/L1/n3+").status_code == 200
    assert client.get("/static/rooms/units.js").status_code == 200


def test_the_room_builds_then_serves_the_atlas(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import steerlab.atlas.corpus as corpus_module

    real = corpus_module.build_corpus
    monkeypatch.setattr(
        corpus_module, "build_corpus", lambda tok, **kw: real(tok, n_tokens=800, max_len=40, texts=TEXTS)
    )
    slug = "random-qwen2"
    models = client.get("/api/units/models").json()
    assert models["loaded_slug"] == slug and models["models"][-1]["source"] == "computable"
    assert client.get(f"/api/units/overview?model={slug}").json()["source"] == "computable"
    assert client.get(f"/api/units/unit?model={slug}&address=L0/n1%2B").status_code == 400
    assert client.post("/api/units/build", json={"model": slug}).json()["state"] == "running"
    for _ in range(600):
        job = client.get("/api/units/models").json()["jobs"][slug]
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done", job
    ov = client.get(f"/api/units/overview?model={slug}").json()
    assert ov["source"] == "computed" and len(ov["layers"]) == 2
    card = client.get(f"/api/units/unit?model={slug}&address=L1/n3%2B").json()
    assert card["address"] == "L1/n3+" and card["fireable"] and card["links"]["trace"].startswith("/?layer=1")
    assert all("title" in e for e in card["edges"]["in"])
    fired = client.post("/api/units/fire", json={"model": slug, "address": "L1/n3+", "text": "hello world"}).json()
    assert len(fired["tokens"]) == len("hello world") and fired["basis"].startswith("percentiles over")
    graph = client.get(f"/api/units/graph?model={slug}&address=L1/n3%2B&up=1&down=0&fan=2").json()
    assert graph["center"] == "L1/n3+" and len(graph["nodes"]) == 3
    assert client.get(f"/api/units/search?model={slug}&q=L1 h2").json()["open"] == "L1/h2"
    assert "results" in client.get(f"/api/units/search?model={slug}&q=the").json()
    assert client.get(f"/api/units/unit?model={slug}&address=L9/n1%2B").status_code == 400
    assert client.get("/api/units/unit?model=nope&address=L0/n1%2B").status_code == 400
