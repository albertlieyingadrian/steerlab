"""The catalog adapter on a tiny synthetic catalog in the real layout (2 layers, 3 neurons, 2 heads; no download)."""

from __future__ import annotations

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from steerlab.atlas.address import Address  # noqa: E402
from steerlab.atlas.catalog import Catalog, LocalSource  # noqa: E402
from steerlab.atlas.models import CatalogModel  # noqa: E402
from steerlab.atlas.service import AtlasService  # noqa: E402

D, L, H = 3, 2, 2
MODELS = {
    "toy": CatalogModel("toy", "org/Toy", L, D, H, True, base="toy-base", child="toy-small", lineage=True),
    "toy-base": CatalogModel("toy-base", "org/Toy-Base", L, D, H, False, post="toy", lineage=False),
    "toy-small": CatalogModel("toy-small", "org/Toy-Small", L, D, H, True, parent="toy", lineage=False),
}


def _keys(layer: int) -> dict[str, list[int]]:
    return {"layer": [layer] * 2 * D, "neuron": list(range(D)) * 2, "pole": [1] * D + [-1] * D}


def _write(root: Path, path: str, columns: dict[str, list[object]]) -> None:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), file)


def make_catalog(root: Path) -> None:
    for slug in ("toy", "toy-base"):
        for layer in range(L):
            n = 2 * D
            keys = _keys(layer)
            rows = [f"L{layer} n{i % D}{'+' if i < D else '-'}" for i in range(n)]
            _write(
                root,
                f"{slug}/neurons/acts/L{layer:02d}.parquet",
                {
                    **keys,
                    "act_top": [1.0 + i for i in range(n)],
                    "act_max": [2.0 + i for i in range(n)],
                    "rate": [0.01] * n,
                    "rate_dclm": [0.01] * n,
                    "rate_zh": [0.0] * n,
                    "rate_code": [0.02] * n,
                    "rate_chat": [0.0] * n,
                    "q50": [0.1] * n,
                    "q90": [0.2] * n,
                    "q99": [0.5] * n,
                    "q999": [0.9] * n,
                    "mean": [0.0] * n,
                    "std": [0.1] * n,
                    "pos_share": [0.5] * n,
                    "neg_lobe": [0.0] * n,
                    "cur_mode": [0.5] * n,
                    "next_mode": [0.1] * n,
                    "peak_tokens": ["' the'×3 | '7'×1"] * n,
                    "next_tokens": ["'\\n'×2"] * n,
                    "first_tokens": ["'¿' 0.5"] * n,
                    "contexts": [f"see {r}⟦ fire⟧ after ‖ more⟦ again⟧ end" for r in rows],
                    "top_val": [[3.0, 1.5]] * n,
                },
            )
            _write(
                root,
                f"{slug}/neurons/drivers/L{layer:02d}.parquet",
                {
                    **keys,
                    "drivers_str": [" a |  b"] * n,
                    "drivers_val": [[0.9, 0.8]] * n,
                    "drivers_sum_str": [" c"] * n,
                    "drivers_sum_val": [[2.0]] * n,
                },
            )
            _write(
                root,
                f"{slug}/neurons/reads/L{layer:02d}.parquet",
                {
                    **keys,
                    "w_norm": [0.5] * n,
                    "cos_write_gate": [0.1] * n,
                    "cos_write_up": [-0.1] * n,
                    "write_R_str": [" federal |  info"] * n,
                    "read_S_str": ["成 | 出"] * n,
                    "read_R_str": [""] * n,
                },
            )
            _write(
                root,
                f"{slug}/neurons/edges/L{layer:02d}.parquet",
                {
                    **keys,
                    "in_layer": [[0, -1]] * n,
                    "in_neuron": [[1, -1]] * n,
                    "in_pole": [[-1, 0]] * n,
                    "in_push": [[0.7, float("nan")]] * n,
                    "in_cos": [[0.2, float("nan")]] * n,
                    "out_layer": [[1, 1]] * n,
                    "out_neuron": [[2, 0]] * n,
                    "out_pole": [[1, -1]] * n,
                    "out_push": [[1.2, -0.4]] * n,
                    "out_cos": [[0.3, -0.1]] * n,
                },
            )
        _write(
            root,
            f"{slug}/heads/acts.parquet",
            {
                "layer": [0, 0, 1, 1],
                "head": [0, 1, 0, 1],
                "kv_head": [0, 0, 0, 0],
                "out_top": [4.0] * 4,
                "out_max": [5.0] * 4,
                "q50": [0.3] * 4,
                "q90": [1.0] * 4,
                "q99": [2.0] * 4,
                "q999": [3.0] * 4,
                "sink": [0.7] * 4,
                "prev": [0.1] * 4,
                "self": [0.05] * 4,
                "induction": [0.0] * 4,
                "duplicate": [0.0] * 4,
                "contexts": ["of⟪ which⟫ things are⟦ a⟧ case"] * 4,
                "top_val": [[4.5]] * 4,
                "peak_tokens": ["' but'×2"] * 4,
                "src_tokens": ["' only'×2"] * 4,
                "out_R_z_str": ["一旦 | x"] * 4,
            },
        )
    for layer in range(L):
        n = 2 * D
        _write(
            root,
            f"toy/neurons/labels/L{layer:02d}.parquet",
            {
                **_keys(layer),
                "fires": [f"label L{layer} #{i}" for i in range(n)],
                "fires_conf": [0.1 * (i + 1) for i in range(n)],
                "writes": ["closing brackets"] * n,
                "writes_conf": [0.2] * n,
                "note": [""] * n,
            },
        )
        _write(
            root,
            f"toy/neurons/lineage/L{layer:02d}.parquet",
            {
                **_keys(layer),
                "parent_layer": [-1] * n,
                "parent_neuron": [-1] * n,
                "parent_pole": [0] * n,
                "parent_status": [""] * n,
                "parent_by": [[]] * n,
                "child_layer": [layer] * D + [-1] * D,
                "child_neuron": [2] * D + [-1] * D,
                "child_pole": [1] * D + [0] * D,
                "child_status": ["identified"] * D + ["cut"] * D,
                "child_by": [["pearson"]] * D + [[]] * D,
            },
        )
    _write(
        root,
        "toy/heads/labels.parquet",
        {
            "layer": [0, 0, 1, 1],
            "head": [0, 1, 0, 1],
            "kv_head": [0] * 4,
            "fires": ["", "lists", "", "sink"],
            "fires_conf": [0.1, 0.5, 0.1, 0.2],
            "attends": ["previous token", "", "", ""],
            "attends_conf": [0.6, 0.1, 0.1, 0.1],
            "writes": [""] * 4,
            "writes_conf": [0.0] * 4,
            "note": [""] * 4,
        },
    )


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Catalog:
    root = tmp_path_factory.mktemp("catalog")
    make_catalog(root)
    return Catalog(LocalSource(root), models=MODELS)


def test_a_neuron_card_reads_every_table(catalog: Catalog) -> None:
    card = catalog.unit("toy", Address.parse("L1/n2-"))
    assert card["source"] == "catalog" and card["display"] == "L1·n2−"
    assert card["label"]["fires"] == "label L1 #5" and card["label"]["by"] == "Claude Haiku 5.5"
    assert card["label"]["lent_from"] is None
    assert card["stats"]["act_top"] == 6.0 and card["stats"]["layer_pct"] == 100.0
    assert card["contexts"][0]["pre"] == "see L1 n2-" and card["contexts"][0]["tok"] == " fire"
    assert card["contexts"][0]["pct"] > 99.9 and card["contexts"][0]["pct_text"].startswith("top")
    assert card["peak_tokens"] == [[" the", 3], ["7", 1]]
    assert card["drivers"]["at"] == [[" a", 0.9], [" b", 0.8]]
    assert {r["key"] for r in card["reads"]} == {"write_R", "read_S"}
    assert card["weights"]["w_norm"] == 0.5
    assert card["edges"]["in"] == [{"address": "L0/n1-", "push": pytest.approx(0.7), "cos": pytest.approx(0.2)}]
    assert [e["address"] for e in card["edges"]["out"]] == ["L1/n2+", "L1/n0-"]
    assert card["links"]["neuron_lens"].endswith("/toy/L1/n2")


def test_a_base_model_lends_its_post_trained_label(catalog: Catalog) -> None:
    card = catalog.unit("toy-base", Address.parse("L0/n1+"))
    assert card["label"]["fires"] == "label L0 #1" and card["label"]["lent_from"] == "toy"
    assert {(x["relation"], x["model"]) for x in card["lineage"]} == {("post-trained", "toy")}


def test_lineage_names_the_pruned_child_or_the_cut(catalog: Catalog) -> None:
    kept = catalog.lineage("toy", Address.parse("L0/n1+"))
    child = [x for x in kept if x["relation"] == "child"]
    assert child and child[0]["address"] == "L0/n2+" and child[0]["status"] == "identified"
    assert child[0]["by"] == ["pearson"]
    cut = catalog.lineage("toy", Address.parse("L0/n1-"))
    assert any(x["status"] == "cut" and x["address"] is None for x in cut)


def test_a_head_card(catalog: Catalog) -> None:
    card = catalog.unit("toy", Address.parse("L0/h0"))
    assert card["kind"] == "head" and card["label"]["attends"] == "previous token"
    assert card["contexts"][0]["src"] == " which" and card["contexts"][0]["tok"] == " a"
    assert card["stats"]["sink"] == pytest.approx(0.7) and card["src_tokens"] == [[" only", 2]]
    assert card["edges"]["in"] == []


def test_bad_addresses_say_why(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="layer 5"):
        catalog.unit("toy", Address.parse("L5/n0+"))
    with pytest.raises(ValueError, match="neuron 9"):
        catalog.unit("toy", Address.parse("L0/n9+"))
    with pytest.raises(ValueError, match="does not cover"):
        catalog.unit("nope", Address.parse("L0/n0+"))


def test_overview_ranks_each_layer_by_label_confidence(catalog: Catalog) -> None:
    ov = catalog.overview("toy", top=2)
    assert [len(layer["units"]) for layer in ov["layers"]] == [2, 2]
    first = ov["layers"][0]["units"][0]
    assert first["address"] == "L0/n2-" and first["title"] == "label L0 #5"
    assert ov["layers"][0]["heads"][0]["address"] == "L0/h0"
    assert catalog.overview("toy-base")["labels"]["lent_from"] == "toy"


def test_service_searches_and_walks_the_catalog(catalog: Catalog, tmp_path: Path) -> None:
    atlas = AtlasService(tmp_path, catalog=catalog)
    assert atlas.search("toy", "L1 n2-")["open"] == "L1/n2-"
    assert atlas.search("toy", "L1")["layer"] == 1
    found = atlas.search("toy", "brackets")
    assert found["total"] == 2 * D * L and found["searched"] == "labels"
    assert atlas.search("toy", "previous", kind="head")["results"][0]["address"] == "L0/h0"
    card = atlas.unit("toy", Address.parse("L0/n0+"))
    assert card["edges"]["in"][0]["title"] == "label L0 #4" and card["links"]["self"] == "/units/toy/L0/n0+"
    assert "trace" not in card["links"] and card["fireable"] is False
    graph = atlas.graph("toy", Address.parse("L0/n0+"), up=1, down=1, fan=2)
    assert graph["center"] == "L0/n0+" and len(graph["links"]) >= 2
    with pytest.raises(ValueError, match="fire runs the loaded model"):
        atlas.fire("toy", Address.parse("L0/n0+"), "hi")
