"""The torch-free pieces of `steerlab.atlas`: addresses, card parsing, percentiles, scores, search, the wiring walk."""

from __future__ import annotations

import pytest

from steerlab.atlas.address import Address, model_slug, parse_layer, unit_href
from steerlab.atlas.cards import (
    describe_percentile,
    neuron_score,
    parse_contexts,
    parse_counted,
    parse_scored,
    percentile,
    pole_score,
    split_tokens,
    strength,
)
from steerlab.atlas.models import CATALOG_MODELS, band, catalog_model, neuron_lens_url
from steerlab.atlas.search import SearchIndex
from steerlab.atlas.wiring import walk


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("L8/n344+", Address.neuron(8, 344, 1)),
        ("L8/n344-", Address.neuron(8, 344, -1)),
        ("L8 n344 −", Address.neuron(8, 344, -1)),
        ("l8n344", Address.neuron(8, 344, 1)),
        ("L8·n344+", Address.neuron(8, 344, 1)),
        ("L8/n344%2B", Address.neuron(8, 344, 1)),
        ("L20/h17", Address.head(20, 17)),
        ("layer 20 head 17", Address.head(20, 17)),
    ],
)
def test_addresses_parse_the_ways_people_type_them(text: str, want: Address) -> None:
    assert Address.parse(text) == want


def test_addresses_round_trip_and_display() -> None:
    a = Address.neuron(5, 7, -1)
    assert a.path() == "L5/n7-" and str(a) == "L5·n7−" and Address.parse(a.path()) == a
    assert a.other_pole() == Address.neuron(5, 7, 1)
    assert Address.head(3, 1).path() == "L3/h1"
    assert Address.try_parse("closing brackets") is None
    assert parse_layer("L8") == 8 and parse_layer("layer 12") == 12 and parse_layer("L8 n3") is None
    with pytest.raises(ValueError):
        Address.parse("n344")


def test_model_slugs_match_the_catalog_folders() -> None:
    assert model_slug("Qwen/Qwen3-0.6B") == "qwen3-0.6b"
    assert model_slug("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b-instruct"
    assert catalog_model("Qwen/Qwen3-8B-Base") is CATALOG_MODELS["qwen3-8b-base"]
    assert catalog_model("Qwen/Qwen2.5-0.5B-Instruct") is None
    for model in CATALOG_MODELS.values():
        assert model_slug(model.hf_id) == model.slug
    base = CATALOG_MODELS["qwen3-0.6b-base"]
    assert base.label_slug == "qwen3-0.6b" and base.parent == "qwen3-1.7b-base"
    assert neuron_lens_url("qwen3-8b", "L8/n344+") == "https://neuron-lens.com/qwen3-8b/L8/n344"
    assert [band(layer, 36) for layer in (0, 3, 4, 8, 9, 20, 21, 35)] == [
        "sensory",
        "sensory",
        "association",
        "association",
        "workspace",
        "workspace",
        "motor",
        "motor",
    ]


def test_contexts_split_around_the_firing_and_attended_tokens() -> None:
    raw = "Page Scan⏎22With⟦ Scan⟧⏎23 ‖ of⟪ which⟫ functions are⟦ a⟧ special case"
    pieces = parse_contexts(raw, [3.5, 2.0])
    assert pieces[0] == {"pre": "Page Scan⏎22With", "tok": " Scan", "post": "⏎23", "value": 3.5}
    assert pieces[1]["src"] == " which" and pieces[1]["mid"] == " functions are" and pieces[1]["tok"] == " a"
    assert pieces[1]["pre"] == "of" and pieces[1]["value"] == 2.0


def test_token_lists_parse() -> None:
    assert parse_counted("'7'×4 | ' to'×3 | '\\n'×2") == [["7", 4], [" to", 3], ["⏎", 2]]
    assert parse_scored("'¿' 0.668 | ' ¿' 0.408") == [["¿", 0.668], [" ¿", 0.408]]
    assert split_tokens(" federal |  info | \t") == [" federal", " info", "⇥"]
    assert split_tokens(None) == []


def test_percentiles_interpolate_the_quantiles() -> None:
    stats = {"q50": 1.0, "q90": 2.0, "q99": 3.0, "q999": 4.0, "act_max": 5.0}
    assert percentile(0.5, stats) == pytest.approx(25.0)
    assert percentile(1.5, stats) == pytest.approx(70.0)
    assert percentile(3.5, stats) == pytest.approx(99.45)
    assert percentile(9.0, stats) == 100.0
    assert percentile(-1.0, stats) == 0.0
    assert percentile(1.0, {"q50": 1.0}) is None
    assert describe_percentile(99.95) == "top 0.05%" and describe_percentile(30) == "30th percentile"


def test_scores_follow_the_easiest_ranking() -> None:
    assert pole_score(0.9, 0.3) == pytest.approx(1.0)
    assert neuron_score([1.0, 0.4]) == pytest.approx(1.1)
    assert strength("neuron", 0.0) == 0.0 and strength("neuron", 5 / 3) == 1.0


def _docs() -> list[tuple[Address, str, str, float]]:
    return [
        (Address.neuron(1, 0, 1), "closing brackets in code", "writes ) ]", 0.9),
        (Address.neuron(2, 5, -1), "brackets and braces", "", 0.5),
        (Address.neuron(3, 2, 1), "words about closing a deal", "", 0.7),
        (Address.head(4, 1), "attends to opening brackets", "", 0.2),
    ]


def test_search_ranks_units_with_every_word_first() -> None:
    index = SearchIndex.build(_docs())
    found = index.search("closing brackets")
    assert [r["address"] for r in found["results"]][:1] == ["L1/n0+"]
    assert found["results"][0]["all_words"] and not found["results"][-1]["all_words"]
    assert index.search("brackets", kind="head")["results"][0]["address"] == "L4/h1"
    assert index.search('"brackets and"')["total"] == 1


def test_search_tolerates_typos_and_suggests() -> None:
    index = SearchIndex.build(_docs())
    found = index.search("brakets")
    assert found["approx"] == {"braket": ["bracket"]} and found["total"] == 3
    assert index.search("zzzz")["results"] == []


def test_walk_follows_edges_layer_by_layer() -> None:
    edges = {
        "L2/n0+": {
            "in": [{"address": "L1/n3+", "push": 2.0, "cos": 0.4}, {"address": "L0/n1-", "push": -0.5}],
            "out": [{"address": "L3/n9+", "push": 1.0, "cos": 0.2}],
        },
        "L1/n3+": {"in": [{"address": "L0/n2+", "push": 1.0}], "out": []},
    }
    graph = walk(
        lambda a: edges.get(a.path(), {"in": [], "out": []}),
        lambda a: f"t{a.layer}",
        Address.parse("L2/n0+"),
        up=2,
        down=1,
        fan=1,
    )
    assert [n["address"] for n in graph["nodes"]] == ["L0/n2+", "L1/n3+", "L2/n0+", "L3/n9+"]
    assert {(link["source"], link["target"]) for link in graph["links"]} == {
        ("L1/n3+", "L2/n0+"),
        ("L0/n2+", "L1/n3+"),
        ("L2/n0+", "L3/n9+"),
    }
    assert graph["nodes"][0]["hop"] == -2 and graph["nodes"][0]["title"] == "t0"


def test_unit_href_is_the_one_link_every_room_uses() -> None:
    assert unit_href("Qwen/Qwen3-8B", 8, 344, 1) == "/units/qwen3-8b/L8/n344+"
    assert unit_href("Qwen/Qwen2.5-0.5B-Instruct", 3, 7, -1) == "/units/qwen2.5-0.5b-instruct/L3/n7-"
    assert unit_href("google/gemma-4-12B", 20, 17) == "/units/gemma-4-12b/L20/h17"
