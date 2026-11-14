"""Graph diff on planted synthetic graphs, the loaders, and a reproduction of the RL-hack study's selection."""

from __future__ import annotations

import json
import random

import pytest

from circuitkit.diff import GraphSummary, Settings, diff_graphs, from_feature_lists, from_neuronpedia, load_summaries
from circuitkit.diff.graphs import save_summaries


def _graphs(
    n: int, seed: int, extra: dict[str, float] | None = None, drop: set[str] = frozenset()
) -> list[GraphSummary]:
    rng = random.Random(seed)
    units = [f"{rng.randrange(20)}:{i}" for i in range(200)]
    out = []
    for k in range(n):
        feats = {u: rng.uniform(0.001, 0.01) for u in units}
        feats = {u: v for u, v in feats.items() if u not in drop}
        feats.update({"5:9001": 0.05 + 0.001 * k, "6:9002": 0.03})
        if extra:
            feats.update({u: v * (1 + 0.01 * k) for u, v in extra.items()})
        edges = {("5:9001", "6:9002"): 2.0}
        if extra:
            edges[("5:9001", next(iter(extra)))] = 3.0
        out.append(
            GraphSummary(name=f"g{k}", features=feats, edges=edges, replacement_score=0.3, completeness_score=0.8)
        )
    return out


@pytest.fixture(scope="module")
def planted():
    before = _graphs(8, seed=1)
    after = _graphs(8, seed=1, extra={"9:7777": 0.08}, drop=set())
    for g in after:
        g.features.pop("6:9002")
        g.features["5:9001"] += 0.04  # strengthened, so both ends of the new edge are top units
    return diff_graphs(before, after, settings=Settings(top_k=5))


def test_planted_gain_is_top_and_gained(planted) -> None:
    top = planted.top[0]
    assert top.unit == "9:7777"
    assert top.status == "gained"
    assert top.present_before == 0 and top.present_after == 8
    assert top.sign_agreement == 1.0
    assert top.ci_low > 0


def test_planted_loss_is_lost(planted) -> None:
    assert "6:9002" in {c.unit for c in planted.lost}


def test_null_and_stability(planted) -> None:
    s = planted.summary
    assert s.change_p <= 0.05
    assert s.min_p == pytest.approx(3 / 257)
    assert s.split_half_overlap is not None and s.split_half_overlap > s.split_half_chance
    assert planted.top[0].p_null == pytest.approx(s.min_p)


def test_identical_graphs_show_no_change() -> None:
    g = _graphs(6, seed=2)
    rep = diff_graphs(g, g)
    assert rep.summary.n_gained == rep.summary.n_lost == rep.summary.n_up == 0
    assert rep.summary.change_score == 0
    assert rep.validity.ok


def test_edges_report_gained_edge(planted) -> None:
    gained = [e for e in planted.edges if e.status == "gained"]
    assert any(e.target == "9:7777" for e in gained)


def test_unpaired_uses_label_permutation() -> None:
    rep = diff_graphs(_graphs(6, 3), _graphs(5, 4, extra={"9:1": 0.1}), settings=Settings(n_null=200, min_present=4))
    assert not rep.summary.paired
    assert rep.top[0].unit == "9:1"


def test_validity_warns_on_worse_reconstruction() -> None:
    b, a = _graphs(4, 5), _graphs(4, 5)
    for g in a:
        g.replacement_score = 0.1
    rep = diff_graphs(b, a, fvu_before=[0.2, 0.3], fvu_after=[0.2, 0.5])
    assert not rep.validity.ok
    assert any("replacement" in w for w in rep.validity.warnings)
    assert any("FVU" in w for w in rep.validity.warnings)


def test_control_and_replication() -> None:
    before, after = _graphs(8, 1), _graphs(8, 1, extra={"9:7777": 0.08})
    rep = diff_graphs(before, after, control=(before, before), replication=(before, after), settings=Settings(top_k=3))
    top = rep.top[0]
    assert top.specific is True and top.replicated is True
    assert rep.summary.control_specific is not None


def test_empty_side_raises() -> None:
    with pytest.raises(ValueError):
        diff_graphs([], _graphs(1, 0))


def test_save_load_roundtrip(tmp_path) -> None:
    g = _graphs(2, 0)
    for suffix in (".json", ".json.gz"):
        p = tmp_path / f"s{suffix}"
        save_summaries(p, {"before": g})
        back = load_summaries(p)["before"]
        assert back[0].features == g[0].features and back[0].edges == g[0].edges
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError):
        load_summaries(tmp_path / "bad.json")


def test_from_feature_lists_sums_positions() -> None:
    s = from_feature_lists(
        "x", [(1, 2, 7), (1, 3, 7), (2, 3, 8)], [0.1, 0.2, 0.3], edges=[(2, 0, 1.0), (2, 1, 0.5), (1, 0, 9.0)]
    )
    assert s.features == {"1:7": pytest.approx(0.3), "2:8": 0.3}
    assert s.edges == {("1:7", "2:8"): 1.5}  # the 1:7 -> 1:7 edge is dropped


def test_from_neuronpedia() -> None:
    data = {
        "metadata": {"slug": "demo", "scan": "gemma-2-2b", "prompt": "hi"},
        "nodes": [
            {"node_id": "27_100_3", "feature_type": "logit", "influence": None},
            {"node_id": "20_15589_2", "feature_type": "cross layer transcoder", "influence": 0.4},
            {"node_id": "20_15589_1", "feature_type": "cross layer transcoder", "influence": 0.6},
            {"node_id": "E_1_0", "feature_type": "embedding", "influence": 0.7},
            {"node_id": "4_12_1", "feature_type": "cross layer transcoder", "influence": 0.9},
        ],
        "links": [{"source": "4_12_1", "target": "20_15589_2", "weight": 1.5}],
    }
    s = from_neuronpedia(data)
    assert s.features["20:15589"] == pytest.approx(0.6)
    assert s.features["4:12"] == pytest.approx(0.2)
    assert s.edges == {("4:12", "20:15589"): 1.5}
    assert s.prompt == "hi"


def test_reproduces_rl_hack_selection(data_dir) -> None:
    """On the RL-hack study's 48 graphs, the diff picks the study's ten features, 9 hack-specific, 6 replicated."""
    g = load_summaries(data_dir / "rl_hack_graphs.json.gz")
    rep = diff_graphs(
        g["installed/before"],
        g["installed/after"],
        control=(g["control/before"], g["control/after"]),
        replication=(g["replication/before"], g["replication/after"]),
        fvu_before=g["installed/before"][0].meta["fvu_by_layer"],
        fvu_after=g["installed/after"][0].meta["fvu_by_layer"],
    )
    study = [
        "21:154092",
        "16:146794",
        "16:58137",
        "26:88470",
        "26:162002",
        "19:143996",
        "19:141589",
        "25:137808",
        "19:78662",
        "22:160272",
    ]
    assert [c.unit for c in rep.top] == study
    assert rep.summary.control_specific == 9
    assert rep.summary.replication_overlap == 6
    assert rep.validity.ok
    assert {(e.source, e.target) for e in rep.edges[:5]} >= {("16:58137", "21:154092"), ("19:141589", "21:154092")}
    json.loads(rep.model_dump_json())
