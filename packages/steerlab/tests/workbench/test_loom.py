"""The loom: building, rerolling, forking, comparing and saving a branching conversation."""

from __future__ import annotations

from pathlib import Path

import pytest

from steerlab.workbench.loom import Loom, LoomError, LoomStore


def tree() -> tuple[Loom, dict[str, str]]:
    loom = Loom(title="t", loom_id="loom-1")
    system = loom.add("system", "be brief")
    hi = loom.add("user", "hi", parent=system.id)
    plain = loom.add("assistant", "hello", parent=hi.id, meta={"steer": []})
    steered = loom.add("assistant", "bonjour", parent=hi.id, meta={"steer": [{"probe": "french", "strength": 4}]})
    more = loom.add("user", "and?", parent=steered.id)
    return loom, {"system": system.id, "hi": hi.id, "plain": plain.id, "steered": steered.id, "more": more.id}


def test_path_and_messages_follow_the_branch() -> None:
    loom, ids = tree()
    assert [n.id for n in loom.path(ids["more"])] == [ids["system"], ids["hi"], ids["steered"], ids["more"]]
    assert loom.messages(ids["plain"]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_reroll_goes_under_the_same_parent_and_fork_makes_a_sibling() -> None:
    loom, ids = tree()
    assert loom.reroll_parent(ids["plain"]) == ids["hi"]
    with pytest.raises(LoomError):
        loom.reroll_parent(ids["hi"])
    fork = loom.fork(ids["hi"], "hey")
    assert fork.parent == ids["system"] and loom.siblings(fork.id) == [ids["hi"], fork.id]
    assert fork.meta["forked_from"] == ids["hi"]
    assert loom.leaf(ids["hi"]) == ids["more"]


def test_compare_splits_at_the_common_ancestor() -> None:
    loom, ids = tree()
    view = loom.compare(ids["plain"], ids["more"])
    assert view["ancestor"] == ids["hi"]
    assert [n["id"] for n in view["trunk"]] == [ids["system"], ids["hi"]]
    assert [n["id"] for n in view["a"]] == [ids["plain"]]
    assert [n["id"] for n in view["b"]] == [ids["steered"], ids["more"]]


def test_remove_drops_the_subtree() -> None:
    loom, ids = tree()
    assert sorted(loom.remove(ids["steered"])) == sorted([ids["steered"], ids["more"]])
    assert loom.get(ids["hi"]).children == [ids["plain"]]
    loom.check()


def test_round_trip_keeps_ids_and_the_counter(tmp_path: Path) -> None:
    loom, _ = tree()
    again = Loom.from_dict(loom.to_dict())
    assert again.to_dict() == loom.to_dict()
    assert again.add("user", "next").id == "n6"  # the counter survives, so ids are never reused
    store = LoomStore(tmp_path)
    store.save(loom)
    assert store.load("loom-1").to_dict() == loom.to_dict()
    assert [row["id"] for row in store.list()] == ["loom-1"]
    store.delete("loom-1")
    assert store.list() == []


def test_bad_trees_and_ids_are_refused(tmp_path: Path) -> None:
    loom, ids = tree()
    data = loom.to_dict()
    data["nodes"][ids["hi"]]["children"] = []
    with pytest.raises(LoomError):
        Loom.from_dict(data)
    with pytest.raises(LoomError):
        loom.add("robot", "x")
    with pytest.raises(LoomError):
        loom.add("user", "x", parent="n99")
    with pytest.raises(LoomError):
        LoomStore(tmp_path).load("../etc")
