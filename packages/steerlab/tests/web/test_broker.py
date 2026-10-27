"""The admin feed: snapshot first, then every event; a tab that stops reading is dropped."""

from __future__ import annotations

import queue

import pytest

from steerlab.web import broker as broker_module
from steerlab.web.broker import Broker


def test_a_new_tab_starts_with_a_snapshot_of_everything_so_far() -> None:
    broker = Broker()
    broker.publish({"type": "state", "status": "ready", "model": "qwen8b"})
    broker.publish({"type": "user", "conversation": "a", "text": "Hi"})
    broker.publish({"type": "start", "conversation": "a"})
    broker.publish({"type": "marker", "conversation": "a", "token_index": 0, "applied": []})
    broker.publish({"type": "token", "conversation": "a", "index": 0, "text": "Hel"})
    tab = broker.subscribe()
    snapshot = tab.get(timeout=1)
    assert snapshot is not None and snapshot["type"] == "snapshot"
    assert snapshot["state"] == {"status": "ready", "model": "qwen8b"}
    [conversation] = snapshot["conversations"]
    assert conversation["label"] == 1
    assert conversation["turns"][1]["tokens"] == [[0, "Hel"]]
    assert conversation["turns"][1]["markers"] == [{"conversation": "a", "token_index": 0, "applied": []}]
    broker.publish({"type": "token", "conversation": "a", "index": 1, "text": "lo"})
    assert tab.get(timeout=1) == {"type": "token", "conversation": "a", "index": 1, "text": "lo"}


def test_a_dropped_turn_leaves_the_mirror_and_a_stopped_one_stays() -> None:
    broker = Broker()
    for conversation, end in (("a", {"reason": "disconnected", "dropped": True}), ("b", {"reason": "stopped"})):
        broker.publish({"type": "user", "conversation": conversation, "text": "Hi"})
        broker.publish({"type": "start", "conversation": conversation})
        broker.publish({"type": "end", "conversation": conversation, **end})
    a, b = broker.snapshot()["conversations"]
    assert a["turns"] == []
    assert [turn["role"] for turn in b["turns"]] == ["user", "assistant"] and b["turns"][1]["finish"] == "stopped"
    broker.publish({"type": "reset"})
    assert all(conversation["turns"] == [] for conversation in broker.snapshot()["conversations"])


def test_a_tab_that_falls_behind_is_dropped_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_module, "TAB_QUEUE_SIZE", 3)
    broker = Broker()
    slow = broker.subscribe()  # holds the snapshot: two more events fit, the third overflows
    for index in range(5):
        broker.publish({"type": "notice", "level": "info", "message": str(index)})
    assert slow.dropped
    fresh = broker.subscribe()
    snapshot = fresh.get(timeout=1)
    assert snapshot is not None and [notice["message"] for notice in snapshot["notices"]] == ["0", "1", "2", "3", "4"]


def test_close_ends_every_open_stream() -> None:
    broker = Broker()
    tab = broker.subscribe()
    tab.get(timeout=1)
    broker.close()
    assert tab.get(timeout=1) is None
    with pytest.raises(queue.Empty):
        tab.get(timeout=0.01)


def test_a_redo_keeps_the_reply_it_replaces_for_comparison() -> None:
    broker = Broker()
    broker.publish({"type": "user", "conversation": "a", "text": "Hi"})
    broker.publish({"type": "start", "conversation": "a"})
    broker.publish({"type": "token", "conversation": "a", "index": 0, "text": "Hello"})
    broker.publish({"type": "end", "conversation": "a", "reason": "eos"})
    broker.publish({"type": "redo", "conversation": "a"})
    broker.publish({"type": "start", "conversation": "a"})
    broker.publish({"type": "token", "conversation": "a", "index": 0, "text": "你好"})
    [conversation] = broker.snapshot()["conversations"]
    reply = conversation["turns"][-1]
    assert reply["tokens"] == [[0, "你好"]]
    assert reply["original"]["tokens"] == [[0, "Hello"]] and reply["original"]["finish"] == "eos"
