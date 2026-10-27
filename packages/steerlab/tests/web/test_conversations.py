"""Turn rules and visitor message limits."""

from __future__ import annotations

import pytest

from steerlab.web.conversations import ConversationStore, MessageRejected


def test_message_limits() -> None:
    store = ConversationStore()
    with pytest.raises(MessageRejected, match="Type a message"):
        store.reserve("a", "   ")
    with pytest.raises(MessageRejected, match="at most 2,000"):
        store.reserve("a", "x" * 2001)
    store.reserve("a", "x" * 2000)
    with pytest.raises(MessageRejected, match="Wait for the reply"):
        store.reserve("a", "again")
    store.reserve("b", "another visitor is fine")


def test_a_dropped_turn_removes_the_user_message_and_a_kept_one_stays() -> None:
    store = ConversationStore()
    store.reserve("a", "Hi")
    assert store.begin_turn("a", "Hi") == [{"role": "user", "content": "Hi"}]
    store.finish_turn("a", None)
    assert store.transcript("a") == []
    store.reserve("a", "Hi")
    store.begin_turn("a", "Hi")
    store.finish_turn("a", "Hel")
    assert store.transcript("a") == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hel"}]


def test_redo_and_clear_bump_the_version() -> None:
    store = ConversationStore()
    assert store.begin_redo("a") is None
    store.reserve("a", "Hi")
    store.begin_turn("a", "Hi")
    store.finish_turn("a", "Hello")
    assert store.begin_redo("a") == [{"role": "user", "content": "Hi"}]
    store.finish_redo("a", "Hello again")
    assert store.transcript("a")[-1] == {"role": "assistant", "content": "Hello again"}
    assert store.version("a") == 2
    store.clear_all()
    assert store.transcript("a") == [] and store.version("a") == 3


def test_a_redo_that_produced_nothing_puts_the_replaced_reply_back() -> None:
    """Otherwise the history ends user, user, and strict chat templates (Gemma) refuse the next message."""
    store = ConversationStore()
    store.reserve("a", "Hi")
    store.begin_turn("a", "Hi")
    store.finish_turn("a", "Hello")
    store.begin_redo("a")
    store.finish_redo("a", None)
    assert store.transcript("a") == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
