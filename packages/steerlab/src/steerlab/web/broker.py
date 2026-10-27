"""The admin feed: every worker event goes to each open admin tab, and a new tab starts from a snapshot.

Think of the broker as the control room's wall of monitors. The worker publishes events (`state`, `user`, `start`,
`token`, `marker`, `end`, `redo`, `reset`, `new`, `notice`); the broker keeps its own copy of what the admin should
see, built from those same events, and copies each event into every tab's queue. A tab that opens late, reconnects,
or fell behind gets the whole picture at once as its first event, a `snapshot`, so there is no replay to get wrong.

Each tab's queue is bounded. A tab that stops reading (a sleeping laptop) is dropped instead of slowing the worker:
its stream ends, the browser's `EventSource` reconnects, and the new connection starts from a fresh snapshot.
"""

from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

Event = dict[str, Any]

TAB_QUEUE_SIZE = 2000
NOTICES_KEPT = 20


@dataclass
class Subscription:
    """One admin tab's feed. `get` returns the next event, or None when the server stops."""

    events: queue.Queue[Event | None] = field(default_factory=lambda: queue.Queue(TAB_QUEUE_SIZE))
    dropped: bool = False

    def get(self, timeout: float | None = None) -> Event | None:
        """The next event; raises `queue.Empty` after `timeout` seconds with nothing to send (time for a ping)."""
        return self.events.get(timeout=timeout)


@dataclass
class _Mirror:
    """The admin's view: the latest state, every conversation's turns (replies with tokens and markers), notices."""

    state: Event = field(default_factory=dict)
    conversations: dict[str, dict[str, Any]] = field(default_factory=dict)
    notices: list[Event] = field(default_factory=list)

    def turns(self, conversation_id: str) -> list[dict[str, Any]]:
        if conversation_id not in self.conversations:
            label = len(self.conversations) + 1  # "Visitor N" by first-seen order
            self.conversations[conversation_id] = {"id": conversation_id, "label": label, "turns": []}
        turns: list[dict[str, Any]] = self.conversations[conversation_id]["turns"]
        return turns

    def apply(self, event: Event) -> None:
        kind = event["type"]
        if kind == "state":
            self.state = {key: value for key, value in event.items() if key != "type"}
        elif kind == "notice":
            self.notices = [*self.notices, event][-NOTICES_KEPT:]
        elif kind == "reset":
            for conversation in self.conversations.values():
                conversation["turns"] = []
        elif kind == "new":
            self.turns(event["conversation"]).clear()
        elif kind == "user":
            self.turns(event["conversation"]).append({"role": "user", "text": event["text"]})
        elif kind == "start":
            reply: dict[str, Any] = {"role": "assistant", "tokens": [], "markers": [], "finish": None}
            turns = self.turns(event["conversation"])
            original = self.conversations[event["conversation"]].pop("replaced", None)
            if original is not None:  # a redo: keep the reply it replaces, for the Original/Steered compare view
                reply["original"] = original
            turns.append(reply)
        elif kind in ("token", "marker", "end", "redo"):
            self._apply_to_reply(event)

    def _apply_to_reply(self, event: Event) -> None:
        turns = self.turns(event["conversation"])
        if not turns or turns[-1]["role"] != "assistant":
            return
        reply, kind = turns[-1], event["type"]
        if kind == "token":
            reply["tokens"].append([event["index"], event["text"]])
        elif kind == "marker":
            reply["markers"].append({key: value for key, value in event.items() if key != "type"})
        elif kind == "redo":
            replaced = turns.pop()
            replaced.pop("original", None)
            self.conversations[event["conversation"]]["replaced"] = replaced
        elif event.get("dropped"):  # the turn is gone from the model's history too
            turns.pop()
            if turns and turns[-1]["role"] == "user" and not event.get("keep_user"):
                turns.pop()
        else:
            reply["finish"] = event["reason"]

    def snapshot(self) -> Event:
        return {
            "type": "snapshot",
            "state": copy.deepcopy(self.state),
            "conversations": copy.deepcopy(list(self.conversations.values())),
            "notices": copy.deepcopy(self.notices),
        }


class Broker:
    """Thread-safe fan-out. `publish` never blocks: a full tab is dropped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mirror = _Mirror()
        self._tabs: list[Subscription] = []

    def publish(self, event: Event) -> None:
        with self._lock:
            self._mirror.apply(event)
            for tab in list(self._tabs):
                try:
                    tab.events.put_nowait(event)
                except queue.Full:
                    tab.dropped = True
                    self._tabs.remove(tab)

    def subscribe(self) -> Subscription:
        """A new tab's feed, starting with a snapshot taken atomically with respect to `publish`."""
        tab = Subscription()
        with self._lock:
            tab.events.put_nowait(self._mirror.snapshot())
            self._tabs.append(tab)
        return tab

    def unsubscribe(self, tab: Subscription) -> None:
        with self._lock:
            if tab in self._tabs:
                self._tabs.remove(tab)

    def close(self) -> None:
        """End every open tab's stream (server shutdown): each tab's next `get` returns None."""
        with self._lock:
            tabs, self._tabs = self._tabs, []
        for tab in tabs:
            try:
                tab.events.put_nowait(None)
            except queue.Full:
                tab.dropped = True

    def snapshot(self) -> Event:
        with self._lock:
            return self._mirror.snapshot()
