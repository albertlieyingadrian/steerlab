"""Per-visitor conversations and their turn rules.

Each visitor tab has a random conversation id; the store maps it to the turns the model has seen. The rules keep the
roles alternating, which the chat templates require:

- a reply cut off by a visitor disconnect, a reset or a failure removes its user turn (as if it was never sent);
- a reply ended by the admin's stop keeps the user turn with the partial reply;
- redo removes the last assistant turn so it can be generated again.

`version` increases whenever the stored turns change behind the visitor's back (admin clear, swap, redo), so an idle
visitor tab polling `/api/status` knows to re-fetch its transcript. Nothing survives a restart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

Message = dict[str, str]

MAX_MESSAGE_CHARS = 2000
MAX_PROMPT_TOKENS = 6000
FULL_MESSAGE = "This conversation is full. Start a new one."


class MessageRejected(ValueError):
    """A visitor message the store refuses (empty, too long, or one already waiting). The message is shown as is."""


@dataclass
class Conversation:
    id: str
    messages: list[Message] = field(default_factory=list)
    version: int = 0
    pending: bool = False
    replaced: Message | None = None


class ConversationStore:
    """Thread-safe: the HTTP side reads transcripts and checks messages while the worker thread writes turns."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversations: dict[str, Conversation] = {}

    def _get(self, conversation_id: str) -> Conversation:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            conversation = self._conversations[conversation_id] = Conversation(conversation_id)
        return conversation

    def reserve(self, conversation_id: str, text: str) -> None:
        """Accept a new visitor message for queueing: one pending message per visitor, 1..2,000 characters."""
        if not text.strip():
            raise MessageRejected("Type a message first.")
        if len(text) > MAX_MESSAGE_CHARS:
            raise MessageRejected(f"Messages can be at most {MAX_MESSAGE_CHARS:,} characters.")
        with self._lock:
            conversation = self._get(conversation_id)
            if conversation.pending:
                raise MessageRejected("Wait for the reply to your last message first.")
            conversation.pending = True

    def release(self, conversation_id: str) -> None:
        """The queued message will not run (queue full, reset, or the visitor left before it started)."""
        with self._lock:
            self._get(conversation_id).pending = False

    def begin_turn(self, conversation_id: str, text: str) -> list[Message]:
        """Add the user turn when its reply starts; returns the history to render (without the system prompt)."""
        with self._lock:
            conversation = self._get(conversation_id)
            conversation.messages.append({"role": "user", "content": text})
            return [dict(message) for message in conversation.messages]

    def finish_turn(self, conversation_id: str, reply: str | None) -> None:
        """Store the reply (`None`: the turn is dropped, user message included) and free the visitor's slot."""
        with self._lock:
            conversation = self._get(conversation_id)
            if reply is None:
                if conversation.messages and conversation.messages[-1]["role"] == "user":
                    conversation.messages.pop()
            else:
                conversation.messages.append({"role": "assistant", "content": reply})
            conversation.pending = False

    def begin_redo(self, conversation_id: str) -> list[Message] | None:
        """Remove the last assistant turn for regeneration; None when there is no reply to redo."""
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None or conversation.pending or not conversation.messages:
                return None
            if conversation.messages[-1]["role"] != "assistant":
                return None
            conversation.replaced = conversation.messages.pop()
            conversation.pending = True
            conversation.version += 1
            return [dict(message) for message in conversation.messages]

    def finish_redo(self, conversation_id: str, reply: str | None) -> None:
        """A redo that produced nothing (failed, full, reset) restores the replaced reply, so roles keep alternating."""
        with self._lock:
            conversation = self._get(conversation_id)
            if reply is not None:
                conversation.messages.append({"role": "assistant", "content": reply})
            elif conversation.replaced is not None and conversation.messages:
                conversation.messages.append(conversation.replaced)
            conversation.replaced = None
            conversation.pending = False
            conversation.version += 1

    def clear_all(self) -> None:
        """Empty every conversation (admin clear, model swap) and bump each version so idle tabs notice."""
        with self._lock:
            for conversation in self._conversations.values():
                conversation.messages.clear()
                conversation.pending = False
                conversation.version += 1

    def reset(self, conversation_id: str) -> None:
        """The visitor started a new conversation under the same id."""
        with self._lock:
            conversation = self._get(conversation_id)
            conversation.messages.clear()
            conversation.version += 1

    def transcript(self, conversation_id: str) -> list[Message]:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            return [dict(message) for message in conversation.messages] if conversation else []

    def version(self, conversation_id: str) -> int:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            return conversation.version if conversation else 0
