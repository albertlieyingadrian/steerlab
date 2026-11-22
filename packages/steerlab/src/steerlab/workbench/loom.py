"""The loom: a conversation as a branching tree of turns, so you can reroll, fork and compare branches.

Think of a chess analysis board. The main line is one path from the first move, but at any move you can try another
one and the board keeps both. Here every turn (a user message or a model reply) is a node; its parent is the turn
before it, and the conversation the model sees is the path from the root to the node you are on.

    loom = Loom()
    hi = loom.add("user", "Hi there", parent=None)
    a = loom.add("assistant", "Hello!", parent=hi.id, meta={"steer": []})
    b = loom.add("assistant", "Bonjour !", parent=hi.id, meta={"steer": [{"direction": "french", "strength": 4}]})
    loom.messages(b.id)          # [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": ...}]
    loom.compare(a.id, b.id)     # the shared trunk, then each branch from where they split

- reroll: a new reply under the same parent (a sibling of the old reply);
- fork: a new turn beside an earlier one (edit what you said, keep both versions);
- compare: two nodes side by side from their last common ancestor (steered against unsteered, say).

Pure Python and JSON: no torch, so the tree is cheap to test and to save. Node ids are short counters ("n1", "n2"),
stable across save and load, so a URL or an exported file can name a node.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROLES: tuple[str, ...] = ("system", "user", "assistant")


@dataclass
class Node:
    """One turn. `meta` holds what produced a reply: steering, edits, sampling, and the per-token readings."""

    id: str
    role: str
    content: str
    parent: str | None
    children: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoomError(ValueError):
    """An operation on the tree that cannot work (unknown node, wrong role)."""


class Loom:
    """The tree. Roots are turns with no parent (a conversation can start with a system or a user turn)."""

    def __init__(self, title: str = "", loom_id: str = "") -> None:
        self.id = loom_id
        self.title = title
        self.nodes: dict[str, Node] = {}
        self.roots: list[str] = []
        self._counter = 0
        self.updated = time.time()

    # Building

    def add(self, role: str, content: str, parent: str | None = None, meta: dict[str, Any] | None = None) -> Node:
        """A new turn under `parent` (None: a new root). Returns the node."""
        if role not in ROLES:
            raise LoomError(f"role must be one of {ROLES}, not {role!r}")
        if parent is not None:
            self.get(parent)
        self._counter += 1
        node = Node(id=f"n{self._counter}", role=role, content=content, parent=parent, meta=dict(meta or {}))
        self.nodes[node.id] = node
        if parent is None:
            self.roots.append(node.id)
        else:
            self.nodes[parent].children.append(node.id)
        self.updated = time.time()
        return node

    def reroll_parent(self, node_id: str) -> str:
        """Where a reroll of reply `node_id` goes: its parent, so the new reply becomes a sibling."""
        node = self.get(node_id)
        if node.role != "assistant":
            raise LoomError(f"only a reply can be rerolled; {node_id} is a {node.role} turn")
        if node.parent is None:
            raise LoomError(f"{node_id} has no turn before it to reply to")
        return node.parent

    def fork(self, node_id: str, content: str) -> Node:
        """A sibling of `node_id` with new text: the same place in the conversation, said differently."""
        node = self.get(node_id)
        return self.add(node.role, content, parent=node.parent, meta={"forked_from": node_id})

    def remove(self, node_id: str) -> list[str]:
        """Delete a node and everything under it. Returns the removed ids."""
        node = self.get(node_id)
        removed: list[str] = []
        stack = [node_id]
        while stack:
            current = stack.pop()
            removed.append(current)
            stack.extend(self.nodes[current].children)
        for gone in removed:
            del self.nodes[gone]
        if node.parent is None:
            self.roots.remove(node_id)
        else:
            self.nodes[node.parent].children.remove(node_id)
        self.updated = time.time()
        return removed

    # Reading

    def get(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise LoomError(f"no node {node_id!r} in this loom") from None

    def path(self, node_id: str) -> list[Node]:
        """Root to `node_id`, inclusive: the conversation as the model sees it at that node."""
        out: list[Node] = []
        current: str | None = node_id
        while current is not None:
            node = self.get(current)
            out.append(node)
            current = node.parent
        return out[::-1]

    def messages(self, node_id: str) -> list[dict[str, str]]:
        return [{"role": node.role, "content": node.content} for node in self.path(node_id)]

    def siblings(self, node_id: str) -> list[str]:
        node = self.get(node_id)
        return list(self.roots if node.parent is None else self.nodes[node.parent].children)

    def leaf(self, node_id: str) -> str:
        """Follow the newest child down from `node_id` to a leaf (the default continuation of a branch)."""
        current = self.get(node_id)
        while current.children:
            current = self.nodes[current.children[-1]]
        return current.id

    def common_ancestor(self, a: str, b: str) -> str | None:
        ancestors = {node.id for node in self.path(a)}
        for node in reversed(self.path(b)):
            if node.id in ancestors:
                return node.id
        return None

    def compare(self, a: str, b: str) -> dict[str, Any]:
        """The shared trunk and the two branches from where they split, for a side-by-side view."""
        split = self.common_ancestor(a, b)
        path_a, path_b = self.path(a), self.path(b)
        cut = 0 if split is None else [node.id for node in path_a].index(split) + 1
        return {
            "ancestor": split,
            "trunk": [node.to_dict() for node in path_a[:cut]],
            "a": [node.to_dict() for node in path_a[cut:]],
            "b": [node.to_dict() for node in path_b[cut:]],
        }

    # Saving

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "counter": self._counter,
            "roots": list(self.roots),
            "updated": self.updated,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Loom:
        loom = cls(title=str(data.get("title", "")), loom_id=str(data.get("id", "")))
        for node_id, raw in data.get("nodes", {}).items():
            loom.nodes[node_id] = Node(
                id=node_id,
                role=str(raw["role"]),
                content=str(raw["content"]),
                parent=raw.get("parent"),
                children=list(raw.get("children", [])),
                meta=dict(raw.get("meta", {})),
                created=float(raw.get("created", 0.0)),
            )
        loom.roots = list(data.get("roots", []))
        loom._counter = int(data.get("counter", len(loom.nodes)))
        loom.updated = float(data.get("updated", time.time()))
        loom.check()
        return loom

    def check(self) -> None:
        """Every parent/child link points both ways and every node is reachable from a root."""
        for node in self.nodes.values():
            if node.parent is not None and node.id not in self.get(node.parent).children:
                raise LoomError(f"{node.id} names parent {node.parent} but is not among its children")
            for child in node.children:
                if self.get(child).parent != node.id:
                    raise LoomError(f"{child} is a child of {node.id} but names another parent")
        reachable: set[str] = set()
        stack = list(self.roots)
        while stack:
            current = stack.pop()
            reachable.add(current)
            stack.extend(self.get(current).children)
        if reachable != set(self.nodes):
            raise LoomError(f"unreachable nodes: {sorted(set(self.nodes) - reachable)}")


class LoomStore:
    """Looms saved as JSON files, one per conversation, in `<root>/<id>.json` (written to a temp file, then renamed)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, loom_id: str) -> Path:
        if not loom_id or not all(c.isalnum() or c in "-_" for c in loom_id):
            raise LoomError(f"bad loom id {loom_id!r}")
        return self.root / f"{loom_id}.json"

    def new(self, title: str = "") -> Loom:
        loom = Loom(title=title or "New conversation", loom_id=f"loom-{time.time_ns():x}")
        self.save(loom)
        return loom

    def save(self, loom: Loom) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(loom.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(loom.to_dict()))
        tmp.replace(path)

    def load(self, loom_id: str) -> Loom:
        path = self._path(loom_id)
        if not path.exists():
            raise LoomError(f"no loom {loom_id!r}")
        return Loom.from_dict(json.loads(path.read_text()))

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append(
                {
                    "id": data.get("id"),
                    "title": data.get("title"),
                    "updated": data.get("updated", 0),
                    "turns": len(data.get("nodes", {})),
                }
            )
        return sorted(out, key=lambda row: row["updated"], reverse=True)

    def delete(self, loom_id: str) -> None:
        self._path(loom_id).unlink(missing_ok=True)
