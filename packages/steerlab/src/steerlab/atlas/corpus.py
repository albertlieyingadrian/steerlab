"""The small activation corpus for computed atlases: web-like prose, Python, and chat turns.

The catalog's corpus is 15.4M positions; ours is a few tens of thousands, enough to find what a unit fires on hardest
but not to estimate rare-event rates finely. Every computed card names its corpus and token count.

- prose: wikitext-2's test split (Salesforce/wikitext, CC BY-SA), fetched once through the Hub cache;
- code: steerlab's own Python sources (shipped with the package, so always available offline);
- chat: short user/assistant turns rendered with the model's chat template when it has one.

Without network or wikitext in the cache the corpus is code and chat alone, and says so in `describe`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")

CHAT_TURNS: tuple[tuple[str, str], ...] = (
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Can you help me write a short poem about the sea?", "Waves fold over grey stones, salt wind in the reeds."),
    ("How do I reverse a list in Python?", "Use my_list.reverse() in place, or my_list[::-1] for a reversed copy."),
    ("Translate 'good morning' into Spanish.", "'Good morning' in Spanish is 'buenos días'."),
    ("Why is the sky blue?", "Sunlight scatters off air molecules, and blue light scatters the most."),
    ("Give me three tips for sleeping better.", "Keep a regular schedule, avoid screens late, and keep the room cool."),
    ("What's 17 times 23?", "17 times 23 is 391."),
    ("Summarise the plot of Romeo and Juliet in one sentence.", "Two young lovers from feuding families die."),
    ("Is it safe to eat raw cookie dough?", "Raw flour and raw eggs can carry bacteria, so it is safer baked."),
    ("Recommend a good book about history.", "Try 'The Guns of August' by Barbara Tuchman."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    (
        "Write a haiku about autumn.",
        "Red leaves drift and fall / the river carries them south / cold light on the stones",
    ),
)


@dataclass
class Corpus:
    docs: list[list[int]]
    slices: list[str]
    sources: dict[str, int] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return sum(len(d) for d in self.docs)

    def describe(self) -> str:
        parts = ", ".join(f"{n} {name}" for name, n in self.sources.items())
        return f"a small computed corpus: {len(self.docs)} documents, {self.n_tokens:,} tokens ({parts})"

    def to_json(self) -> dict[str, Any]:
        return {"docs": self.docs, "slices": self.slices, "sources": self.sources}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Corpus:
        return cls([list(map(int, d)) for d in data["docs"]], list(data["slices"]), dict(data.get("sources", {})))


def _wikitext(cache_dir: Path | None) -> list[str]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        kwargs: dict[str, Any] = {"repo_type": "dataset", "cache_dir": None if cache_dir is None else cache_dir / "hf"}
        try:
            path = hf_hub_download(*WIKITEXT, local_files_only=True, **kwargs)
        except Exception:  # not cached yet
            path = hf_hub_download(*WIKITEXT, **kwargs)
        lines = pq.read_table(path, columns=["text"])["text"].to_pylist()
    except Exception as error:  # offline and not cached: no prose slice, and say why
        warnings.warn(f"atlas corpus: no wikitext prose slice ({type(error).__name__}: {error})", stacklevel=2)
        return []
    # Join article paragraphs into documents of a few hundred words; skip headings and blank lines.
    docs: list[str] = []
    current: list[str] = []
    for line in lines:
        text = (line or "").strip()
        if not text or text.startswith("="):
            if sum(len(t) for t in current) > 600:
                docs.append(" ".join(current))
                current = []
            continue
        current.append(text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", "."))
    if current:
        docs.append(" ".join(current))
    return docs


def _code() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted(p for p in root.rglob("*.py") if "tests" not in p.parts)
    return [p.read_text(encoding="utf-8", errors="ignore") for p in files]


def _chat(tokenizer: Any) -> list[str]:
    out = []
    for user, assistant in CHAT_TURNS:
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        try:
            out.append(str(tokenizer.apply_chat_template(messages, tokenize=False)))
        except Exception:  # no chat template (a base model or a tiny test tokenizer)
            out.append(f"User: {user}\nAssistant: {assistant}\n")
    return out


def build_corpus(
    tokenizer: Any,
    n_tokens: int = 24_000,
    max_len: int = 128,
    cache_dir: Path | None = None,
    texts: dict[str, list[str]] | None = None,
) -> Corpus:
    """Tokenize the slices into documents of at most `max_len` tokens until about `n_tokens` are taken.

    Budget shares: prose 60%, code 25%, chat 15% (chat is small and takes what it has). `texts` replaces the default
    slices (tests pass a few strings; nothing is downloaded then)."""
    slices = texts if texts is not None else {"web": _wikitext(cache_dir), "code": _code(), "chat": _chat(tokenizer)}
    chunks: dict[str, list[list[int]]] = {}
    for name, items in slices.items():
        pieces: list[list[int]] = []
        for text in items:
            ids = tokenizer(text, add_special_tokens=False).input_ids
            for start in range(0, len(ids), max_len):
                piece = ids[start : start + max_len]
                if len(piece) >= 8:
                    pieces.append(list(map(int, piece)))
        if pieces:
            chunks[name] = pieces
    shares = {"web": 0.6, "code": 0.25, "chat": 0.15}
    total_share = sum(shares.get(name, 1 / len(chunks)) for name in chunks) or 1.0
    docs: list[list[int]] = []
    names: list[str] = []
    sources: dict[str, int] = {}
    for name, pool in chunks.items():
        budget = int(n_tokens * shares.get(name, 1 / len(chunks)) / total_share)
        step = max(1, len(pool) // max(1, budget // max_len + 1))
        taken = 0
        for doc in pool[::step]:
            if taken >= budget:
                break
            docs.append(doc)
            names.append(name)
            taken += len(doc)
        sources[name] = names.count(name)
    return Corpus(docs, names, sources)
