"""Word search over unit labels and reads: an inverted index with all-words-first ranking and typo tolerance.

    index = SearchIndex.build(documents)        # (address, title, text, score) from a catalog or a computed atlas
    index.search("closing brackets")            # units matching every word first, then most words, then score

Matching, modelled on neuron-lens.com's search (its behaviour, rebuilt here): each query word counts; a unit holding
every word ranks first, and a title holding the whole phrase rises; a quoted phrase must be in the title. A word that
matches nothing is matched within a typo or two (listed under `approx`, shown as ≈). An address (`L8 n344+`, `L20h17`)
or a layer (`L8`) is answered by the room before the index is asked.

Words are lowercased, split on anything that is not a letter or digit (CJK characters count one by one), and folded
to a crude stem (a trailing "s", "es", "ing", "ed" dropped) so "brackets" finds "bracket".
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from steerlab.atlas.address import Address

_WORD = re.compile(r"[0-9a-zA-ZÀ-ɏͰ-ϿЀ-ӿ]+|[぀-ヿ㐀-鿿가-힯]")
_STOP = frozenset({"a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "with", "by", "at", "as", "its", "it"})


def stem(word: str) -> str:
    word = word.lower()
    for suffix in ("ing", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def words(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text or "") if w.lower() not in _STOP]


@dataclass
class SearchIndex:
    addresses: list[Address] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    postings: dict[str, set[int]] = field(default_factory=dict)
    title_words: list[frozenset[str]] = field(default_factory=list)

    @classmethod
    def build(cls, documents: Iterable[tuple[Address, str, str, float]]) -> SearchIndex:
        index = cls()
        for address, title, text, score in documents:
            i = len(index.addresses)
            index.addresses.append(address)
            index.titles.append(title)
            index.scores.append(float(score))
            index.title_words.append(frozenset(words(title)))
            for word in set(words(f"{title} {text}")):
                index.postings.setdefault(word, set()).add(i)
        return index

    def __len__(self) -> int:
        return len(self.addresses)

    def _close(self, term: str, cutoff: float) -> list[str]:
        # Only words that share the first letter: keeps difflib's scan small on a 100k-word vocabulary.
        vocab = [w for w in self.postings if w[:1] == term[:1] and abs(len(w) - len(term)) <= 2]
        return difflib.get_close_matches(term, vocab, n=3, cutoff=cutoff)

    def search(self, query: str, limit: int = 30, kind: str | None = None) -> dict[str, Any]:
        """Ranked matches: `{"results": [{address, title, score, hits, all_words}], "total", "approx", "suggest"}`."""
        phrases = [p.lower() for p in re.findall(r'"([^"]+)"', query)]
        bare = re.sub(r'"[^"]*"', " ", query)
        terms = list(dict.fromkeys(words(bare) + [w for p in phrases for w in words(p)]))
        if not terms:
            return {"results": [], "total": 0, "terms": [], "approx": {}, "suggest": []}
        hits: dict[int, int] = {}
        approx: dict[str, list[str]] = {}
        for term in terms:
            docs = self.postings.get(term)
            if docs is None:
                close = self._close(term, 0.8)
                if close:
                    approx[term] = close
                docs = set().union(*(self.postings[c] for c in close)) if close else set()
            for i in docs:
                hits[i] = hits.get(i, 0) + 1
        whole = " ".join(w.lower() for w in _WORD.findall(bare))
        ranked = []
        for i, n in hits.items():
            address = self.addresses[i]
            if kind and address.kind != kind:
                continue
            title = self.titles[i].lower()
            if phrases and not all(p in title for p in phrases):
                continue
            phrase_bonus = 1 if whole and whole in title else 0
            in_title = sum(1 for t in terms if t in self.title_words[i])
            ranked.append((-(n == len(terms)), -n, -phrase_bonus, -in_title, -self.scores[i], i))
        ranked.sort()
        results = []
        for key in ranked[:limit]:
            i = key[-1]
            address = self.addresses[i]
            results.append(
                {
                    "address": address.path(),
                    "display": str(address),
                    "kind": address.kind,
                    "layer": address.layer,
                    "title": self.titles[i],
                    "score": round(self.scores[i], 4),
                    "hits": -key[1],
                    "all_words": key[0] == -1,
                }
            )
        suggest: list[str] = []
        if not results:
            for term in terms:
                suggest += self._close(term, 0.6)
        return {"results": results, "total": len(ranked), "terms": terms, "approx": approx, "suggest": suggest[:6]}
