"""Write in someone's voice: a style profile from writing samples, a distance to it, and a rewrite that keeps facts.

    from deslop.voice import profile, delta, attribute, blend, rewrite_messages, fact_check
    twain = profile(twain_passages, name="twain")         # from plain-text samples of one writer
    delta(text, twain)                                     # Burrows' Delta: lower is closer to the voice
    attribute(text, [twain, darwin, lincoln])              # nearest voice and every distance
    mix = blend([twain, darwin], [0.5, 0.5])               # a voice between two writers
    msgs = rewrite_messages(text, twain, samples)          # chat messages for any model
    fact_check(text, rewritten)                            # numbers, names and negations kept?

A profile records what stylometry has long used to tell writers apart: the rates of the most frequent function words
("the", "of", "which", "upon"), sentence and paragraph lengths, punctuation per 1000 words, word length, vocabulary
richness and favourite sentence openers. Burrows' Delta (Burrows 2002) compares function-word rates: each rate becomes
a z-score against a fixed reference corpus (`data/function_words.json`, 318 passages of human prose), and Delta is the
mean absolute difference of z-scores. Function words are hard to imitate on purpose, which makes them a fair test of
whether a rewrite took on a voice or only its surface.

Use it on your own writing or on public-domain authors. Do not imitate a living writer without their consent, and say
when a text was written with AI assistance where that is expected (school, journals, many workplaces).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from importlib import resources
from statistics import mean, pstdev
from typing import Any

from deslop.habits import WORD, count_words, sentences

PUNCTUATION = {
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "dash": "—",
    "exclamation": "!",
    "question": "?",
    "parenthesis": "(",
    "quote": '"',
}
NEGATIONS = {"not", "no", "never", "nor", "none", "nothing", "neither", "nobody", "nowhere", "without", "cannot"}
NUMBER = re.compile(
    r"\b\d[\d,.]*\b|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
    r"hundred|thousand|million|billion|dozen)\b",
    re.I,
)
PRONOUN_I = {"I", "I'm", "I've", "I'd", "I'll"}

# Candidate function words: articles, pronouns, prepositions, conjunctions, auxiliaries, common adverbs and
# determiners (after Mosteller and Wallace's and Burrows' lists). build_reference keeps the most frequent of them.
_POOL = """
a about above after again against all almost also although always am among an and another any are around as at
be because been before being below between both but by can could did do does down during each either enough even
ever every few for from had has have he her here hers herself him himself his how however i if in indeed into is
it its itself just least less many may me might mine more most much must my myself neither never no nor not now
of off often on once one only or other others our ours ourselves out over own perhaps quite rather same shall
she should since so some such than that the their theirs them themselves then there therefore these they this
those though through thus till to too toward towards under unless until up upon us very was we well were what
whatever when where whether which while who whom whose why will with within without would yet you your yours
"""
FUNCTION_WORD_POOL = frozenset(_POOL.split())


@cache
def reference() -> dict[str, Any]:
    """Function words and their per-passage mean and standard deviation (per 1000 words) in the reference corpus."""
    raw = resources.files("deslop").joinpath("data/function_words.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)
    return data


def function_words() -> list[str]:
    return list(reference()["words"])


def tokenize(text: str) -> list[str]:
    """Lower-case words; "don't" becomes "do", "not", and "it's" becomes "it"."""
    out: list[str] = []
    for w in WORD.findall(text.replace("’", "'")):
        lw = w.lower()
        if lw.endswith("n't"):
            out += ["can" if lw == "can't" else "will" if lw == "won't" else lw[:-3], "not"]
        elif "'" in lw:
            out.append(lw.split("'", 1)[0])
        else:
            out.append(lw)
    return out


def word_rates(text: str, words: Sequence[str] | None = None) -> dict[str, float]:
    """Each function word's rate per 1000 words in `text`."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    n = max(len(tokens), 1)
    return {w: 1000.0 * counts.get(w, 0) / n for w in (words or function_words())}


def _spread(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "p10": 0.0, "p90": 0.0}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])

    return {"mean": mean(values), "sd": pstdev(values), "p10": pick(0.1), "p90": pick(0.9)}


def _ttr(tokens: Sequence[str], window: int = 100) -> float:
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    return mean(len(set(tokens[i : i + window])) / window for i in range(0, len(tokens) - window + 1, window // 2))


def profile(texts: Iterable[str], name: str = "voice") -> dict[str, Any]:
    """A voice profile from writing samples (plain text, ideally 3,000 words or more in all)."""
    chosen = [t for t in texts if t.strip()]
    joined = "\n\n".join(chosen)
    words = count_words(joined)
    sents = [s for t in chosen for _, _, s in sentences(t)]
    paragraphs = [p for t in chosen for p in re.split(r"\n\s*\n", t) if p.strip()]
    tokens = tokenize(joined)
    openers = Counter(tokenize(s)[0] for s in sents if count_words(s) > 3).most_common(12)
    return {
        "name": name,
        "words": words,
        "function_words": word_rates(joined),
        "sentence_length": _spread([count_words(s) for s in sents if count_words(s) > 0]),
        "paragraph_length": _spread([count_words(p) for p in paragraphs]),
        "punctuation_per_1k": {k: 1000.0 * joined.count(v) / max(words, 1) for k, v in PUNCTUATION.items()},
        "word_length": mean(len(w) for w in tokens) if tokens else 0.0,
        "type_token_100": _ttr(tokens),
        "openers": [[w, c] for w, c in openers],
    }


def blend(
    profiles: Sequence[Mapping[str, Any]], weights: Sequence[float] | None = None, name: str = "blend"
) -> dict[str, Any]:
    """A weighted average of profiles: a voice between several writers (weights default to equal)."""
    raw = list(weights) if weights is not None else [1.0] * len(profiles)
    w = [x / sum(raw) for x in raw]

    def avg(*path: str) -> float:
        """The weighted mean of a field, given by its keys (missing values count as 0)."""
        total = 0.0
        for wi, p in zip(w, profiles, strict=True):
            value: Any = p
            for key in path:
                value = value.get(key, 0.0) if isinstance(value, Mapping) else 0.0
            total += wi * float(value)
        return total

    spread_keys = ("mean", "sd", "p10", "p90")
    return {
        "name": name,
        "words": sum(p["words"] for p in profiles),
        "function_words": {x: avg("function_words", x) for x in function_words()},
        "sentence_length": {k: avg("sentence_length", k) for k in spread_keys},
        "paragraph_length": {k: avg("paragraph_length", k) for k in spread_keys},
        "punctuation_per_1k": {k: avg("punctuation_per_1k", k) for k in PUNCTUATION},
        "word_length": avg("word_length"),
        "type_token_100": avg("type_token_100"),
        "openers": sorted((o for p in profiles for o in p["openers"]), key=lambda o: -o[1])[:12],
        "blend_of": [[p["name"], wi] for p, wi in zip(profiles, w, strict=True)],
    }


def zscores(rates: Mapping[str, float]) -> dict[str, float]:
    ref = reference()
    return {w: (rates.get(w, 0.0) - m) / s for w, m, s in zip(ref["words"], ref["mean"], ref["sd"], strict=True)}


def delta(text_or_rates: str | Mapping[str, float], voice: Mapping[str, Any]) -> float:
    """Burrows' Delta between a text (or its function-word rates) and a voice profile. Lower is closer."""
    rates = word_rates(text_or_rates) if isinstance(text_or_rates, str) else text_or_rates
    a, b = zscores(rates), zscores(voice["function_words"])
    return mean(abs(a[w] - b[w]) for w in a)


def attribute(text: str, voices: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, float]]:
    """The nearest voice by Burrows' Delta, and the distance to each."""
    rates = word_rates(text)
    distances = {str(v["name"]): delta(rates, v) for v in voices}
    return min(distances, key=lambda k: distances[k]), distances


def style_notes(voice: Mapping[str, Any]) -> str:
    """The profile in plain words, for a prompt."""
    s, p = voice["sentence_length"], voice["punctuation_per_1k"]
    openers = ", ".join(f'"{w}"' for w, _ in voice["openers"][:6])
    marks = ", ".join(f"{k} {v:.0f}" for k, v in sorted(p.items(), key=lambda kv: -kv[1]) if v >= 0.5)
    return (
        f"Sentences average {s['mean']:.0f} words (most between {s['p10']:.0f} and {s['p90']:.0f}); paragraphs "
        f"average {voice['paragraph_length']['mean']:.0f} words. Punctuation per 1000 words: {marks}. Sentences often "
        f"begin with {openers}."
    )


def rewrite_messages(text: str, voice: Mapping[str, Any], samples: Sequence[str]) -> list[dict[str, str]]:
    """Chat messages asking a model to rewrite `text` in the voice of `samples`, keeping every fact."""
    shown = "\n\n---\n\n".join(samples)
    system = (
        "You rewrite text in the voice of a particular writer, using samples of their writing. Match their sentence "
        "length and rhythm, their small words (articles, prepositions, conjunctions, pronouns), their punctuation and "
        "the way they open sentences. Keep every fact of the original: every number, name and claim, and keep "
        "negations negative. Add no new facts, examples, studies or quotations. Do not copy sentences from the "
        "samples. Write prose only, with no headings, lists or emoji. Reply with the rewritten text alone."
    )
    user = (
        f"Samples of the writer:\n\n{shown}\n\n---\n\nStyle notes: {style_notes(voice)}\n\n"
        f"Rewrite this text in that writer's voice, about the same length:\n\n{text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _names(text: str) -> set[str]:
    """Capitalised words that do not start a sentence: a rough list of names and proper nouns."""
    out: set[str] = set()
    for _, _, sent in sentences(text):
        words = WORD.findall(sent)
        out |= {w for w in words[1:] if w[:1].isupper() and w not in PRONOUN_I}
    return out


def fact_check(original: str, rewritten: str) -> dict[str, Any]:
    """Which numbers, names and negations of `original` survive in `rewritten`."""
    nums = {m.group(0).lower().rstrip(".,") for m in NUMBER.finditer(original)}
    low = rewritten.lower()
    kept_nums = {n for n in nums if n in low}
    names = _names(original)
    kept_names = {n for n in names if n.lower() in low}
    return {
        "numbers": len(nums),
        "numbers_kept": len(kept_nums),
        "numbers_missing": sorted(nums - kept_nums),
        "names": len(names),
        "names_kept": len(kept_names),
        "names_missing": sorted(names - kept_names),
        "negations": sum(t in NEGATIONS for t in tokenize(original)),
        "negations_rewrite": sum(t in NEGATIONS for t in tokenize(rewritten)),
        "all_kept": kept_nums == nums and len(kept_names) >= 0.8 * len(names),
    }


def build_reference(texts: Iterable[str], top: int = 150) -> dict[str, Any]:
    """Reference statistics (`data/function_words.json`) from a corpus: the `top` most frequent function words, with
    each one's mean and standard deviation per 1000 words across the passages."""
    chosen = list(texts)
    counts = Counter(t for text in chosen for t in tokenize(text) if t in FUNCTION_WORD_POOL)
    words = [w for w, _ in counts.most_common(top)]
    per = [word_rates(t, words) for t in chosen]
    means = [mean(p[w] for p in per) for w in words]
    sds = [max(pstdev([p[w] for p in per]), 1e-6) for w in words]
    return {"words": words, "mean": means, "sd": sds, "passages": len(chosen)}
