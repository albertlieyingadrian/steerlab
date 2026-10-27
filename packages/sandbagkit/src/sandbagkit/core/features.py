"""Turn a model's text output into numbers detectors can compare.

    "I think this might be a test."  ──►  TextFeatures(word_count=7, hedging_count=2, eval_awareness_count=1, ...)

Counts are raw phrase matches from `sandbagkit.core.text`. Ratios divide by the word count, so long and short
answers are comparable. Empty text gives all zeros, never an error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields

import numpy as np
import numpy.typing as npt

from sandbagkit.core import text as _text


def hedging_count(text: str) -> int:
    """Phrases that soften a claim: "I think", "might", "perhaps", "I'm not sure"."""
    return _text.count_phrases(text, _text.HEDGING)


def eval_awareness_count(text: str) -> int:
    """Phrases where the model says it is being tested: "this is a test", "being evaluated"."""
    return _text.count_phrases(text, _text.EVAL_AWARENESS)


def meta_commentary_count(text: str) -> int:
    """Talk about the situation rather than the task. Includes every eval-awareness phrase."""
    return _text.count_phrases(text, _text.META_COMMENTARY)


def self_reference_count(text: str) -> int:
    """The model talking about itself: "as an AI", "my training", "I was trained"."""
    return _text.count_phrases(text, _text.SELF_REFERENCE)


def reasoning_depth(text: str) -> int:
    """Explicit reasoning structure: numbered or bulleted steps plus logical connectives ("because", "therefore")."""
    return _text.count_phrases(text, _text.REASONING_STEPS) + _text.count_phrases(text, _text.REASONING_CONNECTIVES)


def refusal_count(text: str) -> int:
    return _text.count_phrases(text, _text.REFUSAL)


def confidence_count(text: str) -> int:
    """Phrases that assert certainty: "definitely", "clearly", "I'm confident"."""
    return _text.count_phrases(text, _text.HIGH_CONFIDENCE)


@dataclass(frozen=True)
class TextFeatures:
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    hedging_count: int = 0
    hedging_ratio: float = 0.0
    meta_commentary_count: int = 0
    eval_awareness_count: int = 0
    self_reference_count: int = 0
    self_reference_ratio: float = 0.0
    reasoning_depth: int = 0
    refusal_count: int = 0
    confidence_count: int = 0

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    def vector(self) -> npt.NDArray[np.float64]:
        """The features as a 1-D array, in FEATURE_NAMES order."""
        return np.array(list(self.as_dict().values()), dtype=float)


FEATURE_NAMES: tuple[str, ...] = tuple(field.name for field in fields(TextFeatures))


def extract_features(text: str) -> TextFeatures:
    word_list = _text.words(text)
    n_words = len(word_list)
    if n_words == 0:
        return TextFeatures(char_count=len(text.strip()))
    n_sentences = len(_text.sentences(text))
    hedges = hedging_count(text)
    self_refs = self_reference_count(text)
    return TextFeatures(
        char_count=len(text.strip()),
        word_count=n_words,
        sentence_count=n_sentences,
        avg_sentence_length=n_words / n_sentences if n_sentences else 0.0,
        hedging_count=hedges,
        hedging_ratio=hedges / n_words,
        meta_commentary_count=meta_commentary_count(text),
        eval_awareness_count=eval_awareness_count(text),
        self_reference_count=self_refs,
        self_reference_ratio=self_refs / n_words,
        reasoning_depth=reasoning_depth(text),
        refusal_count=refusal_count(text),
        confidence_count=confidence_count(text),
    )


def feature_matrix(texts: Iterable[str]) -> npt.NDArray[np.float64]:
    """One row per text, one column per feature (FEATURE_NAMES order)."""
    rows = [extract_features(t).vector() for t in texts]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)))
    return np.vstack(rows)
