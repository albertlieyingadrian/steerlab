"""Contrast prompt sets for the difference-of-means directions, shipped inside the package.

    concept = load_concept("chinese")           # pairs, graded levels, held-out prompts
    texts, responses, labels, groups = concept.as_rows()

One JSON file per concept lives in `concepts/`. Each pair is two (prompt, response) examples: the positive one shows
the concept (a reply in Chinese, a refusal, a deliberately wrong answer under pressure), the negative one does not.
The two measured concepts ("chinese" and "sandbagging") also carry five graded prompt sets, asking for
0/25/50/75/100 % of the concept, which the along-manifold source turns into per-level centroids.

Light module: json and dataclasses only, so `steerlab directions list` and the lesson text can read it without torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

CONCEPTS: tuple[str, ...] = ("sandbagging", "refusal", "chinese")
"""The three built-in directions, in a fixed order."""

MEASURED_CONCEPTS: tuple[str, ...] = ("chinese", "sandbagging")
"""The concepts the report measures, which therefore carry graded levels."""

GRADED_LEVELS: tuple[int, ...] = (0, 25, 50, 75, 100)
"""Percent of the concept each graded prompt set asks for."""


@dataclass(frozen=True)
class Example:
    """One user prompt and the assistant reply whose tokens are averaged (`response_mean`)."""

    prompt: str
    response: str


@dataclass(frozen=True)
class ContrastPair:
    positive: Example
    negative: Example


@dataclass(frozen=True)
class Concept:
    name: str
    label: str
    description: str
    pairs: tuple[ContrastPair, ...]
    levels: dict[int, tuple[str, ...]] = field(default_factory=dict)
    held_out: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_rows(self) -> tuple[list[str], list[str], list[int], list[int]]:
        """(texts, responses, labels, groups) in sandbagkit's row layout: positive then negative for each pair.

        `labels` is 1 for the positive example and 0 for the negative; `groups` is the pair index, which
        `steering_vector`'s leakage guard needs.
        """
        texts: list[str] = []
        responses: list[str] = []
        labels: list[int] = []
        groups: list[int] = []
        for index, pair in enumerate(self.pairs):
            for label, example in ((1, pair.positive), (0, pair.negative)):
                texts.append(example.prompt)
                responses.append(example.response)
                labels.append(label)
                groups.append(index)
        return texts, responses, labels, groups


def _folder() -> Any:
    return resources.files(__package__).joinpath("concepts")


def list_concepts() -> list[str]:
    """Names of the shipped concepts, sorted."""
    return sorted(path.name.removesuffix(".json") for path in _folder().iterdir() if path.name.endswith(".json"))


def load_concept(name: str) -> Concept:
    """Read `concepts/<name>.json`. Raises KeyError naming the available concepts when there is no such file."""
    path = _folder().joinpath(f"{name}.json")
    if not path.is_file():
        raise KeyError(f"no concept named {name!r}; the shipped concepts are {', '.join(list_concepts())}")
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    pairs = tuple(
        ContrastPair(
            positive=Example(pair["positive"]["prompt"], pair["positive"]["response"]),
            negative=Example(pair["negative"]["prompt"], pair["negative"]["response"]),
        )
        for pair in raw["pairs"]
    )
    levels = {int(level): tuple(prompts) for level, prompts in raw.get("levels", {}).items()}
    return Concept(
        name=raw["name"],
        label=raw.get("label", raw["name"]),
        description=raw.get("description", ""),
        pairs=pairs,
        levels=dict(sorted(levels.items())),
        held_out=tuple(raw.get("held_out", ())),
        metadata=dict(raw.get("metadata", {})),
    )


__all__ = [
    "CONCEPTS",
    "GRADED_LEVELS",
    "MEASURED_CONCEPTS",
    "Concept",
    "ContrastPair",
    "Example",
    "list_concepts",
    "load_concept",
]
