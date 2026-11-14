"""A tiny linear stand-in for a model, with known answers, for tests and a no-GPU demo.

Each position's logits are the sum of its active features' output weights times their values, so it is obvious
which features cause the output. `demo()` builds one with a switch, two readouts, a redundant backup pair and a
crowd of small noise features to draw controls from.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from circuitkit.validate.backend import ActiveFeature
from circuitkit.validate.types import FeatureNode, Intervention

Unit = tuple[int, int]


@dataclass
class ToyPrompt:
    tokens: list[str]
    # one {(layer, feature): activation} dict per position
    acts: list[dict[Unit, float]]


@dataclass
class ToyBackend:
    vocab: list[str]
    out: dict[Unit, np.ndarray]
    prompts: dict[str, ToyPrompt]
    # pairs of units that act through max(a, b): either one alone is enough
    backups: list[tuple[Unit, Unit]] = field(default_factory=list)
    norms: dict[Unit, float] = field(default_factory=dict)
    calls: int = 0

    def tokens(self, prompt: str) -> list[str]:
        return list(self.prompts[prompt].tokens)

    def token_id(self, token: str) -> int:
        return self.vocab.index(token)

    def describe(self) -> dict[str, str]:
        return {"model": "toy linear model", "transcoders": "toy", "device": "cpu"}

    def decoder_norm(self, layer: int, feature: int) -> float:
        return self.norms.get((layer, feature), float(np.linalg.norm(self.out[(layer, feature)])) or 1.0)

    def active_features(self, prompt: str) -> list[ActiveFeature]:
        p = self.prompts[prompt]
        return [(u[0], pos, u[1], a) for pos, d in enumerate(p.acts) for u, a in d.items() if a != 0.0]

    def logits(self, prompt: str, interventions: Sequence[Intervention] = ()) -> np.ndarray:
        self.calls += 1
        p = self.prompts[prompt]
        acts = [dict(d) for d in p.acts]
        for layer, pos, feat, value in interventions:
            acts[pos][(layer, feat)] = float(value)
        in_backup = {u for pair in self.backups for u in pair}
        rows = []
        for d in acts:
            row = np.zeros(len(self.vocab))
            for u, a in d.items():
                if u not in in_backup and u in self.out:
                    row += a * self.out[u]
            for b1, b2 in self.backups:
                row += max(d.get(b1, 0.0), d.get(b2, 0.0)) * self.out[b1]
            rows.append(row)
        return np.stack(rows)


SWITCH: Unit = (5, 100)
READOUTS: tuple[Unit, Unit] = ((7, 200), (8, 201))
BACKUP: tuple[Unit, Unit] = ((4, 300), (6, 301))


def demo(n_noise: int = 40, seed: int = 0) -> ToyBackend:
    """A toy 'capital of the state containing Dallas' model and its 'Oakland' counterfactual.

    Ground truth: SWITCH is a switch, READOUTS are readouts (big activations, no output weight), and each BACKUP
    feature is a backup (either one alone keeps the answer).
    """
    vocab = ["<bos>", "The", "capital", "of", "Dallas", "Oakland", "is", " Austin", " Sacramento", " the"]
    a, s = vocab.index(" Austin"), vocab.index(" Sacramento")
    rng = random.Random(seed)

    def onehot(i: int, w: float) -> np.ndarray:
        v = np.zeros(len(vocab))
        v[i] = w
        return v

    out: dict[Unit, np.ndarray] = {
        SWITCH: onehot(a, 2.0),
        READOUTS[0]: np.zeros(len(vocab)),
        READOUTS[1]: np.zeros(len(vocab)),
        BACKUP[0]: onehot(a, 1.0),
        BACKUP[1]: onehot(a, 1.0),
        (5, 101): onehot(s, 2.0),  # the counterfactual's own switch
    }
    norms = {READOUTS[0]: 1.0, READOUTS[1]: 1.0}
    noise: list[Unit] = []
    for k in range(n_noise):
        u = (rng.randrange(3, 10), 1000 + k)
        noise.append(u)
        out[u] = np.array([rng.gauss(0, 0.02) for _ in vocab])
        norms[u] = rng.uniform(0.5, 2.0)

    def prompt(city: str, switch: Unit) -> ToyPrompt:
        toks = ["<bos>", "The", "capital", "of", city, "is"]
        acts: list[dict[Unit, float]] = [{} for _ in toks]
        for pos in range(1, len(toks)):
            for u in rng.sample(noise, 8):
                acts[pos][u] = rng.uniform(0.5, 3.0)
        last = len(toks) - 1
        acts[last][switch] = 2.0
        acts[last][READOUTS[0]] = 4.0
        acts[last][READOUTS[1]] = 3.5
        if switch == SWITCH:
            acts[last][BACKUP[0]] = 1.5
            acts[last][BACKUP[1]] = 1.5
        return ToyPrompt(tokens=toks, acts=acts)

    return ToyBackend(
        vocab=vocab,
        out=out,
        prompts={"dallas": prompt("Dallas", SWITCH), "oakland": prompt("Oakland", (5, 101))},
        backups=[BACKUP],
        norms=norms,
    )


def demo_nodes(backend: ToyBackend, prompt: str = "dallas") -> list[FeatureNode]:
    """The features a graph would put on top for the toy prompt: the switch, the readouts and the backups."""
    last = len(backend.tokens(prompt)) - 1
    acts: Mapping[Unit, float] = backend.prompts[prompt].acts[last]
    units = [SWITCH, *READOUTS, *BACKUP]
    return [FeatureNode(layer=u[0], pos=last, feature=u[1], activation=acts[u]) for u in units]
