"""Cheap automatic labels for transcoder features, and scores that say how much to trust them.

The pipeline, per feature:

1. **Examples.** Activation records (`Record`: tokens and the feature's activation on each) are split into an
   explain set (the top `n_explain` records by peak activation) and a held-out set (the next activating records,
   plus records where the feature stays at zero).
2. **Explain.** A model sees the explain set, with the active tokens marked, and writes a short label and a
   confidence (0-100).
3. **Score** the label on the held-out set, which the explainer never saw:
   - *detection*: the model gets the label and a shuffled mix of held-out windows, half where the feature fires and
     half where it does not, and gives each a probability that the feature fires. Scores: AUROC, balanced accuracy
     at 0.5 and the Brier score;
   - *simulation* (optional, costs more): the model predicts the activation (0-10) of every token of each held-out
     window, and the score is the correlation with the real activations (Bills et al., 2023).
4. **Calibrate** (`calibration`): over many features, compare the explainer's confidence with the detection score
   it then earned, and the detector's probabilities with the truth (expected calibration error, reliability bins).

Every call goes through an `LLM` (see `providers`), so with `CachedLLM` a rerun is free and deterministic.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel, Field

from circuitkit.labels.providers import LLM

WINDOW = 24


class Record(BaseModel):
    """One text and the feature's activation on each of its tokens."""

    tokens: list[str]
    acts: list[float]
    source: str | None = None

    @property
    def peak(self) -> float:
        return max(self.acts) if self.acts else 0.0

    def window(self, size: int = WINDOW) -> Record:
        """The `size` tokens around the peak (or the first `size` tokens when the feature never fires)."""
        if len(self.tokens) <= size:
            return self
        c = int(np.argmax(self.acts)) if self.peak > 0 else size // 2
        lo = max(0, min(c - (2 * size) // 3, len(self.tokens) - size))
        return Record(tokens=self.tokens[lo : lo + size], acts=self.acts[lo : lo + size], source=self.source)


class FeatureData(BaseModel):
    unit: str = Field(description="'layer:feature'")
    records: list[Record]


class Split(BaseModel):
    explain: list[Record]
    positive: list[Record]
    negative: list[Record]


class DetectionScore(BaseModel):
    auroc: float | None
    balanced_accuracy: float
    brier: float
    n_positive: int
    n_negative: int
    probs: list[float]
    truth: list[int]


class SimulationScore(BaseModel):
    correlation: float | None
    n_tokens: int


class FeatureLabel(BaseModel):
    unit: str
    label: str
    confidence: float | None = Field(None, description="the explainer's own confidence, 0-1")
    model: str
    detection: DetectionScore | None = None
    simulation: SimulationScore | None = None
    n_explain: int = 0
    error: str | None = None

    @property
    def score(self) -> float | None:
        """The headline score: detection AUROC if run, else simulation correlation."""
        if self.detection is not None and self.detection.auroc is not None:
            return self.detection.auroc
        return self.simulation.correlation if self.simulation else None


# ---------------------------------------------------------------- examples


def _dedupe(records: Sequence[Record]) -> list[Record]:
    seen: set[tuple[str, ...]] = set()
    out = []
    for r in records:
        k = tuple(r.tokens)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def split_records(
    records: Sequence[Record], n_explain: int = 12, n_positive: int = 8, n_negative: int = 8, seed: int = 0
) -> Split:
    """Top `n_explain` records by peak to explain; the next `n_positive` activating and `n_negative` silent ones
    (chosen at random) to score."""
    recs = _dedupe(records)
    act = sorted([r for r in recs if r.peak > 0], key=lambda r: -r.peak)
    silent = [r for r in recs if r.peak <= 0]
    rng = random.Random(seed)
    rest = act[n_explain:]
    pos = rest[:: max(1, len(rest) // n_positive)][:n_positive] if rest else []
    rng.shuffle(silent)
    return Split(explain=act[:n_explain], positive=pos, negative=silent[:n_negative])


def _clean(tok: str) -> str:
    return tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n").replace("\n", "\\n")


def format_record(r: Record, marks: bool = True) -> str:
    """Tokens joined into text; with `marks`, each active token is wrapped as <<token>> with its activation (0-10)."""
    w = r.window()
    top = max(w.peak, 1e-9)
    out = []
    for t, a in zip(w.tokens, w.acts, strict=True):
        s = _clean(t)
        if marks and a > 0:
            out.append(f"<<{s}>>({round(10 * a / top)})")
        else:
            out.append(s)
    return "".join(out)


# ---------------------------------------------------------------- prompts

EXPLAIN_SYSTEM = (
    "You label features of a language model. A feature is a direction in the model that is active on some tokens. "
    "You see text excerpts; tokens where the feature is active are wrapped as <<token>>(a), where a is its strength "
    "from 1 to 10. Write the shortest label (at most 12 words) that predicts where the feature fires: the kind of "
    'token and the context. Answer with JSON only: {"label": "...", "confidence": <0-100>}, where confidence '
    "is how well you expect the label to predict activity on new text."
)

DETECT_SYSTEM = (
    "You test a label for a language-model feature. Given the label and numbered excerpts, estimate for each "
    "excerpt the probability that the feature is active on at least one of its tokens. About half the excerpts "
    'contain activity. Answer with JSON only: {"probs": [p1, p2, ...]}, one number from 0 to 1 per excerpt, in order.'
)

SIMULATE_SYSTEM = (
    "You simulate a language-model feature from its label. Given the label and a numbered list of tokens, predict "
    "the feature's activation on each token as an integer from 0 (inactive) to 10 (strongest). Most tokens are 0. "
    'Answer with JSON only: {"acts": [a0, a1, ...]}, one integer per token, in order.'
)


def _json(text: str) -> dict[str, object]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError("reply is not a JSON object")
    return out


def explain(llm: LLM, examples: Sequence[Record]) -> tuple[str, float | None]:
    body = "\n".join(f"{i + 1}. {format_record(r)}" for i, r in enumerate(examples))
    reply = _json(llm.complete(EXPLAIN_SYSTEM, f"Excerpts:\n{body}", max_tokens=200))
    label = str(reply.get("label", "")).strip()
    conf = reply.get("confidence")
    return label, (float(conf) / 100 if isinstance(conf, int | float) else None)


def detect(
    llm: LLM, label: str, positive: Sequence[Record], negative: Sequence[Record], seed: int = 0
) -> DetectionScore:
    items = [(r, 1) for r in positive] + [(r, 0) for r in negative]
    random.Random(seed).shuffle(items)
    body = "\n".join(f"{i + 1}. {format_record(r, marks=False)}" for i, (r, _) in enumerate(items))
    prompt = f"Label: {label}\n\nThere are exactly {len(items)} excerpts.\n\nExcerpts:\n{body}"
    probs: list[float] = []
    for attempt in range(2):  # one retry: small models sometimes miscount or return nothing
        ask = prompt if attempt == 0 else prompt + f"\n\nReturn exactly {len(items)} numbers."
        try:
            raw = _json(llm.complete(DETECT_SYSTEM, ask, max_tokens=400)).get("probs", [])
        except ValueError:
            continue
        probs = [float(p) for p in raw] if isinstance(raw, list) else []
        if len(probs) == len(items):
            break
    if len(probs) != len(items):
        raise ValueError(f"detector returned {len(probs)} probabilities for {len(items)} excerpts")
    probs = [min(1.0, max(0.0, p)) for p in probs]
    truth = [y for _, y in items]
    return DetectionScore(
        auroc=auroc(probs, truth),
        balanced_accuracy=balanced_accuracy(probs, truth),
        brier=float(np.mean([(p - y) ** 2 for p, y in zip(probs, truth, strict=True)])),
        n_positive=sum(truth),
        n_negative=len(truth) - sum(truth),
        probs=probs,
        truth=truth,
    )


def simulate(llm: LLM, label: str, records: Sequence[Record]) -> SimulationScore:
    pred: list[float] = []
    real: list[float] = []
    for r in records:
        w = r.window()
        body = "\n".join(f"{i}: {_clean(t)!r}" for i, t in enumerate(w.tokens))
        reply = _json(llm.complete(SIMULATE_SYSTEM, f"Label: {label}\n\nTokens:\n{body}", max_tokens=300))
        raw = reply.get("acts", [])
        acts = [float(a) for a in raw] if isinstance(raw, list) else []
        if len(acts) != len(w.tokens):
            acts = (acts + [0.0] * len(w.tokens))[: len(w.tokens)]
        pred += acts
        real += list(w.acts)
    return SimulationScore(correlation=correlation(pred, real), n_tokens=len(real))


# ---------------------------------------------------------------- metrics


def auroc(scores: Sequence[float], truth: Sequence[int]) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half); None without both."""
    pos = [s for s, y in zip(scores, truth, strict=True) if y]
    neg = [s for s, y in zip(scores, truth, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def balanced_accuracy(probs: Sequence[float], truth: Sequence[int], threshold: float = 0.5) -> float:
    pos = [p >= threshold for p, y in zip(probs, truth, strict=True) if y]
    neg = [p < threshold for p, y in zip(probs, truth, strict=True) if not y]
    parts = [float(np.mean(x)) for x in (pos, neg) if x]
    return float(np.mean(parts)) if parts else 0.0


def correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class Calibration(BaseModel):
    """How well stated confidence and detector probabilities match what happened."""

    n_features: int
    confidence_vs_score: float | None = Field(description="correlation of explainer confidence with detection AUROC")
    mean_confidence: float | None
    mean_score: float | None
    detector_ece: float | None = Field(description="expected calibration error of the detection probabilities")
    bins: list[dict[str, float]] = Field(default_factory=list, description="reliability bins: mean p, share true, n")


def calibration(labels: Sequence[FeatureLabel], n_bins: int = 5) -> Calibration:
    scored = [lab for lab in labels if lab.score is not None]
    conf = [lab.confidence for lab in scored if lab.confidence is not None]
    sc = [lab.score for lab in scored if lab.confidence is not None]
    probs: list[float] = []
    truth: list[int] = []
    for lab in labels:
        if lab.detection:
            probs += lab.detection.probs
            truth += lab.detection.truth
    bins: list[dict[str, float]] = []
    ece = None
    if probs:
        p, t = np.asarray(probs), np.asarray(truth)
        edges = np.linspace(0, 1, n_bins + 1)
        ece_sum = 0.0
        for i in range(n_bins):
            m = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= 1))
            if m.any():
                bins.append({"p": float(p[m].mean()), "true": float(t[m].mean()), "n": float(m.sum())})
                ece_sum += m.sum() * abs(p[m].mean() - t[m].mean())
        ece = float(ece_sum / len(p))
    return Calibration(
        n_features=len(scored),
        confidence_vs_score=correlation(conf, [float(s) for s in sc if s is not None]) if len(conf) > 2 else None,
        mean_confidence=float(np.mean(conf)) if conf else None,
        mean_score=float(np.mean([s for s in sc if s is not None])) if sc else None,
        detector_ece=ece,
        bins=bins,
    )


def isotonic(x: Sequence[float], y: Sequence[float]) -> Callable[[float], float]:
    """Monotone (pool-adjacent-violators) map from stated confidence to the score it earns, for recalibration."""
    order = np.argsort(np.asarray(x, float))
    xs = np.asarray(x, float)[order]
    blocks: list[list[float]] = [[float(v), 1.0] for v in np.asarray(y, float)[order]]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    vals = np.repeat([b[0] for b in blocks], [int(b[1]) for b in blocks])

    def f(c: float) -> float:
        return float(np.interp(c, xs, vals)) if xs.size else float("nan")

    return f


# ---------------------------------------------------------------- the pipeline


def label_feature(
    llm: LLM,
    data: FeatureData,
    *,
    score: str = "detection",
    detector: LLM | None = None,
    n_explain: int = 12,
    n_positive: int = 8,
    n_negative: int = 8,
    seed: int = 0,
    label: str | None = None,
) -> FeatureLabel:
    """Explain one feature (or take `label` as given) and score the label on held-out records."""
    sp = split_records(data.records, n_explain, n_positive, n_negative, seed)
    det = detector or llm
    try:
        conf: float | None = None
        if label is None:
            if not sp.explain:
                return FeatureLabel(unit=data.unit, label="", model=llm.model, error="no activating records")
            label, conf = explain(llm, sp.explain)
        out = FeatureLabel(unit=data.unit, label=label, confidence=conf, model=llm.model, n_explain=len(sp.explain))
        if score in ("detection", "both") and sp.positive and sp.negative:
            out.detection = detect(det, label, sp.positive, sp.negative, seed)
        if score in ("simulation", "both") and (sp.positive or sp.negative):
            out.simulation = simulate(det, label, [*sp.positive, *sp.negative])
        return out
    except Exception as e:  # one bad reply should not sink a batch; the error is kept on the record
        return FeatureLabel(unit=data.unit, label=label or "", model=llm.model, error=f"{type(e).__name__}: {e}")


def label_features(llm: LLM, features: Sequence[FeatureData], *, workers: int = 8, **kw: object) -> list[FeatureLabel]:
    """`label_feature` over many features in parallel threads; results come back in input order."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda f: label_feature(llm, f, **kw), features))  # type: ignore[arg-type]


def score_labels(
    llm: LLM, features: Sequence[FeatureData], labels: Mapping[str, str], *, workers: int = 8, **kw: object
) -> list[FeatureLabel]:
    """Score labels written elsewhere (e.g. Neuronpedia's) on the same held-out split, with the same detector."""
    todo = [f for f in features if labels.get(f.unit)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda f: label_feature(llm, f, label=labels[f.unit], **kw), todo))  # type: ignore[arg-type]
