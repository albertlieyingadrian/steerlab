"""A linear probe, and the honest way to score one: cross-validation, bootstrap CI, AUROC, shuffled labels.

    probe = LinearProbe().fit(X_train, y_train)           logistic regression on activation vectors
    probe.predict_proba(X)                                p(label 1) per row, computed with numpy only
    probe.save("outputs/probe_l14"); LinearProbe.load(...) .npz + .json, no pickle

    ev = evaluate_probe(X, y, groups=pair_ids, layer=14)  every honest number, in one typed object
    ev.cv_accuracy        ConfidenceInterval  mean of the 5 fold scores, with a pair-bootstrap 95% interval
    ev.auroc              ConfidenceInterval  out-of-fold ranking quality (0.5 = coin flip)
    ev.shuffled           ShuffledSummary     the same pipeline retrained on shuffled labels
    ev.to_result()        DetectorResult      the package-wide result type

The defaults reproduce the reference toolkit's probe exactly, which is what lets a rebuild of its probe get
identical fold scores: L2 logistic regression with C=1, lbfgs, max_iter=1000, random_state=42, no feature
scaling, and unshuffled StratifiedKFold(5). scikit-learn is used to FIT; predictions are plain numpy on the
saved weights, so a loaded probe needs neither sklearn nor pickle.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from sandbagkit.core import stats
from sandbagkit.core.types import ConfidenceInterval, DetectorResult
from sandbagkit.probing.io import load_arrays, save_arrays

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

TOOLKIT_SEED = 42
"""The reference toolkit's np.random.seed(42), used for its split and its LogisticRegression."""


def _as_matrix(X: npt.ArrayLike) -> FloatArray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"X must be 2-D [n_samples, n_features], got shape {matrix.shape}")
    return matrix


def _as_labels(y: npt.ArrayLike, n: int | None = None) -> IntArray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if n is not None and labels.shape[0] != n:
        raise ValueError(f"{n} rows but {labels.shape[0]} labels")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be 0 or 1")
    return labels.astype(np.int64)


def _bootstrap_method(groups: object) -> str:
    return "percentile bootstrap" if groups is None else "pair bootstrap"


# --- the probe --------------------------------------------------------------------------------------------------


class LinearProbe:
    """Binary logistic regression on activation vectors: p(1) = sigmoid(x · w + b)."""

    KIND = "linear_probe"

    def __init__(
        self, *, C: float = 1.0, max_iter: int = 1000, seed: int = TOOLKIT_SEED, standardize: bool = False
    ) -> None:
        if C <= 0:
            raise ValueError("C must be positive")
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.standardize = standardize
        self._coef: FloatArray | None = None  # [1, n_features], the same layout sklearn uses
        self._intercept: FloatArray | None = None  # [1]
        self._mean: FloatArray | None = None
        self._scale: FloatArray | None = None

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        from sklearn.linear_model import LogisticRegression

        matrix = _as_matrix(X)
        labels = _as_labels(y, matrix.shape[0])
        if len(np.unique(labels)) < 2:
            raise ValueError("fit needs both labels (0 and 1) in the training rows")
        if self.standardize:
            self._mean = matrix.mean(axis=0)
            scale = matrix.std(axis=0)
            self._scale = np.where(scale == 0, 1.0, scale)
        model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=self.max_iter, random_state=self.seed)
        model.fit(self._transform(matrix), labels)
        self._coef = np.asarray(model.coef_, dtype=np.float64)
        self._intercept = np.asarray(model.intercept_, dtype=np.float64)
        return self

    @property
    def fitted(self) -> bool:
        return self._coef is not None

    def _require_fitted(self) -> tuple[FloatArray, FloatArray]:
        if self._coef is None or self._intercept is None:
            raise RuntimeError("this probe isn't fitted yet; call fit(X, y) or LinearProbe.load(path)")
        return self._coef, self._intercept

    def _transform(self, matrix: FloatArray) -> FloatArray:
        if self._mean is None or self._scale is None:
            return matrix
        return (matrix - self._mean) / self._scale

    def decision_function(self, X: npt.ArrayLike) -> FloatArray:
        """x · w + b per row: positive means label 1. Same arithmetic as sklearn's decision_function."""
        coef, intercept = self._require_fitted()
        matrix = _as_matrix(X)
        if matrix.shape[1] != coef.shape[1]:
            raise ValueError(f"X has {matrix.shape[1]} features, the probe was trained on {coef.shape[1]}")
        scores: FloatArray = (self._transform(matrix) @ coef.T)[:, 0] + intercept[0]
        return scores

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """p(label 1) per row (a 1-D array, unlike sklearn's two columns)."""
        z = self.decision_function(X)
        proba: FloatArray = 1.0 / (1.0 + np.exp(-z))
        return proba

    def predict(self, X: npt.ArrayLike) -> IntArray:
        return (self.decision_function(X) > 0).astype(np.int64)

    def accuracy(self, X: npt.ArrayLike, y: npt.ArrayLike) -> float:
        predictions = self.predict(X)
        return float(np.mean(predictions == _as_labels(y, predictions.shape[0])))

    @property
    def coef(self) -> FloatArray:
        """The weight vector w, shape [n_features]."""
        weights: FloatArray = self._require_fitted()[0][0]
        return weights

    @property
    def intercept(self) -> float:
        return float(self._require_fitted()[1][0])

    @property
    def direction(self) -> FloatArray:
        """w scaled to length 1: the direction in activation space the probe reads."""
        w = self.coef
        unit: FloatArray = w / np.linalg.norm(w)
        return unit

    def config(self) -> dict[str, Any]:
        return {"C": self.C, "max_iter": self.max_iter, "seed": self.seed, "standardize": self.standardize}

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
        """Write `<path>.npz` (weights) and `<path>.json` (settings + your metadata, e.g. layer and model)."""
        coef, intercept = self._require_fitted()
        arrays: dict[str, FloatArray] = {"coef": coef, "intercept": intercept}
        if self._mean is not None and self._scale is not None:
            arrays |= {"mean": self._mean, "scale": self._scale}
        return save_arrays(path, arrays, kind=self.KIND, metadata={"config": self.config(), **dict(metadata or {})})

    @classmethod
    def load(cls, path: str | Path) -> LinearProbe:
        probe, _ = cls.load_with_metadata(path)
        return probe

    @classmethod
    def load_with_metadata(cls, path: str | Path) -> tuple[LinearProbe, dict[str, Any]]:
        arrays, metadata = load_arrays(path, kind=cls.KIND)
        probe = cls(**metadata.get("config", {}))
        probe._coef = arrays["coef"].astype(np.float64)
        probe._intercept = arrays["intercept"].astype(np.float64)
        if "mean" in arrays:
            probe._mean, probe._scale = arrays["mean"], arrays["scale"]
        return probe, metadata


ProbeFactory = Callable[[], LinearProbe]


# --- splits -----------------------------------------------------------------------------------------------------


def pair_split(n_pairs: int, *, train_fraction: float = 0.8, seed: int = TOOLKIT_SEED) -> tuple[IntArray, IntArray]:
    """(train pair indexes, test pair indexes): a seeded permutation, first `train_fraction` for training.

    With the defaults this is the toolkit's `np.random.seed(42); np.random.permutation(n)` split (25/7 for 32
    pairs), without touching numpy's global random state.
    """
    if n_pairs < 2:
        raise ValueError("need at least 2 pairs to split")
    order = np.random.RandomState(seed).permutation(n_pairs)
    cut = int(n_pairs * train_fraction)
    return order[:cut].astype(np.int64), order[cut:].astype(np.int64)


def pair_rows(pairs: Sequence[int] | npt.ArrayLike) -> IntArray:
    """Row numbers for pair indexes, for the layout "pair p is rows 2p (label 1) and 2p + 1 (label 0)"."""
    return np.array([row for p in np.asarray(pairs).tolist() for row in (2 * p, 2 * p + 1)], dtype=np.int64)


# --- cross-validation -------------------------------------------------------------------------------------------


@dataclass
class CrossValidation:
    """Per-fold accuracies, plus one out-of-fold p(1) per row (from a probe that never saw that row)."""

    fold_scores: list[float]
    oof_proba: FloatArray
    folds: list[IntArray] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))


def cross_validate(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    k: int = 5,
    groups: npt.ArrayLike | None = None,
    probe_factory: ProbeFactory = LinearProbe,
) -> CrossValidation:
    """k-fold CV with sklearn's unshuffled StratifiedKFold (the toolkit's `cross_val_score(cv=5)` folds).

    With `groups`, StratifiedGroupKFold is used instead, so rows that share a group (both prompts of a pair)
    always land in the same fold.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    matrix = _as_matrix(X)
    labels = _as_labels(y, matrix.shape[0])
    if groups is None:
        splits = StratifiedKFold(n_splits=k).split(matrix, labels)
    else:
        splits = StratifiedGroupKFold(n_splits=k).split(matrix, labels, np.asarray(groups))
    fold_scores, folds = [], []
    proba = np.zeros(labels.shape[0])
    for train, test in splits:
        probe = probe_factory().fit(matrix[train], labels[train])
        proba[test] = probe.predict_proba(matrix[test])
        fold_scores.append(float(np.mean(probe.predict(matrix[test]) == labels[test])))
        folds.append(test.astype(np.int64))
    return CrossValidation(fold_scores, proba, folds)


# --- uncertainty and baselines ----------------------------------------------------------------------------------


def group_bootstrap_ci(
    n_rows: int,
    groups: npt.ArrayLike | None,
    statistic: Callable[[IntArray], float],
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    method: str = "group bootstrap",
) -> ConfidenceInterval:
    """Percentile bootstrap that resamples whole GROUPS with replacement (rows of a group move together).

    `statistic(rows)` gets the row indexes of one resample. The two prompts of a contrast pair share a
    question, so they aren't independent evidence: resampling pairs, not prompts, gives an honest interval.
    `groups=None` makes every row its own group. Resamples where the statistic raises ValueError (e.g. AUROC on
    a resample with only one class) are skipped.
    """
    group_ids = np.arange(n_rows) if groups is None else np.asarray(groups)
    if group_ids.shape[0] != n_rows:
        raise ValueError(f"{n_rows} rows but {group_ids.shape[0]} group ids")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    members = [np.flatnonzero(group_ids == g) for g in np.unique(group_ids)]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(members), len(members))
        rows = np.concatenate([members[i] for i in picked])
        try:
            estimates.append(float(statistic(rows)))
        except ValueError:
            continue
    if not estimates:
        raise ValueError("the statistic failed on every resample")
    low, high = np.percentile(estimates, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    estimate = float(statistic(np.arange(n_rows)))
    return ConfidenceInterval(low=float(low), high=float(high), level=level, estimate=estimate, method=method)


def shuffled_label_baseline(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    observed: float | None = None,
    n_shuffles: int = 30,
    seed: int = 0,
    k: int = 5,
    groups: npt.ArrayLike | None = None,
    probe_factory: ProbeFactory = LinearProbe,
) -> stats.ShuffledBaseline:
    """Retrain the whole CV pipeline on shuffled labels, `n_shuffles` times.

    Each value is an out-of-fold accuracy, scored exactly like the real run. With hundreds of features and
    few rows, a probe can fit noise, so "90% accuracy" only means something if shuffled labels score far less.
    The p-value is the share of shuffles at least as good as `observed` (+1 on top and bottom).

    This differs from `core.stats.shuffled_baseline`, which shuffles labels against FIXED scores: here the
    probe is retrained on every shuffle, because the probe itself is what might be fitting noise.
    """
    matrix = _as_matrix(X)
    labels = _as_labels(y, matrix.shape[0])
    if observed is None:
        real = cross_validate(matrix, labels, k=k, groups=groups, probe_factory=probe_factory)
        observed = float(np.mean((real.oof_proba > 0.5) == labels))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_shuffles):
        fake = rng.permutation(labels)
        cv = cross_validate(matrix, fake, k=k, groups=groups, probe_factory=probe_factory)
        values.append(float(np.mean((cv.oof_proba > 0.5) == fake)))
    at_least_as_good = sum(v >= observed for v in values)
    return stats.ShuffledBaseline(
        observed=observed,
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        p_value=(at_least_as_good + 1) / (n_shuffles + 1),
        values=values,
    )


# --- the full evaluation ----------------------------------------------------------------------------------------


class ShuffledSummary(BaseModel):
    """The shuffled-label baseline in a JSON-friendly form: mean, middle-95% range and p-value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    low: float
    high: float
    p_value: float = Field(ge=0.0, le=1.0)
    n_shuffles: int = Field(ge=1)

    @classmethod
    def from_baseline(cls, baseline: stats.ShuffledBaseline) -> ShuffledSummary:
        low, high = np.percentile(baseline.values, [2.5, 97.5])
        return cls(
            mean=baseline.mean,
            low=float(low),
            high=float(high),
            p_value=baseline.p_value,
            n_shuffles=len(baseline.values),
        )


class ProbeEvaluation(BaseModel):
    """Everything a probe result must report: CV with a CI, AUROC with a CI, and a shuffled baseline."""

    model_config = ConfigDict(extra="forbid")

    layer: int | None = None
    n: int = Field(ge=0, description="training rows")
    n_groups: int = Field(ge=0, description="independent units the bootstrap resamples (pairs)")
    fold_scores: list[float]
    cv_accuracy: ConfidenceInterval
    auroc: ConfidenceInterval
    shuffled: ShuffledSummary
    test_accuracy: ConfidenceInterval | None = None
    test_auroc: float | None = None
    n_test: int = 0
    settings: dict[str, Any] = Field(default_factory=dict)

    @property
    def cv(self) -> float:
        """Mean of the fold scores: the toolkit's "CV accuracy"."""
        return float(np.mean(self.fold_scores))

    @property
    def beats_shuffled(self) -> bool:
        """CV accuracy above the top of the shuffled range, and shuffled p < 0.05."""
        return self.cv > self.shuffled.high and self.shuffled.p_value < 0.05

    def to_result(self, detector: str = "linear_probe") -> DetectorResult:
        """The package-wide result: detected = the activations carry a real, linearly readable signal."""
        ci = self.cv_accuracy
        where = f"layer {self.layer}: " if self.layer is not None else ""
        reason = (
            f"{where}CV {self.cv:.0%} [{ci.low:.0%}, {ci.high:.0%}], AUROC {self.auroc.estimate:.3f}, "
            f"shuffled labels {self.shuffled.mean:.0%} (p = {self.shuffled.p_value:.3f})"
        )
        return DetectorResult(
            detector=detector,
            detected=self.beats_shuffled and ci.low > 0.5,
            confidence=1.0 - self.shuffled.p_value,
            score=self.cv,
            p_value=self.shuffled.p_value,
            ci=ci,
            n=self.n,
            reason=reason,
            details=self.model_dump(mode="json"),
        )


def evaluate_probe(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    groups: npt.ArrayLike | None = None,
    X_test: npt.ArrayLike | None = None,
    y_test: npt.ArrayLike | None = None,
    groups_test: npt.ArrayLike | None = None,
    layer: int | None = None,
    k: int = 5,
    group_folds: bool = False,
    n_bootstrap: int = 2000,
    n_shuffles: int = 30,
    seed: int = 0,
    probe_factory: ProbeFactory = LinearProbe,
) -> ProbeEvaluation:
    """Cross-validate a probe on (X, y) and attach every number an honest probe result needs.

    - `groups`: which rows belong together (pair ids). The bootstrap resamples groups; with `group_folds=True`
      the CV folds also respect them. The default folds are the toolkit's (see `cross_validate`).
    - `X_test`/`y_test`: an optional held-out set, scored by one probe trained on all of (X, y).

    The defaults: 5 folds, 2000 bootstrap resamples and 30 shuffles, seed 0.
    """
    matrix = _as_matrix(X)
    labels = _as_labels(y, matrix.shape[0])
    fold_groups = groups if group_folds else None
    cv = cross_validate(matrix, labels, k=k, groups=fold_groups, probe_factory=probe_factory)
    correct = ((cv.oof_proba > 0.5) == labels).astype(np.float64)
    accuracy_ci = group_bootstrap_ci(
        len(labels),
        groups,
        lambda rows: float(correct[rows].mean()),
        n_resamples=n_bootstrap,
        seed=seed,
        method=_bootstrap_method(groups),
    )
    auroc_ci = group_bootstrap_ci(
        len(labels),
        groups,
        lambda rows: stats.auroc(labels[rows], cv.oof_proba[rows]),
        n_resamples=n_bootstrap,
        seed=seed,
        method=_bootstrap_method(groups),
    )
    baseline = shuffled_label_baseline(
        matrix,
        labels,
        observed=float(correct.mean()),
        n_shuffles=n_shuffles,
        seed=seed,
        k=k,
        groups=fold_groups,
        probe_factory=probe_factory,
    )

    test_ci, test_auroc, n_test = None, None, 0
    if X_test is not None or y_test is not None:
        if X_test is None or y_test is None:
            raise ValueError("pass both X_test and y_test")
        test_matrix = _as_matrix(X_test)
        test_labels = _as_labels(y_test, test_matrix.shape[0])
        probe = probe_factory().fit(matrix, labels)
        test_correct = (probe.predict(test_matrix) == test_labels).astype(np.float64)
        test_ci = group_bootstrap_ci(
            len(test_labels),
            groups_test,
            lambda rows: float(test_correct[rows].mean()),
            n_resamples=n_bootstrap,
            seed=seed,
            method=_bootstrap_method(groups_test),
        )
        if len(np.unique(test_labels)) == 2:
            test_auroc = stats.auroc(test_labels, probe.predict_proba(test_matrix))
        n_test = len(test_labels)

    n_groups = len(labels) if groups is None else len(np.unique(np.asarray(groups)))
    return ProbeEvaluation(
        layer=layer,
        n=len(labels),
        n_groups=n_groups,
        fold_scores=cv.fold_scores,
        cv_accuracy=accuracy_ci,
        auroc=auroc_ci,
        shuffled=ShuffledSummary.from_baseline(baseline),
        test_accuracy=test_ci,
        test_auroc=test_auroc,
        n_test=n_test,
        settings={
            "k": k,
            "group_folds": group_folds,
            "n_bootstrap": n_bootstrap,
            "n_shuffles": n_shuffles,
            "seed": seed,
            "probe": probe_factory().config(),
        },
    )


def evaluate_layers(
    activations: Mapping[int, npt.ArrayLike],
    y: npt.ArrayLike,
    *,
    train_rows: npt.ArrayLike | None = None,
    test_rows: npt.ArrayLike | None = None,
    groups: npt.ArrayLike | None = None,
    **kwargs: Any,
) -> dict[int, ProbeEvaluation]:
    """`evaluate_probe` at every layer of `{layer: [n_rows, hidden]}` (what extract_activations returns).

    `train_rows`/`test_rows` index into the full row set (and into `y` and `groups`). Without them, every row
    is used for CV and there is no held-out score.
    """
    labels = np.asarray(y)
    group_ids = None if groups is None else np.asarray(groups)
    rows = np.arange(labels.shape[0]) if train_rows is None else np.asarray(train_rows)
    held = None if test_rows is None else np.asarray(test_rows)
    results = {}
    for layer in sorted(activations):
        X = np.asarray(activations[layer], dtype=np.float64)
        results[layer] = evaluate_probe(
            X[rows],
            labels[rows],
            groups=None if group_ids is None else group_ids[rows],
            X_test=None if held is None else X[held],
            y_test=None if held is None else labels[held],
            groups_test=None if held is None or group_ids is None else group_ids[held],
            layer=layer,
            **kwargs,
        )
    return results


def best_layer(evaluations: Mapping[int, ProbeEvaluation]) -> int:
    """The layer with the highest CV accuracy (the earliest one on a tie)."""
    if not evaluations:
        raise ValueError("no evaluations")
    return max(sorted(evaluations), key=lambda layer: evaluations[layer].cv)
