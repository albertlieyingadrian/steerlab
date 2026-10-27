"""The statistics every detector needs, written out by hand.

scipy is used only for the probability distributions themselves (chi² tail areas, Beta quantiles, normal
quantiles, ranks). The tests check each function against scipy's own implementation.

Plain-language guide:

| function                   | answers the question                                                     |
|----------------------------|--------------------------------------------------------------------------|
| chi_square_independence    | do two categorical things (context, correctness) move together?         |
| chi_square_goodness_of_fit | are these counts (e.g. chosen option A/B/C/D) spread as expected?        |
| fisher_exact               | same as the 2×2 chi-square, but exact, for small counts                  |
| compare_proportions        | is accuracy A different from accuracy B? (picks Fisher or chi-square)    |
| cohens_d / cohens_h        | how BIG is a difference (not just: is it real?)                         |
| bootstrap_ci               | how uncertain is a number computed from a sample?                        |
| kl_divergence / js         | how different are two distributions?                                     |
| beta_posterior, wilson_ci  | what rate is plausible, given k successes out of n?                      |
| cohens_kappa               | do two sets of labels agree more than chance would?                      |
| z_score, mean_z_score      | how many standard deviations away from normal is this?                  |
| auroc                      | how well does a score rank positives above negatives?                    |
| permutation_test,          | could this result have come from random labels?                          |
| shuffled_baseline          |                                                                          |
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import stats as _sps  # type: ignore[import-untyped]  # scipy ships no type hints

from sandbagkit.core.types import ConfidenceInterval

Alternative = Literal["two-sided", "less", "greater"]


class SignificanceResult(NamedTuple):
    """A test statistic and its p-value: the chance of a result at least this extreme if nothing is going on."""

    statistic: float
    p_value: float
    dof: int | None = None


# --- Contingency tables ----------------------------------------------------------------------------------------


def _counts_table(table: npt.ArrayLike) -> npt.NDArray[np.float64]:
    counts = np.asarray(table, dtype=float)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError(f"expected a 2-D table of at least 2×2 counts, got shape {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("counts cannot be negative")
    return counts


def chi_square_independence(table: npt.ArrayLike, *, correction: bool = True) -> SignificanceResult:
    """Pearson's chi-square test of independence on an r×c table of counts.

    "Expected" is what each cell would hold if rows and columns were unrelated: row_total × col_total / total.
    The statistic adds up (observed − expected)² / expected. For 2×2 tables, `correction` applies Yates'
    continuity correction (each |observed − expected| shrinks by up to 0.5), matching scipy's default.
    """
    observed = _counts_table(table)
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    if np.any(row_totals == 0) or np.any(col_totals == 0):
        raise ValueError("table has a zero row or column total; use fisher_exact or drop the empty category")
    expected = np.outer(row_totals, col_totals) / observed.sum()
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    gap = np.abs(observed - expected)
    if correction and dof == 1:
        gap = np.maximum(gap - 0.5, 0.0)
    statistic = float(np.sum(gap**2 / expected))
    return SignificanceResult(statistic, float(_sps.chi2.sf(statistic, dof)), dof)


def chi_square_goodness_of_fit(observed: npt.ArrayLike, expected: npt.ArrayLike | None = None) -> SignificanceResult:
    """Do the observed counts match the expected counts? Expected defaults to "all categories equally likely"."""
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 1 or obs.size < 2:
        raise ValueError("observed must be a 1-D sequence of at least 2 counts")
    if np.any(obs < 0):
        raise ValueError("counts cannot be negative")
    if expected is None:
        exp = np.full(obs.size, obs.sum() / obs.size)
    else:
        exp = np.asarray(expected, dtype=float)
        if exp.shape != obs.shape:
            raise ValueError("observed and expected must have the same length")
        if not math.isclose(exp.sum(), obs.sum(), rel_tol=1e-8):
            raise ValueError("observed and expected must have the same total")
    if np.any(exp <= 0):
        raise ValueError("expected counts must be positive")
    statistic = float(np.sum((obs - exp) ** 2 / exp))
    dof = obs.size - 1
    return SignificanceResult(statistic, float(_sps.chi2.sf(statistic, dof)), dof)


def fisher_exact(table: npt.ArrayLike, *, alternative: Alternative = "two-sided") -> SignificanceResult:
    """Fisher's exact test on a 2×2 table [[a, b], [c, d]]. Returns (odds ratio, p-value).

    With the row and column totals fixed, the top-left count follows a hypergeometric distribution (like
    drawing cards without replacement). The two-sided p-value adds up the probability of every table that is
    at most as likely as the one observed.
    """
    counts = _counts_table(table)
    if counts.shape != (2, 2):
        raise ValueError("fisher_exact needs a 2×2 table")
    if not np.all(counts == np.round(counts)):
        raise ValueError("fisher_exact needs whole-number counts")
    a, b, c, d = (int(x) for x in counts.ravel())
    row1, col1, total = a + b, a + c, a + b + c + d
    if min(row1, c + d, col1, b + d) == 0:
        return SignificanceResult(math.nan, 1.0)
    odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else math.inf

    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / math.comb(total, row1)

    support = range(max(0, row1 - (b + d)), min(row1, col1) + 1)
    if alternative == "less":
        p = sum(probability(x) for x in support if x <= a)
    elif alternative == "greater":
        p = sum(probability(x) for x in support if x >= a)
    elif alternative == "two-sided":
        p_observed = probability(a)
        p = sum(probability(x) for x in support if probability(x) <= p_observed * (1 + 1e-7))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return SignificanceResult(float(odds_ratio), min(1.0, p))


class ProportionComparison(NamedTuple):
    rate_a: float
    rate_b: float
    difference: float
    cohens_h: float
    p_value: float
    method: Literal["fisher", "chi-square"]


def compare_proportions(successes_a: int, n_a: int, successes_b: int, n_b: int) -> ProportionComparison:
    """Compare two success rates (e.g. neutral vs evaluation accuracy).

    Uses Fisher's exact test when any expected cell count is below 5 (the usual rule of thumb for when the
    chi-square approximation is unreliable), otherwise the Yates-corrected chi-square test.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both groups need at least one trial")
    if not (0 <= successes_a <= n_a and 0 <= successes_b <= n_b):
        raise ValueError("successes must be between 0 and n")
    table = np.array([[successes_a, n_a - successes_a], [successes_b, n_b - successes_b]], dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    rate_a, rate_b = successes_a / n_a, successes_b / n_b
    h = cohens_h(rate_a, rate_b)
    if np.any(expected < 5):
        return ProportionComparison(rate_a, rate_b, rate_a - rate_b, h, fisher_exact(table).p_value, "fisher")
    p_value = chi_square_independence(table).p_value
    return ProportionComparison(rate_a, rate_b, rate_a - rate_b, h, p_value, "chi-square")


# --- Effect sizes ----------------------------------------------------------------------------------------------


def cohens_d(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Difference in means, measured in pooled standard deviations. 0.2 small, 0.5 medium, 0.8 large.

    Returns ±inf when both groups have zero spread but different means.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError("cohens_d needs at least 2 values per group")
    difference = float(x.mean() - y.mean())
    pooled_var = ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (x.size + y.size - 2)
    if pooled_var == 0:
        return 0.0 if difference == 0 else math.copysign(math.inf, difference)
    return difference / math.sqrt(pooled_var)


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions: 2·asin(√p1) − 2·asin(√p2). Same rough scale as Cohen's d."""
    for p in (p1, p2):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"proportions must be in [0, 1], got {p}")
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


# --- Bootstrap -------------------------------------------------------------------------------------------------


def bootstrap_ci(
    data: npt.ArrayLike,
    statistic: Callable[[npt.NDArray[Any]], float] | None = None,
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval.

    Resample the data WITH replacement (same size), recompute the statistic, repeat many times; the middle
    `level` share of those values is the interval. Rows are resampled together, so for paired data (labels and
    scores) pass `np.column_stack([labels, scores])`. Resamples where `statistic` raises ValueError (e.g. AUROC
    on a resample with one class) are skipped.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    stat = statistic if statistic is not None else (lambda sample: float(np.mean(sample)))
    rng = np.random.default_rng(seed)
    n = values.shape[0]

    estimates = []
    for _ in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        try:
            estimates.append(float(stat(sample)))
        except ValueError:
            continue
    if not estimates:
        raise ValueError("the statistic failed on every resample")
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(estimates, [tail, 100 - tail])
    return ConfidenceInterval(
        low=float(low), high=float(high), level=level, estimate=float(stat(values)), method="percentile bootstrap"
    )


# --- Divergences -----------------------------------------------------------------------------------------------


def _distribution(values: npt.ArrayLike, smoothing: float) -> npt.NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("expected a non-empty 1-D sequence")
    if np.any(arr < 0):
        raise ValueError("probabilities or counts cannot be negative")
    if smoothing < 0:
        raise ValueError("smoothing cannot be negative")
    arr = arr + smoothing
    total = arr.sum()
    if total == 0:
        raise ValueError("all values are zero")
    normalized: npt.NDArray[np.float64] = arr / total
    return normalized


def kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, *, smoothing: float = 1e-9) -> float:
    """KL(P ‖ Q) = Σ p·ln(p/q) in nats: the extra surprise from assuming Q when the truth is P.

    Accepts probabilities or raw counts (both are normalized). It is 0 when P = Q and grows without limit, so
    a bin where Q is 0 but P isn't gives infinity. `smoothing` is added to every bin first (a pseudocount):
    for counts, 0.5 or 1 (Laplace) is typical; the tiny default only keeps the result finite.
    """
    p_arr, q_arr = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same length")
    p_dist, q_dist = _distribution(p_arr, smoothing), _distribution(q_arr, smoothing)
    mask = p_dist > 0
    if np.any(q_dist[mask] == 0):
        return math.inf
    return float(np.sum(p_dist[mask] * np.log(p_dist[mask] / q_dist[mask])))


def js_divergence(p: npt.ArrayLike, q: npt.ArrayLike) -> float:
    """Jensen-Shannon divergence: a symmetric, always-finite cousin of KL, between 0 and ln 2."""
    p_dist, q_dist = _distribution(p, 0.0), _distribution(q, 0.0)
    if p_dist.shape != q_dist.shape:
        raise ValueError("p and q must have the same length")
    middle = (p_dist + q_dist) / 2
    return 0.5 * kl_divergence(p_dist, middle, smoothing=0.0) + 0.5 * kl_divergence(q_dist, middle, smoothing=0.0)


def gaussian_kl(mean_p: float, std_p: float, mean_q: float, std_q: float, *, min_std: float = 1e-3) -> float:
    """KL between two normal distributions, KL(N(mean_p, std_p²) ‖ N(mean_q, std_q²)).

    Standard deviations are floored at `min_std`. Without the floor a one-sample "distribution" (std 0) makes
    the divergence infinite, which is the reference toolkit's single-sample bug.
    """
    sp, sq = max(std_p, min_std), max(std_q, min_std)
    return math.log(sq / sp) + (sp**2 + (mean_p - mean_q) ** 2) / (2 * sq**2) - 0.5


# --- Rates: Beta-Binomial and Wilson ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaPosterior:
    """Belief about an unknown rate after seeing successes and failures: Beta(alpha, beta).

    Start from a prior Beta(a, b) (Beta(1, 1) = "any rate equally likely"). Each success adds 1 to alpha and
    each failure adds 1 to beta. That's the whole Bayesian update for a yes/no rate.
    """

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total**2 * (total + 1))

    @property
    def mode(self) -> float | None:
        """Most likely rate; None when the density has no single peak (alpha or beta <= 1)."""
        if self.alpha <= 1 or self.beta <= 1:
            return None
        return (self.alpha - 1) / (self.alpha + self.beta - 2)

    def update(self, successes: int, failures: int) -> BetaPosterior:
        return beta_posterior(successes, failures, prior_alpha=self.alpha, prior_beta=self.beta)

    def credible_interval(self, level: float = 0.95) -> ConfidenceInterval:
        """Equal-tailed interval holding `level` of the posterior probability."""
        tail = (1 - level) / 2
        low, high = _sps.beta.ppf([tail, 1 - tail], self.alpha, self.beta)
        return ConfidenceInterval(
            low=float(low), high=float(high), level=level, estimate=self.mean, method="beta posterior"
        )


def beta_posterior(
    successes: int, failures: int, *, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> BetaPosterior:
    if successes < 0 or failures < 0:
        raise ValueError("successes and failures cannot be negative")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior parameters must be positive")
    return BetaPosterior(float(prior_alpha + successes), float(prior_beta + failures))


def wilson_ci(successes: int, n: int, *, level: float = 0.95) -> ConfidenceInterval:
    """Wilson score interval for a proportion. Unlike p ± 1.96·SE it stays inside [0, 1] and works at k = 0."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = float(_sps.norm.ppf(1 - (1 - level) / 2))
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return ConfidenceInterval(
        low=max(0.0, center - half), high=min(1.0, center + half), level=level, estimate=p, method="wilson"
    )


# --- Agreement -------------------------------------------------------------------------------------------------


def kappa_from_agreement(observed: float, chance: float) -> float:
    """κ = (observed agreement − chance agreement) / (1 − chance agreement). 1 = perfect, 0 = chance level."""
    if chance >= 1:
        return 1.0 if observed >= 1 else 0.0
    return (observed - chance) / (1 - chance)


def cohens_kappa(labels_a: Sequence[Any] | npt.ArrayLike, labels_b: Sequence[Any] | npt.ArrayLike) -> float:
    """Cohen's kappa: agreement between two label sequences, corrected for the agreement chance would give."""
    a, b = list(np.asarray(labels_a).tolist()), list(np.asarray(labels_b).tolist())
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be non-empty and the same length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    chance = 0.0
    for label in set(a) | set(b):
        chance += (a.count(label) / n) * (b.count(label) / n)
    return kappa_from_agreement(observed, chance)


# --- z-scores --------------------------------------------------------------------------------------------------


def z_score(value: float, mean: float, std: float) -> float:
    """How many standard deviations `value` is from `mean`. Zero spread gives 0 (same value) or ±inf."""
    if std < 0:
        raise ValueError("std cannot be negative")
    difference = value - mean
    if std == 0:
        return 0.0 if difference == 0 else math.copysign(math.inf, difference)
    return difference / std


def mean_z_score(values: npt.ArrayLike, mean: float, std: float) -> float:
    """z-score of a window's MEAN against a per-sample baseline, using the standard error std / √n.

    The mean of n values wobbles √n times less than a single value, so comparing it against the per-sample std
    (as the reference toolkit does) misjudges how unusual a window's mean is.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values cannot be empty")
    return z_score(float(arr.mean()), mean, std / math.sqrt(arr.size))


# --- Ranking and random baselines ------------------------------------------------------------------------------


def auroc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the ROC curve: the chance a random positive scores above a random negative (ties count ½).

    1.0 = perfect ranking, 0.5 = coin flip. Computed from ranks (the Mann-Whitney U statistic).
    """
    y = np.asarray(labels).astype(float)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be 1-D and the same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc needs both positive and negative labels")
    ranks = _sps.rankdata(s)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _mean_difference(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    return float(np.mean(x) - np.mean(y))


def permutation_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    statistic: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float] = _mean_difference,
    *,
    n_permutations: int = 10_000,
    alternative: Alternative = "two-sided",
    seed: int = 0,
) -> SignificanceResult:
    """Is the difference between groups a and b bigger than random regrouping would give?

    Pool both groups, shuffle, split back into the original sizes, recompute; repeat. The p-value is the share
    of shuffles at least as extreme as the real split, with +1 on top and bottom so it is never exactly 0.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"unknown alternative {alternative!r}")
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("both groups need at least one value")
    observed = statistic(x, y)
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    tolerance = 1e-12 * max(1.0, abs(observed))

    extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        value = statistic(shuffled[: x.size], shuffled[x.size :])
        if alternative == "two-sided":
            extreme += abs(value) >= abs(observed) - tolerance
        elif alternative == "greater":
            extreme += value >= observed - tolerance
        else:
            extreme += value <= observed + tolerance
    return SignificanceResult(observed, (extreme + 1) / (n_permutations + 1))


class ShuffledBaseline(NamedTuple):
    """What a metric scores when the labels are shuffled, i.e. when there is no real signal to find."""

    observed: float
    mean: float
    std: float
    p_value: float
    values: list[float]


def shuffled_baseline(
    labels: npt.ArrayLike,
    scores: npt.ArrayLike,
    metric: Callable[[npt.NDArray[Any], npt.NDArray[Any]], float],
    *,
    n_shuffles: int = 100,
    seed: int = 0,
) -> ShuffledBaseline:
    """Recompute `metric(labels, scores)` with shuffled labels, `n_shuffles` times (higher metric = better).

    If the real metric isn't clearly above the shuffled ones, the "signal" could be an artifact of the data
    (class balance, small n) rather than anything the scores know.
    """
    y, s = np.asarray(labels), np.asarray(scores)
    if y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must be the same length")
    observed = float(metric(y, s))
    rng = np.random.default_rng(seed)
    values = [float(metric(rng.permutation(y), s)) for _ in range(n_shuffles)]
    at_least_as_good = sum(v >= observed for v in values)
    return ShuffledBaseline(
        observed=observed,
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        p_value=(at_least_as_good + 1) / (n_shuffles + 1),
        values=values,
    )
