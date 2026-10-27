"""Statistics core: every function is checked against scipy (or a hand-computed value) on fixed inputs."""

import math

import numpy as np
import pytest
from scipy import stats as sps

from sandbagkit.core import stats

# A sandbagging-shaped 2×2 table: rows = context (neutral, evaluation), columns = (correct, wrong).
TABLE = [[18, 2], [9, 11]]


class TestChiSquare:
    def test_independence_matches_scipy_with_yates(self):
        ours = stats.chi_square_independence(TABLE)
        ref = sps.chi2_contingency(TABLE, correction=True)
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)
        assert ours.dof == ref.dof == 1

    def test_independence_without_correction(self):
        ours = stats.chi_square_independence(TABLE, correction=False)
        ref = sps.chi2_contingency(TABLE, correction=False)
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)

    def test_independence_larger_table(self):
        table = [[10, 20, 30], [25, 15, 5]]
        ours = stats.chi_square_independence(table)
        ref = sps.chi2_contingency(table)
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)
        assert ours.dof == 2

    def test_goodness_of_fit_uniform(self):
        observed = [30, 10, 10, 10]  # a model that loves option A
        ours = stats.chi_square_goodness_of_fit(observed)
        ref = sps.chisquare(observed)
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)

    def test_goodness_of_fit_custom_expected(self):
        observed = [12, 18, 20]
        expected = [10, 20, 20]
        ours = stats.chi_square_goodness_of_fit(observed, expected)
        ref = sps.chisquare(observed, expected)
        assert ours.p_value == pytest.approx(ref.pvalue)

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="negative"):
            stats.chi_square_independence([[1, -1], [2, 3]])

    def test_rejects_zero_row(self):
        with pytest.raises(ValueError, match="zero"):
            stats.chi_square_independence([[0, 0], [2, 3]])


class TestFisher:
    @pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
    def test_matches_scipy(self, alternative):
        ours = stats.fisher_exact(TABLE, alternative=alternative)
        ref = sps.fisher_exact(TABLE, alternative=alternative)
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)

    @pytest.mark.parametrize("table", [[[3, 0], [0, 3]], [[1, 9], [11, 3]], [[0, 5], [5, 0]], [[7, 1], [2, 6]]])
    def test_matches_scipy_small_tables(self, table):
        ours = stats.fisher_exact(table)
        ref = sps.fisher_exact(table)
        assert ours.p_value == pytest.approx(ref.pvalue)
        assert ours.statistic == pytest.approx(ref.statistic)

    def test_empty_margin_gives_p_one(self):
        result = stats.fisher_exact([[0, 0], [3, 4]])
        assert result.p_value == 1.0
        assert math.isnan(result.statistic)


class TestCompareProportions:
    def test_small_counts_use_fisher(self):
        result = stats.compare_proportions(9, 10, 3, 10)
        assert result.method == "fisher"
        assert result.p_value == pytest.approx(sps.fisher_exact([[9, 1], [3, 7]]).pvalue)

    def test_large_counts_use_chi_square(self):
        result = stats.compare_proportions(80, 100, 60, 100)
        assert result.method == "chi-square"
        assert result.p_value == pytest.approx(sps.chi2_contingency([[80, 20], [60, 40]]).pvalue)

    def test_reports_difference_and_effect_size(self):
        result = stats.compare_proportions(80, 100, 60, 100)
        assert result.difference == pytest.approx(0.2)
        assert result.cohens_h == pytest.approx(stats.cohens_h(0.8, 0.6))


class TestEffectSizes:
    def test_cohens_d_hand_computed(self):
        a = [2, 4, 4, 4, 5, 5, 7, 9]
        b = [1, 2, 3, 4]
        pooled = math.sqrt(((8 - 1) * np.var(a, ddof=1) + (4 - 1) * np.var(b, ddof=1)) / (8 + 4 - 2))
        assert stats.cohens_d(a, b) == pytest.approx((np.mean(a) - np.mean(b)) / pooled)

    def test_cohens_d_sign_and_zero(self):
        assert stats.cohens_d([1, 2, 3], [1, 2, 3]) == 0.0
        assert stats.cohens_d([1, 2, 3], [4, 5, 6]) < 0

    def test_cohens_d_zero_variance(self):
        assert stats.cohens_d([1, 1], [1, 1]) == 0.0
        assert stats.cohens_d([2, 2], [1, 1]) == math.inf

    def test_cohens_d_needs_two_per_group(self):
        with pytest.raises(ValueError):
            stats.cohens_d([1], [1, 2])

    def test_cohens_h(self):
        assert stats.cohens_h(0.5, 0.5) == 0.0
        assert stats.cohens_h(1.0, 0.0) == pytest.approx(math.pi)
        assert stats.cohens_h(0.8, 0.6) == pytest.approx(2 * math.asin(math.sqrt(0.8)) - 2 * math.asin(math.sqrt(0.6)))

    def test_cohens_h_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            stats.cohens_h(1.2, 0.5)


class TestBootstrap:
    def test_close_to_scipy_percentile(self):
        rng = np.random.default_rng(1)
        data = rng.normal(10, 2, size=200)
        ours = stats.bootstrap_ci(data, n_resamples=20_000, seed=0)
        ref = sps.bootstrap((data,), np.mean, n_resamples=20_000, method="percentile", random_state=0)
        assert ours.low == pytest.approx(ref.confidence_interval.low, abs=0.05)
        assert ours.high == pytest.approx(ref.confidence_interval.high, abs=0.05)

    def test_contains_estimate_and_is_deterministic(self):
        data = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        first = stats.bootstrap_ci(data, seed=3)
        second = stats.bootstrap_ci(data, seed=3)
        assert first == second
        assert first.low <= first.estimate <= first.high
        assert first.estimate == pytest.approx(0.7)
        assert first.level == 0.95

    def test_narrows_with_more_data(self):
        rng = np.random.default_rng(0)
        small = stats.bootstrap_ci(rng.integers(0, 2, 20), seed=0)
        large = stats.bootstrap_ci(rng.integers(0, 2, 2000), seed=0)
        assert (large.high - large.low) < (small.high - small.low)

    def test_paired_rows_with_custom_statistic(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3])
        pairs = np.column_stack([labels, scores])
        ci = stats.bootstrap_ci(pairs, statistic=lambda rows: stats.auroc(rows[:, 0], rows[:, 1]), seed=0)
        assert ci.estimate == pytest.approx(stats.auroc(labels, scores))
        assert 0.0 <= ci.low <= ci.high <= 1.0

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            stats.bootstrap_ci([])


class TestKL:
    def test_identical_is_zero(self):
        assert stats.kl_divergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0, abs=1e-12)

    def test_matches_scipy_entropy(self):
        p, q = [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]
        assert stats.kl_divergence(p, q, smoothing=0.0) == pytest.approx(sps.entropy(p, q))

    def test_accepts_counts(self):
        assert stats.kl_divergence([1, 4, 5], [3, 3, 4], smoothing=0.0) == pytest.approx(
            sps.entropy([0.1, 0.4, 0.5], [0.3, 0.3, 0.4])
        )

    def test_zero_count_bin_is_infinite_without_smoothing(self):
        assert stats.kl_divergence([5, 5], [10, 0], smoothing=0.0) == math.inf

    def test_zero_count_bin_is_finite_with_smoothing(self):
        value = stats.kl_divergence([5, 5], [10, 0], smoothing=1.0)
        assert math.isfinite(value)
        assert value > 0

    def test_zero_in_p_contributes_nothing(self):
        assert stats.kl_divergence([1, 0], [0.5, 0.5], smoothing=0.0) == pytest.approx(math.log(2))

    def test_non_negative_and_asymmetric(self):
        p, q = [0.9, 0.1], [0.5, 0.5]
        assert stats.kl_divergence(p, q) > 0
        assert stats.kl_divergence(p, q) != pytest.approx(stats.kl_divergence(q, p))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            stats.kl_divergence([1, 2], [1, 2, 3])

    def test_js_divergence_symmetric_and_bounded(self):
        p, q = [1, 0], [0, 1]
        assert stats.js_divergence(p, q) == pytest.approx(math.log(2))
        assert stats.js_divergence([0.9, 0.1], [0.5, 0.5]) == pytest.approx(stats.js_divergence([0.5, 0.5], [0.9, 0.1]))

    def test_gaussian_kl_closed_form(self):
        expected = math.log(2 / 1) + (1**2 + (0 - 1) ** 2) / (2 * 2**2) - 0.5
        assert stats.gaussian_kl(0, 1, 1, 2) == pytest.approx(expected)

    def test_gaussian_kl_single_sample_is_finite(self):
        # The reference toolkit's bug: a one-sample "distribution" has std 0, and KL explodes.
        value = stats.gaussian_kl(5.0, 0.0, 3.0, 1.0)
        assert math.isfinite(value)


class TestBeta:
    def test_posterior_parameters(self):
        post = stats.beta_posterior(successes=7, failures=3)
        assert (post.alpha, post.beta) == (8.0, 4.0)

    def test_moments_match_scipy(self):
        post = stats.beta_posterior(7, 3, prior_alpha=2, prior_beta=2)
        dist = sps.beta(post.alpha, post.beta)
        assert post.mean == pytest.approx(dist.mean())
        assert post.variance == pytest.approx(dist.var())

    def test_credible_interval_matches_scipy(self):
        post = stats.beta_posterior(7, 3)
        ci = post.credible_interval(0.9)
        low, high = sps.beta(post.alpha, post.beta).interval(0.9)
        assert ci.low == pytest.approx(low)
        assert ci.high == pytest.approx(high)
        assert ci.level == 0.9

    def test_interval_narrows_with_more_samples(self):
        few = stats.beta_posterior(3, 1).credible_interval()
        many = stats.beta_posterior(300, 100).credible_interval()
        assert (many.high - many.low) < (few.high - few.low)

    def test_mode(self):
        assert stats.beta_posterior(7, 3).mode == pytest.approx(7 / 10)
        assert stats.beta_posterior(0, 0).mode is None  # Beta(1, 1) is flat: no single mode

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            stats.beta_posterior(-1, 3)
        with pytest.raises(ValueError):
            stats.beta_posterior(1, 3, prior_alpha=0)

    def test_wilson_matches_scipy(self):
        ci = stats.wilson_ci(5, 10)
        ref = sps.binomtest(5, 10).proportion_ci(confidence_level=0.95, method="wilson")
        assert ci.low == pytest.approx(ref.low)
        assert ci.high == pytest.approx(ref.high)

    def test_wilson_edges(self):
        assert stats.wilson_ci(0, 20).low == pytest.approx(0.0, abs=1e-12)
        assert stats.wilson_ci(20, 20).high == pytest.approx(1.0)
        with pytest.raises(ValueError):
            stats.wilson_ci(3, 0)


class TestKappa:
    def test_hand_computed(self):
        # Two raters, 10 items. Agreement 8/10; both say "yes" 6/10 times: pe = 0.6² + 0.4² = 0.52.
        a = ["y", "y", "y", "y", "y", "y", "n", "n", "n", "n"]
        b = ["y", "y", "y", "y", "y", "n", "y", "n", "n", "n"]
        assert stats.cohens_kappa(a, b) == pytest.approx((0.8 - 0.52) / (1 - 0.52))

    def test_matches_sklearn(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        a = rng.integers(0, 3, 50)
        b = np.where(rng.random(50) < 0.7, a, rng.integers(0, 3, 50))
        assert stats.cohens_kappa(a, b) == pytest.approx(sklearn_metrics.cohen_kappa_score(a, b))

    def test_perfect_and_degenerate(self):
        assert stats.cohens_kappa([1, 0, 1], [1, 0, 1]) == 1.0
        assert stats.cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0  # all one label: agreement is total

    def test_kappa_from_agreement(self):
        assert stats.kappa_from_agreement(0.8, 0.52) == pytest.approx(0.28 / 0.48)
        assert stats.kappa_from_agreement(1.0, 1.0) == 1.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            stats.cohens_kappa([1, 2], [1])


class TestZScore:
    def test_basic(self):
        assert stats.z_score(12, 10, 2) == 1.0

    def test_zero_std(self):
        assert stats.z_score(10, 10, 0) == 0.0
        assert stats.z_score(11, 10, 0) == math.inf
        assert stats.z_score(9, 10, 0) == -math.inf

    def test_mean_z_score_uses_standard_error(self):
        # A window of 25 values averaging 10.4 against a baseline of mean 10, std 1: SE = 1 / 5.
        window = [10.4] * 25
        assert stats.mean_z_score(window, 10.0, 1.0) == pytest.approx(2.0)


class TestAuroc:
    def test_matches_mann_whitney(self):
        labels = [0, 0, 0, 1, 1, 1, 1]
        scores = [0.1, 0.5, 0.5, 0.5, 0.7, 0.8, 0.2]
        positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
        negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
        u = sps.mannwhitneyu(positives, negatives).statistic
        assert stats.auroc(labels, scores) == pytest.approx(u / (len(positives) * len(negatives)))

    def test_perfect_and_chance(self):
        assert stats.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert stats.auroc([0, 1], [0.5, 0.5]) == 0.5

    def test_needs_both_classes(self):
        with pytest.raises(ValueError, match="both"):
            stats.auroc([1, 1], [0.2, 0.3])


class TestPermutation:
    def test_close_to_scipy(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(0.5, 1, 30), rng.normal(0, 1, 30)
        ours = stats.permutation_test(a, b, n_permutations=5000, seed=0)
        ref = sps.permutation_test(
            (a, b), lambda x, y: np.mean(x) - np.mean(y), n_resamples=5000, random_state=0, vectorized=False
        )
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue, abs=0.03)

    def test_identical_groups_not_significant(self):
        result = stats.permutation_test([1, 2, 3, 4], [1, 2, 3, 4], n_permutations=500, seed=0)
        assert result.p_value > 0.5

    def test_p_value_never_zero(self):
        result = stats.permutation_test([10, 11, 12], [0, 1, 2], n_permutations=200, seed=0)
        assert 0 < result.p_value <= 1

    def test_shuffled_baseline_near_chance(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200)
        scores = labels + rng.normal(0, 0.5, 200)  # a real signal
        baseline = stats.shuffled_baseline(labels, scores, stats.auroc, n_shuffles=200, seed=0)
        assert baseline.observed > 0.8
        assert baseline.mean == pytest.approx(0.5, abs=0.05)
        assert baseline.p_value < 0.01
        assert len(baseline.values) == 200

    def test_shuffled_baseline_deterministic(self):
        labels, scores = [0, 1, 0, 1, 1, 0], [0.2, 0.6, 0.1, 0.9, 0.4, 0.5]
        first = stats.shuffled_baseline(labels, scores, stats.auroc, n_shuffles=50, seed=1)
        second = stats.shuffled_baseline(labels, scores, stats.auroc, n_shuffles=50, seed=1)
        assert first.values == second.values
