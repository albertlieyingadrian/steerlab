"""The linear probe and its scoring: matches sklearn, round-trips exactly, reports CI + AUROC + shuffled baseline."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sandbagkit.core.stats import auroc
from sandbagkit.core.types import ConfidenceInterval, DetectorResult
from sandbagkit.probing import probe as pr


def pair_data(n_pairs=25, dim=40, signal=1.5, seed=0):
    """Contrast-pair-shaped data: pair p is rows 2p (label 1) and 2p + 1 (label 0), sharing a 'question' vector."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    rows, labels, groups = [], [], []
    for p in range(n_pairs):
        question = rng.normal(size=dim) * 2
        for label in (1, 0):
            rows.append(question + signal * label * direction + rng.normal(size=dim) * 0.5)
            labels.append(label)
            groups.append(p)
    return np.array(rows), np.array(labels), np.array(groups)


def toolkit_logreg():
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)


class TestLinearProbe:
    def test_matches_sklearn_exactly(self):
        X, y, _ = pair_data()
        ours = pr.LinearProbe().fit(X, y)
        theirs = toolkit_logreg().fit(X, y)
        np.testing.assert_array_equal(ours.predict(X), theirs.predict(X))
        np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X)[:, 1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ours.decision_function(X), theirs.decision_function(X), rtol=0, atol=1e-12)

    def test_save_load_round_trips_to_identical_predictions(self, tmp_path):
        X, y, _ = pair_data()
        probe = pr.LinearProbe().fit(X, y)
        npz, meta = probe.save(tmp_path / "probe_l14", metadata={"layer": 14, "model": "tiny"})
        assert npz.name == "probe_l14.npz" and meta.name == "probe_l14.json"
        loaded, metadata = pr.LinearProbe.load_with_metadata(tmp_path / "probe_l14")
        np.testing.assert_array_equal(loaded.predict_proba(X), probe.predict_proba(X))
        np.testing.assert_array_equal(loaded.predict(X), probe.predict(X))
        assert metadata["layer"] == 14
        assert loaded.config() == probe.config()

    def test_standardized_probe_round_trips_too(self, tmp_path):
        X, y, _ = pair_data()
        probe = pr.LinearProbe(standardize=True, C=0.5).fit(X * 100 + 3, y)
        probe.save(tmp_path / "scaled")
        loaded = pr.LinearProbe.load(tmp_path / "scaled.npz")
        np.testing.assert_array_equal(loaded.predict_proba(X * 100 + 3), probe.predict_proba(X * 100 + 3))
        assert loaded.standardize and loaded.C == 0.5

    def test_direction_is_unit_length(self):
        X, y, _ = pair_data()
        probe = pr.LinearProbe().fit(X, y)
        assert np.linalg.norm(probe.direction) == pytest.approx(1.0)
        assert probe.coef.shape == (40,)

    def test_unfitted_probe(self):
        with pytest.raises(RuntimeError, match="isn't fitted"):
            pr.LinearProbe().predict(np.zeros((2, 3)))

    def test_wrong_feature_count(self):
        X, y, _ = pair_data()
        probe = pr.LinearProbe().fit(X, y)
        with pytest.raises(ValueError, match="40"):
            probe.predict(np.zeros((2, 3)))

    @pytest.mark.parametrize(
        ("labels", "match"), [([0, 1, 2, 1], "0 or 1"), ([1, 1, 1, 1], "both labels"), ([0, 1], "4 rows")]
    )
    def test_bad_labels(self, labels, match):
        with pytest.raises(ValueError, match=match):
            pr.LinearProbe().fit(np.zeros((4, 3)), labels)


class TestSplits:
    def test_pair_split_is_the_toolkits_split_without_touching_global_state(self):
        np.random.seed(42)
        expected = np.random.permutation(32)
        np.random.seed(123)
        state = np.random.get_state()[1].copy()
        train, test = pr.pair_split(32)
        assert np.array_equal(np.random.get_state()[1], state)
        assert train.tolist() == expected[:25].tolist()
        assert test.tolist() == expected[25:].tolist()

    def test_pair_rows(self):
        assert pr.pair_rows([3, 0]).tolist() == [6, 7, 0, 1]


class TestCrossValidation:
    def test_folds_match_sklearn_cross_val_score(self):
        X, y, _ = pair_data()
        cv = pr.cross_validate(X, y)
        expected = cross_val_score(toolkit_logreg(), X, y, cv=5)
        assert cv.fold_scores == expected.tolist()
        assert len(cv.folds) == 5
        assert sorted(np.concatenate(cv.folds).tolist()) == list(range(len(y)))

    def test_group_folds_keep_pairs_together(self):
        X, y, groups = pair_data()
        cv = pr.cross_validate(X, y, groups=groups)
        for fold in cv.folds:
            for g in np.unique(groups[fold]):
                assert set(np.flatnonzero(groups == g)) <= set(fold.tolist())


class TestBootstrap:
    def test_matches_the_capstones_pair_bootstrap(self):
        """The reference bootstrap_ci, copied verbatim."""
        correct = (np.random.default_rng(5).random(50) > 0.2).astype(float)
        pair_ids = np.repeat(np.arange(25), 2)

        groups = [correct[pair_ids == p] for p in np.unique(pair_ids)]
        rng = np.random.default_rng(0)
        means = [
            np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(2000)
        ]
        low, high = np.percentile(means, [2.5, 97.5])

        ci = pr.group_bootstrap_ci(len(correct), pair_ids, lambda rows: float(correct[rows].mean()))
        assert isinstance(ci, ConfidenceInterval)
        assert (ci.low, ci.high) == pytest.approx((low, high), abs=1e-12)
        assert ci.estimate == pytest.approx(correct.mean())

    def test_constant_statistic_gives_a_zero_width_interval(self):
        ci = pr.group_bootstrap_ci(10, None, lambda rows: 0.7)
        assert ci.width == 0

    def test_resamples_where_the_statistic_fails_are_skipped(self):
        labels = np.array([1, 0] + [0] * 8)  # many resamples have no positive at all
        scores = np.linspace(0, 1, 10)
        ci = pr.group_bootstrap_ci(10, None, lambda rows: auroc(labels[rows], scores[rows]), n_resamples=200)
        assert 0 <= ci.low <= ci.high <= 1


class TestShuffledBaseline:
    def test_real_signal_beats_every_shuffle(self):
        X, y, _ = pair_data(signal=6.0)
        baseline = pr.shuffled_label_baseline(X, y, n_shuffles=20)
        assert baseline.observed > 0.9
        assert 0.3 < baseline.mean < 0.7
        assert baseline.p_value == pytest.approx(1 / 21)

    def test_no_signal_is_not_significant(self):
        X, y, _ = pair_data(signal=0.0)
        baseline = pr.shuffled_label_baseline(X, y, n_shuffles=20)
        assert baseline.p_value > 0.05


class TestEvaluateProbe:
    def test_reports_everything_ktd11_requires(self):
        X, y, groups = pair_data(signal=6.0)
        Xt, yt, gt = pair_data(n_pairs=7, signal=6.0, seed=1)
        ev = pr.evaluate_probe(X, y, groups=groups, X_test=Xt, y_test=yt, groups_test=gt, layer=14, n_shuffles=20)
        assert len(ev.fold_scores) == 5
        assert ev.cv == pytest.approx(np.mean(ev.fold_scores))
        assert ev.cv_accuracy.method == "pair bootstrap"
        assert ev.cv_accuracy.low <= ev.cv_accuracy.estimate <= ev.cv_accuracy.high
        assert ev.auroc.estimate > 0.9
        assert ev.shuffled.n_shuffles == 20
        assert (ev.n, ev.n_groups, ev.n_test) == (50, 25, 14)
        assert ev.test_accuracy is not None and ev.test_auroc is not None
        assert ev.beats_shuffled

    def test_to_result_is_the_core_detector_result(self):
        X, y, groups = pair_data(signal=6.0)
        ev = pr.evaluate_probe(X, y, groups=groups, layer=14, n_shuffles=20)  # 20 shuffles: p can reach 1/21
        result = ev.to_result()
        assert isinstance(result, DetectorResult)
        assert result.detected
        assert result.ci == ev.cv_accuracy
        assert result.p_value == ev.shuffled.p_value
        assert "layer 14" in result.reason
        restored = pr.ProbeEvaluation.model_validate(json.loads(json.dumps(result.details)))
        assert restored == ev

    def test_noise_is_not_detected(self):
        X, y, groups = pair_data(signal=0.0)
        result = pr.evaluate_probe(X, y, groups=groups, n_shuffles=10).to_result()
        assert not result.detected

    def test_test_set_needs_both_halves(self):
        X, y, _ = pair_data()
        with pytest.raises(ValueError, match="both"):
            pr.evaluate_probe(X, y, X_test=X, n_shuffles=2)

    def test_evaluate_layers_and_best_layer(self):
        weak, y, groups = pair_data(signal=0.3)
        strong, _, _ = pair_data(signal=6.0)
        train_rows, test_rows = pr.pair_rows(range(20)), pr.pair_rows(range(20, 25))
        results = pr.evaluate_layers(
            {2: weak, 14: strong}, y, train_rows=train_rows, test_rows=test_rows, groups=groups, n_shuffles=5
        )
        assert sorted(results) == [2, 14]
        assert results[14].n == 40 and results[14].n_test == 10
        assert pr.best_layer(results) == 14
