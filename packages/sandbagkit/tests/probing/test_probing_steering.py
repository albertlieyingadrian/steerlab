"""Steering: the known mean difference, the train-only guard, and an injector that is exact and cleans up."""

import numpy as np
import pytest
import torch

from sandbagkit.probing import hooks as h
from sandbagkit.probing import steering as st


def _ids(tokenizer, text="the capital of france is paris"):
    return tokenizer(text, return_tensors="pt")["input_ids"]


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids).logits


class TestMeanDifference:
    def test_equals_the_known_mean_difference(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [2.0, 2.0]])
        labels = np.array([1, 1, 0, 0])
        # mean(label 1) = [2, 3], mean(label 0) = [1, 1]
        assert st.mean_difference(X, labels).tolist() == [1.0, 2.0]

    def test_synthetic_tensors(self):
        rng = np.random.default_rng(0)
        offset = rng.normal(size=16)
        noise = rng.normal(size=(30, 16))
        noise -= noise.mean(axis=0)  # zero-mean noise, so the class means are exact
        X = np.vstack([noise + offset, noise])
        labels = np.array([1] * 30 + [0] * 30)
        np.testing.assert_allclose(st.mean_difference(X, labels), offset, atol=1e-12)

    def test_needs_both_labels(self):
        with pytest.raises(ValueError, match="each label"):
            st.mean_difference(np.ones((3, 2)), [1, 1, 1])


class TestTrainOnlyGuard:
    def setup_method(self):
        # 4 pairs: rows 2p (label 1) and 2p + 1 (label 0). Pairs 2 and 3 are "test" and hold huge values,
        # so any leak would show up in the vector.
        self.X = np.array([[1.0], [0.0], [3.0], [0.0], [1000.0], [0.0], [1000.0], [0.0]])
        self.labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    def test_test_groups_never_contribute(self):
        vec = st.steering_vector(self.X, self.labels, groups=self.groups, test_groups=[2, 3], layer=1)
        assert vec.vector.tolist() == [2.0]  # mean of 1 and 3, minus 0
        assert vec.train_groups == (0, 1)
        assert (vec.n_positive, vec.n_negative, vec.layer) == (2, 2, 1)

    def test_explicit_train_groups(self):
        vec = st.steering_vector(self.X, self.labels, groups=self.groups, train_groups=[1], test_groups=[2, 3])
        assert vec.vector.tolist() == [3.0]

    def test_overlap_is_leakage(self):
        with pytest.raises(st.LeakageError, match=r"\[2\]"):
            st.steering_vector(self.X, self.labels, groups=self.groups, train_groups=[0, 1, 2], test_groups=[2, 3])

    def test_test_groups_need_group_ids(self):
        with pytest.raises(ValueError, match="groups="):
            st.steering_vector(self.X, self.labels, test_groups=[2])

    def test_unguarded_vector_says_so(self):
        vec = st.steering_vector(self.X, self.labels)
        assert vec.train_groups is None
        assert vec.vector.tolist() == [501.0]

    def test_save_and_load(self, tmp_path):
        vec = st.steering_vector(self.X, self.labels, groups=self.groups, test_groups=[2, 3], layer=14)
        vec.save(tmp_path / "dir", metadata={"model": "tiny"})
        loaded = st.SteeringVector.load(tmp_path / "dir")
        np.testing.assert_array_equal(loaded.vector, vec.vector)
        assert (loaded.layer, loaded.train_groups, loaded.n_positive) == (14, (0, 1), 2)
        assert loaded.metadata == {"model": "tiny"}
        assert loaded.norm == pytest.approx(2.0)
        assert np.linalg.norm(loaded.unit()) == pytest.approx(1.0)


class TestInject:
    def test_strength_zero_does_not_change_logits(self, tiny_model, tokenizer):
        input_ids = _ids(tokenizer)
        before = _logits(tiny_model, input_ids)
        vector = np.random.default_rng(0).normal(size=32) * 10
        with st.inject(tiny_model, vector, layer=1, strength=0.0) as injection:
            during = _logits(tiny_model, input_ids)
        assert injection.calls == 1  # the hook did run; it just added zero
        assert torch.equal(before, during)

    def test_nonzero_strength_adds_exactly_strength_times_vector(self, tiny_model, tokenizer):
        input_ids = _ids(tokenizer)
        vector = torch.randn(32, generator=torch.Generator().manual_seed(0))
        with torch.no_grad(), h.capture_layers(tiny_model, [1]) as plain:
            tiny_model(input_ids=input_ids)
        baseline = plain[1].clone()
        # the capture is registered after the injector, so it sees the steered output
        with (
            st.inject(tiny_model, vector, layer=1, strength=-2.0),
            torch.no_grad(),
            h.capture_layers(tiny_model, [1]) as steered,
        ):
            logits = tiny_model(input_ids=input_ids).logits
        assert torch.allclose(steered[1], baseline - 2.0 * vector, atol=1e-6)
        assert not torch.allclose(logits, _logits(tiny_model, input_ids))

    def test_exiting_removes_the_hook(self, tiny_model, tokenizer):
        input_ids = _ids(tokenizer)
        before_logits = _logits(tiny_model, input_ids)
        before = h.count_forward_hooks(tiny_model)
        with st.inject(tiny_model, np.ones(32), layer=2, strength=5.0):
            assert h.count_forward_hooks(tiny_model) == before + 1
        assert h.count_forward_hooks(tiny_model) == before
        assert torch.equal(_logits(tiny_model, input_ids), before_logits)

    def test_hook_removed_even_when_the_block_raises(self, tiny_model):
        before = h.count_forward_hooks(tiny_model)
        with pytest.raises(KeyError), st.inject(tiny_model, np.ones(32), layer=0):
            raise KeyError("oops")
        assert h.count_forward_hooks(tiny_model) == before

    def test_uses_the_vectors_own_layer(self, tiny_model):
        vec = st.SteeringVector(np.ones(32), layer=-1)
        with st.inject(tiny_model, vec) as injection:
            assert injection.layer == 3

    def test_needs_a_layer(self, tiny_model):
        with pytest.raises(ValueError, match="which layer"), st.inject(tiny_model, np.ones(32)):
            pass

    def test_wrong_size(self, tiny_model):
        with pytest.raises(ValueError, match="hidden size is 32"), st.inject(tiny_model, np.ones(8), layer=0):
            pass

    def test_unsupported_layer(self, tiny_model):
        with pytest.raises(h.LayerIndexError), st.inject(tiny_model, np.ones(32), layer=4):
            pass

    def test_tuple_output_layers(self, tuple_stack):
        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            plain = tuple_stack(x)
            with st.inject(tuple_stack, torch.zeros(32), layer=1, strength=3.0):
                same = tuple_stack(x)
            with st.inject(tuple_stack, torch.ones(32), layer=2, strength=1.0):
                shifted = tuple_stack(x)
        assert torch.equal(plain, same)
        assert torch.allclose(shifted, plain + 1.0)
