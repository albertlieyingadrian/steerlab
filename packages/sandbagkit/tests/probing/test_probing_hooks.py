"""Hooks: one forward pass captures every requested layer, tuples and tensors both work, hooks always go away."""

import pytest
import torch

from sandbagkit.probing import hooks as h


def _ids(tokenizer, text):
    return tokenizer(text, return_tensors="pt")["input_ids"]


def _noop(module, inputs, output):
    return None


class TestCaptureLayers:
    def test_layers_1_and_2_match_output_hidden_states(self, tiny_model, tokenizer):
        input_ids = _ids(tokenizer, "the capital of france is paris")
        with torch.no_grad():
            reference = tiny_model(input_ids=input_ids, output_hidden_states=True).hidden_states
            with h.capture_layers(tiny_model, [1, 2]) as captured:
                tiny_model(input_ids=input_ids)  # ONE forward pass
        assert sorted(captured) == [1, 2]
        for layer in (1, 2):
            # hidden_states[0] is the embeddings, so layer i's output is hidden_states[i + 1]
            assert torch.allclose(captured[layer], reference[layer + 1], atol=1e-6)

    def test_one_forward_pass_fills_every_layer(self, tiny_model, tokenizer):
        calls = []
        handle = tiny_model.register_forward_hook(lambda module, inputs, output: calls.append(1))
        try:
            with torch.no_grad(), h.capture_layers(tiny_model, [0, 1, 2, 3]) as captured:
                tiny_model(input_ids=_ids(tokenizer, "hello world"))
        finally:
            handle.remove()
        assert len(calls) == 1
        assert sorted(captured) == [0, 1, 2, 3]

    def test_captured_tensors_are_detached(self, tiny_model, tokenizer):
        with h.capture_layers(tiny_model, [1]) as captured:
            tiny_model(input_ids=_ids(tokenizer, "hello world"))
        assert not captured[1].requires_grad

    def test_tuple_outputs(self, tuple_stack):
        x = torch.randn(2, 5, 32)
        with torch.no_grad(), h.capture_layers(tuple_stack, [0, 2], layers_path="layers") as captured:
            out = tuple_stack(x)
        assert captured[0].shape == (2, 5, 32)
        assert torch.equal(captured[2], out)


class TestHookRemoval:
    def test_hooks_removed_on_exit(self, tiny_model, tokenizer):
        before = h.count_forward_hooks(tiny_model)
        with h.capture_layers(tiny_model, [0, 1, 2]):
            assert h.count_forward_hooks(tiny_model) == before + 3
            tiny_model(input_ids=_ids(tokenizer, "hello"))
        assert h.count_forward_hooks(tiny_model) == before

    def test_hooks_removed_when_the_block_raises(self, tiny_model):
        before = h.count_forward_hooks(tiny_model)
        with pytest.raises(RuntimeError, match="boom"), h.capture_layers(tiny_model, [1, 3]):
            raise RuntimeError("boom")
        assert h.count_forward_hooks(tiny_model) == before

    def test_hooks_removed_when_a_hook_raises_mid_forward(self, tiny_model, tokenizer):
        def bad_hook(module, inputs, output):
            raise ValueError("hook failed")

        before = h.count_forward_hooks(tiny_model)
        with pytest.raises(ValueError, match="hook failed"), h.register_hooks(tiny_model, {0: _noop, 2: bad_hook}):
            tiny_model(input_ids=_ids(tokenizer, "hello"))
        assert h.count_forward_hooks(tiny_model) == before

    def test_nothing_registered_when_a_layer_is_invalid(self, tiny_model):
        before = h.count_forward_hooks(tiny_model)
        with pytest.raises(h.LayerIndexError), h.register_hooks(tiny_model, {0: _noop, 9: _noop}):
            pass
        assert h.count_forward_hooks(tiny_model) == before

    def test_two_hooks_on_the_same_layer_are_rejected(self, tiny_model):
        with pytest.raises(h.LayerIndexError, match="same layer"), h.register_hooks(tiny_model, {3: _noop, -1: _noop}):
            pass


class TestLayerIndexes:
    @pytest.mark.parametrize("bad", [4, 100, -5])
    def test_out_of_range_is_a_clear_error(self, tiny_model, bad):
        with pytest.raises(h.LayerIndexError, match=r"this model has 4 layers \(valid: 0..3"):
            h.resolve_layers(tiny_model, [1, bad])

    @pytest.mark.parametrize("bad", [1.5, "2", True, None])
    def test_non_integers_are_rejected(self, tiny_model, bad):
        with pytest.raises(h.LayerIndexError, match="integers"):
            h.resolve_layers(tiny_model, [bad])

    def test_empty_request(self, tiny_model):
        with pytest.raises(h.LayerIndexError, match="no layers"):
            h.resolve_layers(tiny_model, [])

    def test_negative_indexes_count_from_the_end_and_duplicates_collapse(self, tiny_model):
        assert h.resolve_layers(tiny_model, [-1, 3, 0, -4]) == [0, 3]
        assert h.resolve_layers(tiny_model, 2) == [2]

    def test_capture_rejects_bad_layers_too(self, tiny_model):
        with pytest.raises(h.LayerIndexError), h.capture_layers(tiny_model, [7]):
            pass

    def test_layer_index_error_is_also_a_value_error(self):
        assert issubclass(h.LayerIndexError, ValueError)


class TestArchitectureDiscovery:
    def test_finds_qwen_layers(self, tiny_model):
        assert h.num_layers(tiny_model) == 4
        assert h.decoder_layers(tiny_model) is tiny_model.model.layers

    def test_bare_stack(self, tuple_stack):
        assert h.num_layers(tuple_stack) == 3

    def test_unknown_architecture(self):
        with pytest.raises(TypeError, match="layers_path"):
            h.decoder_layers(torch.nn.Linear(2, 2))

    def test_output_helpers(self):
        t = torch.ones(1, 2, 3)
        assert h.layer_output_tensor(t) is t
        assert h.layer_output_tensor((t, "cache")) is t
        new = torch.zeros(1, 2, 3)
        assert h.replace_layer_output(t, new) is new
        assert h.replace_layer_output((t, "cache"), new) == (new, "cache")
