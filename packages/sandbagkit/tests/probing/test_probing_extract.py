"""Extraction: batching and padding change nothing, positions pick the right tokens, the cache is exact."""

import numpy as np
import pytest
import torch

from sandbagkit.core.types import ModelInfo
from sandbagkit.probing import extract as ex
from sandbagkit.probing import hooks as h

ATOL = 1e-5  # batched matmuls may round differently in the last bits; nothing else may differ


def _hidden_states(model, ids):
    """Every layer's output for ONE unpadded text, via output_hidden_states (no hooks involved)."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    return [state[0] for state in out.hidden_states[1:]]  # drop the embeddings: entry i = layer i


class TestBatchingAndPadding:
    @pytest.mark.parametrize("batch_size", [2, 3, 8])
    def test_padding_does_not_change_last_token_vectors(self, tiny_model, tokenizer, texts, batch_size):
        one_by_one = ex.extract_activations(tiny_model, tokenizer, texts, [0, 2, 3], batch_size=1)
        batched = ex.extract_activations(tiny_model, tokenizer, texts, [0, 2, 3], batch_size=batch_size)
        lengths = {len(tokenizer(t)["input_ids"]) for t in texts}
        assert len(lengths) > 1, "the texts must differ in length, or no padding happens"
        for layer in (0, 2, 3):
            np.testing.assert_allclose(batched[layer], one_by_one[layer], atol=ATOL)

    def test_tokenizer_padding_side_is_irrelevant(self, tiny_model, tokenizer, texts):
        right = ex.extract_activations(tiny_model, tokenizer, texts, [1], batch_size=5)
        original = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            left = ex.extract_activations(tiny_model, tokenizer, texts, [1], batch_size=5)
        finally:
            tokenizer.padding_side = original
        np.testing.assert_array_equal(left[1], right[1])

    def test_last_token_matches_output_hidden_states(self, tiny_model, tokenizer, texts):
        acts = ex.extract_activations(tiny_model, tokenizer, texts, [1, 2], batch_size=4)
        for i, text in enumerate(texts):
            states = _hidden_states(tiny_model, tokenizer(text)["input_ids"])
            for layer in (1, 2):
                np.testing.assert_allclose(acts[layer][i], states[layer][-1].numpy(), atol=ATOL)

    def test_shapes_and_dtype(self, tiny_model, tokenizer, texts):
        acts = ex.extract_activations(tiny_model, tokenizer, texts, [3, 1])
        assert sorted(acts) == [1, 3]
        assert acts[1].shape == (len(texts), 32)
        assert acts[1].dtype == np.float32

    def test_deterministic(self, tiny_model, tokenizer, texts):
        a = ex.extract_activations(tiny_model, tokenizer, texts, [2])
        b = ex.extract_activations(tiny_model, tokenizer, texts, [2])
        np.testing.assert_array_equal(a[2], b[2])

    def test_hooks_are_removed_afterwards(self, tiny_model, tokenizer, texts):
        before = h.count_forward_hooks(tiny_model)
        ex.extract_activations(tiny_model, tokenizer, texts, [0, 1, 2, 3])
        assert h.count_forward_hooks(tiny_model) == before

    def test_bad_layer(self, tiny_model, tokenizer, texts):
        with pytest.raises(h.LayerIndexError, match="4 layers"):
            ex.extract_activations(tiny_model, tokenizer, texts, [2, 4])


class TestPositions:
    def test_first_and_mean(self, tiny_model, tokenizer, texts):
        first = ex.extract_activations(tiny_model, tokenizer, texts, [2], position="first", batch_size=5)
        mean = ex.extract_activations(tiny_model, tokenizer, texts, [2], position="mean", batch_size=5)
        for i, text in enumerate(texts):
            state = _hidden_states(tiny_model, tokenizer(text)["input_ids"])[2]
            np.testing.assert_allclose(first[2][i], state[0].numpy(), atol=ATOL)
            np.testing.assert_allclose(mean[2][i], state.mean(dim=0).numpy(), atol=ATOL)

    def test_response_tokens(self, tiny_model, tokenizer):
        prompts = ["what is two plus two", "the capital of france is"]
        responses = ["four", "paris i think it is paris"]
        acts = ex.extract_activations(
            tiny_model, tokenizer, prompts, [1], responses=responses, position="response_mean", chat_template=True
        )
        firsts = ex.extract_activations(
            tiny_model, tokenizer, prompts, [1], responses=responses, position="response_first", chat_template=True
        )
        for i, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            encoded = ex.encode(tokenizer, prompt, response=response, chat_template=True)
            n_response = len(tokenizer(response, add_special_tokens=False)["input_ids"])
            assert len(encoded) - encoded.response_start == n_response
            state = _hidden_states(tiny_model, list(encoded.input_ids))[1]
            expected = state[encoded.response_start :].mean(dim=0)
            np.testing.assert_allclose(acts[1][i], expected.numpy(), atol=ATOL)
            np.testing.assert_allclose(firsts[1][i], state[encoded.response_start].numpy(), atol=ATOL)

    def test_response_positions_need_responses(self, tiny_model, tokenizer, texts):
        with pytest.raises(ValueError, match="needs responses"):
            ex.extract_activations(tiny_model, tokenizer, texts, [1], position="response_mean")

    def test_unknown_position(self, tiny_model, tokenizer, texts):
        with pytest.raises(ValueError, match="unknown position"):
            ex.extract_activations(tiny_model, tokenizer, texts, [1], position="middle")

    def test_response_count_must_match(self, tiny_model, tokenizer, texts):
        with pytest.raises(ValueError, match="responses"):
            ex.extract_activations(tiny_model, tokenizer, texts, [1], responses=["a"])

    def test_truncation_that_removes_the_response_is_an_error(self, tokenizer):
        with pytest.raises(ValueError, match="no tokens left"):
            ex.encode(tokenizer, "the capital of france is", response="paris", max_length=3)


class TestChatTemplate:
    def test_render(self, tokenizer):
        raw = ex.render_prompt(tokenizer, "hello world")
        chat = ex.render_prompt(tokenizer, "hello world", chat_template=True)
        assert raw == "hello world"
        assert chat == "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n"

    def test_system_prompt(self, tokenizer):
        chat = ex.render_prompt(tokenizer, "hello", chat_template=True, system_prompt="you are a helpful assistant")
        assert chat.startswith("<|im_start|>system\nyou are a helpful assistant<|im_end|>\n<|im_start|>user")
        with pytest.raises(ValueError, match="chat_template=True"):
            ex.render_prompt(tokenizer, "hello", system_prompt="x")

    def test_template_changes_the_vectors(self, tiny_model, tokenizer, texts):
        raw = ex.extract_activations(tiny_model, tokenizer, texts, [2])
        chat = ex.extract_activations(tiny_model, tokenizer, texts, [2], chat_template=True)
        assert not np.allclose(raw[2], chat[2])

    def test_missing_template(self, tokenizer):
        original = tokenizer.chat_template
        tokenizer.chat_template = None
        try:
            with pytest.raises(ValueError, match="no chat template"):
                ex.render_prompt(tokenizer, "hi", chat_template=True)
        finally:
            tokenizer.chat_template = original


class TestNonFinite:
    def test_nan_activations_raise(self, tiny_model, tokenizer, texts):
        def poison(module, inputs, output):
            return h.layer_output_tensor(output) * float("nan")

        with (
            h.register_hooks(tiny_model, {0: poison}),
            pytest.raises(ex.NonFiniteActivationError, match="NaN or inf in layer"),
        ):
            ex.extract_activations(tiny_model, tokenizer, texts, [2])


class TestCache:
    def test_second_run_is_all_hits_and_identical(self, tiny_model, tokenizer, texts, tmp_path):
        cache = ex.ActivationCache(tmp_path / "acts")
        first = ex.extract_activations(tiny_model, tokenizer, texts, [1, 3], cache=cache, model_id="tiny")
        # one lookup per (text, layer): 5 texts × 2 layers
        assert (cache.hits, cache.misses, len(cache)) == (0, 2 * len(texts), 2 * len(texts))

        again = ex.ActivationCache(tmp_path / "acts")
        second = ex.extract_activations(tiny_model, tokenizer, texts, [1, 3], cache=again, model_id="tiny")
        assert (again.hits, again.misses) == (2 * len(texts), 0)
        for layer in (1, 3):
            np.testing.assert_array_equal(first[layer], second[layer])

    def test_partial_hits_fill_in_the_rest(self, tiny_model, tokenizer, texts, tmp_path):
        cache = ex.ActivationCache(tmp_path)
        ex.extract_activations(tiny_model, tokenizer, texts[:2], [1], cache=cache, model_id="tiny")
        mixed = ex.extract_activations(tiny_model, tokenizer, texts, [1], cache=cache, model_id="tiny")
        fresh = ex.extract_activations(tiny_model, tokenizer, texts, [1])
        np.testing.assert_allclose(mixed[1], fresh[1], atol=ATOL)
        assert len(cache) == len(texts)

    def test_key_covers_model_text_layer_and_position(self, tokenizer):
        enc = ex.encode(tokenizer, "hello world")
        base = ex.ActivationCache.key("m", enc, 1, "last")
        assert base == ex.ActivationCache.key("m", ex.encode(tokenizer, "hello world"), 1, "last")
        assert base != ex.ActivationCache.key("other", enc, 1, "last")
        assert base != ex.ActivationCache.key("m", ex.encode(tokenizer, "hello"), 1, "last")
        assert base != ex.ActivationCache.key("m", enc, 2, "last")
        assert base != ex.ActivationCache.key("m", enc, 1, "first")
        templated = ex.encode(tokenizer, "hello world", chat_template=True)
        assert base != ex.ActivationCache.key("m", templated, 1, "last")
        with_response = ex.encode(tokenizer, "hello", response="world")
        assert with_response.input_ids == enc.input_ids
        assert base != ex.ActivationCache.key("m", with_response, 1, "last")  # same ids, different response span

    def test_model_without_a_name_needs_a_model_id(self, tiny_model, tokenizer, texts, tmp_path):
        assert ex.model_id_for(tiny_model) == ""
        assert ex.model_revision(tiny_model) is None
        with pytest.raises(ValueError, match="model_id"):
            ex.extract_activations(tiny_model, tokenizer, texts, [1], cache=ex.ActivationCache(tmp_path))


class TestModelInfo:
    def test_model_info_is_the_core_type(self, tiny_model):
        info = ex.model_info(tiny_model)
        assert isinstance(info, ModelInfo)
        assert info.provider == "hf"
        assert info.dtype == "float32"
        assert info.supports_activations
        assert info.num_parameters == sum(p.numel() for p in tiny_model.parameters())

    def test_device_parity_on_cpu_is_exact(self, tiny_model, tokenizer, texts):
        gaps = ex.check_device_parity(tiny_model, tokenizer, texts, [1, 3], reference_device="cpu")
        assert gaps == {1: 0.0, 3: 0.0}


class _SpyTokenizer:
    """Wraps a tokenizer and records every keyword `apply_chat_template` was called with."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        return self._inner.apply_chat_template(messages, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestTemplateKwargs:
    """`template_kwargs` go to `apply_chat_template` unchanged (`{"enable_thinking": False}` for Qwen3)."""

    def test_render_prompt_forwards_template_kwargs(self, tokenizer):
        spy = _SpyTokenizer(tokenizer)
        ex.render_prompt(spy, "hello", chat_template=True, template_kwargs={"enable_thinking": False})
        assert spy.calls == [{"add_generation_prompt": True, "tokenize": False, "enable_thinking": False}]

    def test_default_render_is_unchanged(self, tokenizer):
        spy = _SpyTokenizer(tokenizer)
        text = ex.render_prompt(spy, "hello world", chat_template=True)
        assert spy.calls == [{"add_generation_prompt": True, "tokenize": False}]
        assert text == "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n"

    def test_template_kwargs_need_the_chat_template(self, tokenizer):
        with pytest.raises(ValueError, match="chat_template=True"):
            ex.render_prompt(tokenizer, "hello", template_kwargs={"enable_thinking": False})

    def test_extract_activations_forwards_template_kwargs(self, tiny_model, tokenizer, texts):
        spy = _SpyTokenizer(tokenizer)
        ex.extract_activations(
            tiny_model, spy, texts[:2], [1], chat_template=True, template_kwargs={"enable_thinking": False}
        )
        assert len(spy.calls) == 2
        assert all(call["enable_thinking"] is False for call in spy.calls)


class TestSkipFirstResponseToken:
    """`response_mean` can leave the first response token out (it is often a template or formatting token)."""

    PROMPTS = ("what is two plus two", "the capital of france is")
    RESPONSES = ("four i think", "paris i think it is paris")

    def test_mean_excludes_exactly_the_first_response_token(self, tiny_model, tokenizer):
        acts = ex.extract_activations(
            tiny_model,
            tokenizer,
            self.PROMPTS,
            [1],
            responses=self.RESPONSES,
            position="response_mean",
            chat_template=True,
            skip_first_response_token=True,
        )
        plain = ex.extract_activations(
            tiny_model,
            tokenizer,
            self.PROMPTS,
            [1],
            responses=self.RESPONSES,
            position="response_mean",
            chat_template=True,
        )
        for i, (prompt, response) in enumerate(zip(self.PROMPTS, self.RESPONSES, strict=True)):
            encoded = ex.encode(tokenizer, prompt, response=response, chat_template=True)
            state = _hidden_states(tiny_model, list(encoded.input_ids))[1]
            expected = state[encoded.response_start + 1 :].mean(dim=0)
            np.testing.assert_allclose(acts[1][i], expected.numpy(), atol=ATOL)
            assert not np.allclose(acts[1][i], plain[1][i])

    def test_skipping_is_part_of_the_cache_key(self, tiny_model, tokenizer, tmp_path):
        cache = ex.ActivationCache(tmp_path)
        common = dict(responses=self.RESPONSES, position="response_mean", chat_template=True, model_id="tiny")
        ex.extract_activations(tiny_model, tokenizer, self.PROMPTS, [1], cache=cache, **common)
        ex.extract_activations(
            tiny_model, tokenizer, self.PROMPTS, [1], cache=cache, skip_first_response_token=True, **common
        )
        assert (cache.hits, cache.misses) == (0, 4)

    def test_a_one_token_response_cannot_be_skipped(self, tiny_model, tokenizer):
        with pytest.raises(ValueError, match="skip_first_response_token"):
            ex.extract_activations(
                tiny_model,
                tokenizer,
                ["hello"],
                [1],
                responses=["world"],
                position="response_mean",
                chat_template=True,
                skip_first_response_token=True,
            )

    def test_skipping_only_applies_to_response_mean(self, tiny_model, tokenizer, texts):
        with pytest.raises(ValueError, match="skip_first_response_token"):
            ex.extract_activations(tiny_model, tokenizer, texts, [1], position="last", skip_first_response_token=True)
