"""Fixtures for the `steerlab.inspect` tests: an `Inspector` around sandbagkit's tiny random Qwen2 model.

Two layers, 32-wide stream, 4 heads, 64 neurons, a character tokenizer: no download, milliseconds per view. Its words
are random, so the tests check the plumbing (sums, hooks, shapes) and never what a neuron means.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from sandbagkit.backends.tiny import char_tokenizer, tiny_model  # noqa: E402

from steerlab.inspect.core import Inspector  # noqa: E402

PROMPT = "hello wor"


@pytest.fixture(scope="session")
def inspector() -> Inspector:
    tokenizer = char_tokenizer()
    model = tiny_model(len(tokenizer), seed=0)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return Inspector(model, tokenizer, name="tiny-random-qwen2")
