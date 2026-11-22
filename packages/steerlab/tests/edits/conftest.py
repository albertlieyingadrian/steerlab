"""Tiny random decoders for the edit tests: Qwen2, Qwen3, Llama, Mistral (SwiGLU, pre-norm) and Gemma 2 / 3 (GeGLU,
with post-norms), each two layers wide 32 with GQA, random norm gains (so the gain folding is exercised) and a
character tokenizer. No downloads; a model builds in milliseconds."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from sandbagkit.backends.tiny import char_tokenizer  # noqa: E402

EXACT = ("qwen2", "qwen3", "llama", "mistral")
GEMMA = ("gemma2", "gemma3")
ARCHS = EXACT + GEMMA


def build(arch: str, vocab: int, seed: int = 0) -> Any:
    import transformers as tf

    shape: dict[str, Any] = dict(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    configs = {
        "qwen2": lambda: tf.Qwen2Config(**shape),
        "qwen3": lambda: tf.Qwen3Config(**shape, head_dim=8),
        "llama": lambda: tf.LlamaConfig(**shape),
        "mistral": lambda: tf.MistralConfig(**shape, sliding_window=None),
        "gemma2": lambda: tf.Gemma2Config(**shape, head_dim=8, sliding_window=64),
        "gemma3": lambda: tf.Gemma3TextConfig(**shape, head_dim=8, sliding_window=64),
    }
    torch.manual_seed(seed)
    config = configs[arch]()
    config._attn_implementation = "eager"
    model = tf.AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if type(module).__name__.endswith("RMSNorm") and getattr(module, "weight", None) is not None:
                if arch.startswith("gemma"):
                    module.weight.uniform_(-0.6, 0.8)  # gain is 1 + w
                else:
                    module.weight.uniform_(0.3, 1.8)
            if name.endswith("proj") and getattr(module, "bias", None) is not None:
                module.bias.normal_(0, 0.1)
    return model


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    return char_tokenizer()


@pytest.fixture(params=ARCHS)
def model(request: pytest.FixtureRequest, tokenizer: Any) -> Any:
    m = build(request.param, len(tokenizer))
    m.arch_name = request.param
    return m


@pytest.fixture(params=EXACT)
def exact_model(request: pytest.FixtureRequest, tokenizer: Any) -> Any:
    m = build(request.param, len(tokenizer))
    m.arch_name = request.param
    return m
