"""A tiny, randomly initialized model with a character-level tokenizer: no downloads, runs in milliseconds.

Tests and offline demos use it wherever real model behavior doesn't matter, only the plumbing does: hooks,
weight noise, restore, determinism, prompt building. Its outputs are random characters.

    backend = tiny_backend(seed=0)              # an HFBackend on CPU, float32, chat template included
    backend = tiny_backend(variant="qwen3")     # same size, Qwen3-shaped (q/k norms); also "gemma3_text"
"""

from __future__ import annotations

import string
from typing import Any

import torch

from sandbagkit.backends.hf import HFBackend

SPECIAL_TOKENS = ("<pad>", "<unk>", "<|im_start|>", "<|im_end|>")

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def char_tokenizer(*, chat_template: bool = True) -> Any:
    """A tokenizer whose tokens are single printable ASCII characters, plus four special tokens."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for char in string.printable:
        vocab.setdefault(char, len(vocab))
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    core.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call,unused-ignore]
        tokenizer_object=core,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    if chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


VARIANTS = ("qwen2", "qwen3", "gemma3_text")
"""Architectures `tiny_model` can build. The default, `qwen2`, is the original tiny model; the other two exist so
steering code can be checked against a Qwen3-shaped and a Gemma-3-shaped decoder stack without a download."""


def tiny_model(
    vocab_size: int, *, seed: int = 0, hidden_size: int = 32, layers: int = 2, variant: str = "qwen2"
) -> Any:
    """A 2-layer causal LM with random weights, on CPU in float32: Qwen2-style by default (see `VARIANTS`)."""
    from transformers import AutoModelForCausalLM, Gemma3TextConfig, Qwen2Config, Qwen3Config

    if variant not in VARIANTS:
        raise ValueError(f"unknown tiny model variant {variant!r}; choose one of {VARIANTS}")
    shape: dict[str, Any] = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    if variant == "qwen2":
        config: Any = Qwen2Config(**shape)
    elif variant == "qwen3":
        config = Qwen3Config(**shape, head_dim=hidden_size // 4)
    else:
        config = Gemma3TextConfig(**shape, head_dim=hidden_size // 4, sliding_window=64)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)  # type: ignore[no-untyped-call,unused-ignore]
    return model.eval()


def tiny_backend(
    *, seed: int = 0, chat_template: bool = True, name: str | None = None, variant: str = "qwen2"
) -> HFBackend:
    """`tiny_model` plus the character tokenizer as an HFBackend. The name defaults to `tiny-random-<variant>`."""
    tokenizer = char_tokenizer(chat_template=chat_template)
    model = tiny_model(len(tokenizer), seed=seed, variant=variant)
    name = name or f"tiny-random-{variant}"
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return HFBackend(model, tokenizer, name=name)
