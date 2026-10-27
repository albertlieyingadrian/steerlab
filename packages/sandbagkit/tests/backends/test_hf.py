"""HFBackend on a tiny random model: offline, CPU, a few milliseconds per test."""

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")  # needs the [probing] extra; runs in CI's probing job
import torch

from sandbagkit.backends._common import truncate_at_stop
from sandbagkit.backends.hf import HFBackend, check_dtype, estimate_parameters, pick_dtype
from sandbagkit.backends.tiny import char_tokenizer, tiny_backend, tiny_model
from sandbagkit.core.backend import GenerationConfig, ModelBackend


@pytest.fixture(scope="module")
def backend() -> HFBackend:
    return tiny_backend(seed=0)


def test_satisfies_the_protocol(backend: HFBackend) -> None:
    assert isinstance(backend, ModelBackend)


def test_greedy_generation_is_deterministic(backend: HFBackend) -> None:
    config = GenerationConfig(max_new_tokens=12)
    first = backend.generate("What is 2 + 2?", config=config)
    second = backend.generate("What is 2 + 2?", config=config)
    assert first.text == second.text
    assert first.metadata["greedy"] is True
    assert first.usage["output_tokens"] <= 12
    assert first.finish_reason in ("stop", "length")


def test_sampling_with_a_fixed_seed_repeats(backend: HFBackend) -> None:
    config = GenerationConfig(max_new_tokens=12, temperature=1.0, seed=7)
    assert backend.generate("hello", config=config).text == backend.generate("hello", config=config).text


def test_chat_template_wraps_system_and_user(backend: HFBackend) -> None:
    text = backend.build_prompt("Q?", system_prompt="Be brief.")
    assert text.startswith("<|im_start|>system\nBe brief.<|im_end|>")
    assert text.endswith("<|im_start|>assistant\n")


def test_plain_prompts_without_a_chat_template() -> None:
    plain = tiny_backend(chat_template=False)
    assert plain.use_chat_template is False
    assert plain.build_prompt("Q?", system_prompt="S") == "S\n\nQ?"


def test_asking_for_a_missing_chat_template_fails() -> None:
    tokenizer = char_tokenizer(chat_template=False)
    with pytest.raises(ValueError, match="no chat template"):
        HFBackend(tiny_model(len(tokenizer)), tokenizer, use_chat_template=True)


def test_generate_with_cot_puts_the_reasoning_in_the_reply(backend: HFBackend) -> None:
    generation = backend.generate_with_cot("Q?", "Step one.", config=GenerationConfig(max_new_tokens=4))
    assert generation.cot == "Step one."
    prompt = backend.build_prompt("Q?", assistant_prefix="Step one." + backend.cot_answer_prefix)
    assert prompt.endswith("<|im_start|>assistant\nStep one.\n\nTherefore, the final answer is")


def test_choice_logprobs_form_a_distribution(backend: HFBackend) -> None:
    scores = backend.choice_logprobs("Pick one.\nA. x\nB. y\nC. z\nD. w\nAnswer:", 4)
    assert list(scores) == ["A", "B", "C", "D"]
    assert math.isclose(sum(math.exp(s) for s in scores.values()), 1.0, rel_tol=1e-9)
    assert 0 <= backend.choose("Pick one. Answer:", 4) < 4


def test_choice_logprobs_reject_impossible_option_counts(backend: HFBackend) -> None:
    with pytest.raises(ValueError):
        backend.choice_logprobs("x", 1)


def test_model_info(backend: HFBackend) -> None:
    info = backend.model_info()
    assert info.provider == "hf"
    assert info.dtype == "float32"
    assert info.device == "cpu"
    assert info.supports_activations is True
    assert info.num_parameters == sum(p.numel() for p in backend.model.parameters())


def test_dtype_policy() -> None:
    assert pick_dtype(0.5e9) == torch.float32
    assert pick_dtype(1.5e9) == torch.float32
    assert pick_dtype(7e9) == torch.bfloat16
    with pytest.raises(ValueError, match="float16"):
        check_dtype(torch.float16)
    assert check_dtype(torch.bfloat16) == torch.bfloat16


def test_parameter_estimate_is_in_the_right_ballpark() -> None:
    qwen_05b = SimpleNamespace(hidden_size=896, num_hidden_layers=24, vocab_size=151936)
    mistral_7b = SimpleNamespace(hidden_size=4096, num_hidden_layers=32, vocab_size=32000)
    assert 0.3e9 < estimate_parameters(qwen_05b) < 0.6e9
    assert 6e9 < estimate_parameters(mistral_7b) < 8e9


def test_stop_strings_truncate() -> None:
    assert truncate_at_stop("A. done\nQ: next", ["\nQ:"]) == ("A. done", True)
    assert truncate_at_stop("no stop here", ["###", ""]) == ("no stop here", False)


def test_importing_backends_does_not_import_torch() -> None:
    code = "import sys, sandbagkit.backends; print('torch' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


@pytest.mark.parametrize(
    ("variant", "model_type"), [("qwen2", "qwen2"), ("qwen3", "qwen3"), ("gemma3_text", "gemma3_text")]
)
def test_tiny_model_variants_build_the_named_architecture(variant: str, model_type: str) -> None:
    other = tiny_backend(seed=0, variant=variant)
    assert other.model.config.model_type == model_type
    assert other.name == f"tiny-random-{variant}"
    assert other.generate("hi", config=GenerationConfig(max_new_tokens=3)).usage["output_tokens"] <= 3


def test_default_tiny_model_is_still_the_qwen2_one() -> None:
    default = tiny_backend(seed=0)
    assert default.model.config.model_type == "qwen2" and default.name == "tiny-random-qwen2"
    explicit = tiny_backend(seed=0, variant="qwen2")
    assert all(torch.equal(a, b) for a, b in zip(default.model.parameters(), explicit.model.parameters(), strict=True))
    with pytest.raises(ValueError, match="variant"):
        tiny_model(10, variant="llama")


# --- template_kwargs (Qwen3's thinking block is a template flag) ------------------------------------------------

# Qwen3's template, reduced to the one thing that matters here: with `enable_thinking=False` the assistant turn
# starts with an EMPTY, already-closed think block, so the model answers at once instead of reasoning first.
THINKING_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


@pytest.fixture
def thinking_backend() -> HFBackend:
    tokenizer = char_tokenizer()
    tokenizer.chat_template = THINKING_TEMPLATE
    return HFBackend(tiny_model(len(tokenizer)), tokenizer, name="tiny-thinking")


def test_template_kwargs_reach_the_chat_template(thinking_backend: HFBackend) -> None:
    off = thinking_backend.build_prompt("Q?", template_kwargs={"enable_thinking": False})
    assert off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert off.startswith("<|im_start|>user\nQ?<|im_end|>\n")


def test_default_build_prompt_is_byte_identical_to_before(thinking_backend: HFBackend) -> None:
    default = thinking_backend.build_prompt("Q?", system_prompt="S", assistant_prefix="A:")
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}]
    before = thinking_backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert default == before + "A:"
    assert "<think>" not in default  # the flag off the table, the template leaves the thinking block out entirely
    assert thinking_backend.build_prompt("Q?", template_kwargs={}) == thinking_backend.build_prompt("Q?")
