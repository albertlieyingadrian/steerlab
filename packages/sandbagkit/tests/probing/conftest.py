"""Fixtures for the probing tests: a tiny, randomly initialized transformer and an offline tokenizer.

The model is a real Qwen2 architecture (the same code path as Qwen2.5-0.5B), just with 4 layers of width 32, so
every test runs on CPU in milliseconds with no download. The tokenizer is a word-level vocabulary built in
memory, with a Qwen-style chat template.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("sklearn")

WORDS = " ".join(
    [
        "the a an model is being evaluated deployed test answer question hello world this capability evaluation",
        "please solve what two plus four capital of france paris i think it you are helpful assistant short longer",
        "sentence with many more words in it than others",
    ]
).split()
SPECIAL = ["<unk>", "<pad>", "<eos>", "<|im_start|>", "<|im_end|>", "user", "assistant", "system"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
HIDDEN = 32
N_LAYERS = 4

TEXTS = [
    "hello world",
    "this is a capability evaluation please solve what is two plus two",
    "the capital of france is paris",
    "a longer sentence with many more words in it than others",
    "test",
]


def build_tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    config = transformers.AutoConfig.for_model(
        "qwen2",
        vocab_size=len(SPECIAL) + len(set(WORDS)) + 8,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return transformers.AutoModelForCausalLM.from_config(config).eval()


def build_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {token: i for i, token in enumerate(SPECIAL)}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny_model()


@pytest.fixture(scope="session")
def tokenizer():
    return build_tokenizer()


@pytest.fixture(scope="session")
def texts():
    return list(TEXTS)


class TupleLayer(torch.nn.Module):
    """A layer that returns (hidden, extra), like transformers v4 decoder layers."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN, HIDDEN)
        self.scale = scale

    def forward(self, x):
        return (torch.tanh(self.linear(x)) * self.scale, None)


class TupleStack(torch.nn.Module):
    """A bare decoder stack (found at `layers`) whose layers return tuples."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.layers = torch.nn.ModuleList([TupleLayer(1.0 + i) for i in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


@pytest.fixture
def tuple_stack():
    return TupleStack().eval()
