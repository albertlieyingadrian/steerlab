"""The sampler and the hooks, on toy torch models (skipped when torch or transformers is missing)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deslop.sampler import antislop_generate  # noqa: E402
from deslop.steer import Direction, NeuronEdit, bake_neuron_edits, load_direction, neuron_edits, steering  # noqa: E402

# A bigram "language model": the next token depends only on the last one. Greedy text: "We delve in."
VOCAB = ["<eos>", "We", " delve", " dig", " in", "."]
NEXT = {1: [2, 3], 2: [4], 3: [4], 4: [5], 5: [0]}  # preferred next tokens, best first


class Tok:
    eos_token_id = 0

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and i == 0))


class Bigram(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=0)

    def forward(self, ids: Any, past_key_values: Any = None, use_cache: bool = True) -> Any:
        logits = torch.full((1, ids.shape[1], len(VOCAB)), -10.0)
        for rank, token in enumerate(NEXT[int(ids[0, -1])]):
            logits[0, -1, token] = 5.0 - rank
        return SimpleNamespace(logits=logits)


def test_sampler_backtracks_from_a_slop_word() -> None:
    model, tok = Bigram(), Tok()
    plain = antislop_generate(model, tok, prompt_ids=[1], habits=(), temperature=0)
    assert plain.text == " delve in." and plain.backtracks == 0 and plain.finished
    out = antislop_generate(model, tok, prompt_ids=[1], temperature=0)
    assert out.text == " dig in."
    assert out.backtracks == 1 and out.finished and not out.accepted


def test_sampler_keeps_span_when_budget_is_spent() -> None:
    out = antislop_generate(Bigram(), Tok(), prompt_ids=[1], temperature=0, max_backtracks=0)
    assert out.text == " delve in." and [s.text for s in out.accepted] == ["delve"]


def test_sampler_rejects_unknown_habit() -> None:
    with pytest.raises(KeyError):
        antislop_generate(Bigram(), Tok(), prompt_ids=[1], habits=("nope",))


class Layer(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, d: int, m: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(m, d, bias=False)
        self.up = torch.nn.Linear(d, m, bias=False)

    def forward(self, x: Any) -> Any:
        return x + self.mlp.down_proj(torch.relu(self.up(x)))


class Toy(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, d: int = 4, m: int = 6, n: int = 2) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer(d, m) for _ in range(n)])

    def forward(self, x: Any) -> Any:
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_steering_adds_the_scaled_direction_and_removes_the_hook() -> None:
    import numpy as np

    model, x = Toy(), torch.ones(1, 3, 4)
    base = model(x)
    unit = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with steering(model, Direction(unit=unit, layer=1, class_gap=2.0), strength=-0.5):
        steered = model(x)
    assert torch.allclose(steered - base, torch.tensor([-1.0, 0.0, 0.0, 0.0]).expand(1, 3, 4), atol=1e-6)
    assert torch.allclose(model(x), base)


def test_neuron_edit_hook_matches_baked_weights() -> None:
    model, x = Toy(), torch.randn(1, 3, 4)
    edits = [NeuronEdit(0, 2, -1.0), NeuronEdit(1, 5, 0.0)]
    with neuron_edits(model, edits):
        hooked = model(x)
    assert not torch.allclose(hooked, model(x))
    bake_neuron_edits(model, edits)
    assert torch.allclose(model(x), hooked, atol=1e-6)


def test_neuron_edit_parse() -> None:
    assert NeuronEdit.parse("L32.N4852") == NeuronEdit(32, 4852, -1.0)
    assert NeuronEdit.parse("l3.n7*0") == NeuronEdit(3, 7, 0.0)


def test_load_direction_reads_steerlab_format(tmp_path: Any) -> None:
    import json

    import numpy as np

    np.savez(tmp_path / "reg.npz", vector=np.array([3.0, 4.0]))
    meta = {"metadata": {"layer": 7, "class_gap": 2.5, "concept": "register"}}
    (tmp_path / "reg.json").write_text(json.dumps(meta))
    d = load_direction(tmp_path / "reg")
    assert d.layer == 7 and d.class_gap == 2.5 and d.label == "register"
    assert np.allclose(d.unit, [0.6, 0.8])
