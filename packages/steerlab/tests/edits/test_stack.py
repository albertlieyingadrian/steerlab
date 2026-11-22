"""The shared edit list, its artifact JSON, the checks and the before / after / controls comparison."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from steerlab.edits.checks import AxisProbe, choice_margin, compare, generate, word_counts  # noqa: E402
from steerlab.edits.reflection import Editor, Units, pair_axes, select_top  # noqa: E402
from steerlab.edits.stack import EditRecord, EditStack, from_inspect_edit, load_artifact  # noqa: E402
from steerlab.inspect.core import Edit as InspectEdit  # noqa: E402
from steerlab.inspect.core import Inspector  # noqa: E402

from .conftest import build  # noqa: E402

IDS = [5, 17, 40, 22, 9, 31]


def logits(model: Any) -> Any:
    with torch.no_grad():
        return model(torch.tensor([IDS]), use_cache=False).logits


@pytest.fixture
def qwen(tokenizer: Any) -> Any:
    model = build("qwen2", len(tokenizer))
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def test_word_counts() -> None:
    c = word_counts("My mother and my Mother; motherhood. father", ["mother"], ["father", "dad"])
    assert c == {"source": 2, "target": 1, "score": pytest.approx(1 / 3)}
    assert word_counts("nothing here", ["a b"], ["c"])["score"] == 0.0


def test_generate_is_greedy_and_seeded(qwen: Any, tokenizer: Any) -> None:
    a = generate(qwen, tokenizer, "hi", max_new_tokens=6)
    assert a == generate(qwen, tokenizer, "hi", max_new_tokens=6)
    s1 = generate(qwen, tokenizer, "hi", max_new_tokens=6, sample=True, seed=3)
    assert s1 == generate(qwen, tokenizer, "hi", max_new_tokens=6, sample=True, seed=3)


def test_choice_margin_is_antisymmetric(qwen: Any, tokenizer: Any) -> None:
    m = choice_margin(qwen, tokenizer, "pick", "ab", "c")
    assert m == pytest.approx(-choice_margin(qwen, tokenizer, "pick", "c", "ab"), abs=1e-5)


def test_neuron_records_match_inspect_edits(tokenizer: Any) -> None:
    """The stack's neuron hook does what the Trace room's Edit does, mode for mode."""
    model = build("qwen2", len(tokenizer))
    inspector = Inspector(model, tokenizer, name="tiny")
    for mode, value in (("off", 0.0), ("flip", 0.0), ("scale", 3.0), ("set", 1.5)):
        edit = InspectEdit(1, 7, mode, value)  # type: ignore[arg-type]
        with inspector.edited([edit]):
            expected = logits(model)
        stack = EditStack()
        stack.add(from_inspect_edit(edit))
        with stack.applied(model, tokenizer):
            assert torch.equal(logits(model), expected)
        assert stack.neuron_edits() == [(1, 7, mode, value)]


def test_stack_applies_and_restores_bit_identically(qwen: Any, tokenizer: Any) -> None:
    before = logits(qwen)
    editor = Editor(qwen)
    axes = pair_axes(qwen, tokenizer, [("cat", "dog")])
    stack = EditStack()
    stack.add_reflection(axes, select_top(editor, axes, per_layer=6), k=2.0)
    stack.add_neuron(0, 3, "scale", 4.0)
    with stack.applied(qwen, tokenizer, editor):
        edited = logits(qwen)
        assert not torch.equal(edited, before)
    assert torch.equal(logits(qwen), before)
    stack.records[0].enabled = False
    with stack.applied(qwen, tokenizer):
        only_neuron = logits(qwen)
    assert not torch.equal(only_neuron, edited)
    assert torch.equal(logits(qwen), before)


def test_artifact_round_trip_rebuilds_the_same_edit(qwen: Any, tokenizer: Any) -> None:
    editor = Editor(qwen)
    axes = pair_axes(qwen, tokenizer, [("cat", "dog"), ("sun", "moon")])
    units = select_top(editor, axes, per_layer=4)
    stack = EditStack()
    record = stack.add_reflection(axes, units, k=-1.0, sides="reads", scope={"selection": "top", "per_layer": 4})
    with stack.applied(qwen, tokenizer):
        expected = logits(qwen)
    art = record.to_artifact("tiny")
    assert art["type"] == "edit" and art["model"] == "tiny" and art["k"] == -1.0
    assert set(art) >= {"name", "axis_source", "units", "k", "mode", "created", "sides", "parts", "kind"}
    assert isinstance(art["created"], float) and art["mode"] == "amplify"
    assert art["units"][0] == {"kind": "neuron", "layer": 0, "index": units.neurons[0][0]}
    again = EditStack()
    again.add(EditRecord.from_artifact(art))  # axes rebuilt from the source, not carried
    with again.applied(qwen, tokenizer):
        assert torch.allclose(logits(qwen), expected, atol=1e-6)
    with pytest.raises(ValueError, match="type"):
        EditRecord.from_artifact({"type": "probe"})
    with pytest.raises(ValueError, match="mode"):
        EditRecord("neuron", units=Units(neurons={0: [1]}), mode="melt")


def test_compare_runs_four_columns_and_restores(qwen: Any, tokenizer: Any) -> None:
    editor = Editor(qwen)
    axes = pair_axes(qwen, tokenizer, [("cat", "dog")])
    units = select_top(editor, axes, per_layer=8)
    stack = EditStack()
    stack.add_neuron(1, 2, "off")
    before = logits(qwen)
    checks = [
        {"kind": "words", "source": ["cat"], "target": ["dog"]},
        {"kind": "choice", "question": "x", "a": "a", "b": "b"},
        {"kind": "probe"},
    ]
    result = compare(
        editor,
        tokenizer,
        axes,
        units,
        prompts=["hi", "a cat"],
        checks=checks,
        max_new_tokens=5,
        base=stack.base(qwen, tokenizer, editor),
    )
    assert result["order"] == ["before", "after", "random_axis", "random_units"]
    assert result["restored"] is True
    assert torch.equal(logits(qwen), before)
    for name in result["order"]:
        col = result["columns"][name]
        assert len(col["replies"]) == 2
        assert [c["kind"] for c in col["checks"]] == ["words", "choice", "probe"]
    probe_before = result["columns"]["before"]["checks"][2]["value"]
    assert probe_before == pytest.approx(1.0, abs=1e-5)  # the probe is fitted on the base model
    assert result["columns"]["after"]["info"]["neurons"] == units.count()["neurons"]
    assert result["columns"]["random_units"]["info"]["neurons"] == units.count()["neurons"]


def test_axis_probe_moves_under_the_edit(tokenizer: Any) -> None:
    """kept is 1 on the model it was fitted on and moves when the axis is mirrored on every neuron and head. (In a
    random model most of the cat/dog difference is the embedding itself, which no unit edit touches, so the size of
    the move says nothing here; the slow Qwen3 test checks the direction.)"""
    model = build("llama", len(tokenizer))
    editor = Editor(model)
    axes = pair_axes(model, tokenizer, [("cat", "dog")])
    probe = AxisProbe.fit(model, tokenizer, axes, layer=1)
    assert probe is not None
    everything = Units(
        neurons={i: list(range(editor.arch.d_mlp(i))) for i in range(2)},
        heads={i: list(range(editor.arch.n_heads)) for i in range(2)},
    )
    assert probe.kept(model, tokenizer) == pytest.approx(1.0, abs=1e-6)
    with editor.applied(axes, everything, k=2.0):
        assert abs(probe.kept(model, tokenizer) - 1.0) > 1e-3
    assert probe.kept(model, tokenizer) == pytest.approx(1.0, abs=1e-6)


def test_artifact_with_axes_needs_no_tokenizer(qwen: Any, tokenizer: Any) -> None:
    """stack.artifact carries the axes as npz arrays; load_artifact gives the same edit back without rebuilding."""
    editor = Editor(qwen)
    axes = pair_axes(qwen, tokenizer, [("cat", "dog")])
    stack = EditStack()
    record = stack.add_reflection(axes, select_top(editor, axes, per_layer=3), k=2.0, parts="qo")
    with stack.applied(qwen, tokenizer):
        expected = logits(qwen)
    meta, arrays = stack.artifact(record, "tiny", "cat-dog")
    assert meta["name"] == "cat-dog" and meta["axis_layers"] == [0, 1] and arrays["reads"].shape == (2, 32)
    again, carried = load_artifact(meta, arrays, d_model=32)
    assert carried is not None and torch.equal(carried.writes[1], axes.writes[1])
    other = EditStack()
    other.add(again, carried)
    with other.applied(qwen, None):  # no tokenizer: the axes came with the artifact
        assert torch.equal(logits(qwen), expected)
    with pytest.raises(ValueError, match="wide"):
        load_artifact(meta, arrays, d_model=64)


def test_the_first_workbench_shape_still_loads(qwen: Any) -> None:
    """One `axis` and `mode`: the units' writes go through H (down_proj columns, o_proj head blocks), nothing else."""
    weight = qwen.model.layers[1].mlp.down_proj.weight
    o = qwen.model.layers[0].self_attn.o_proj.weight
    gate = qwen.model.layers[1].mlp.gate_proj.weight
    before, gate_before = weight.detach().clone(), gate.detach().clone()
    axis = torch.randn(32)
    meta = {
        "type": "edit",
        "name": "old",
        "mode": "ablate",
        "axis": axis.tolist(),
        "units": [{"kind": "neuron", "layer": 1, "index": 2}, {"kind": "head", "layer": 0, "index": 1}],
    }
    record, axes = load_artifact(meta, d_model=32)
    assert record.sides == "writes" and record.parts == "o" and record.k == 1.0
    stack = EditStack()
    stack.add(record, axes)
    d = axis / axis.norm()
    with stack.applied(qwen, None):
        assert torch.allclose(d @ weight[:, 2], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(d @ o[:, 8:16], torch.zeros(8), atol=1e-5)
        assert torch.equal(weight[:, 3], before[:, 3]) and torch.equal(gate, gate_before)
    assert torch.equal(weight, before)
    with pytest.raises(ValueError, match="wide"):
        load_artifact({**meta, "axis": [1.0]}, d_model=32)
