"""The edit's identities on tiny random models (after neuron-reflection's tests/test_edit.py, MIT):

- k = 2 on every neuron: the MLP block computes H_out · MLP(norm(H_in x)), gain folding included;
- k = 1: the block neither reads the read axis nor writes the write axis;
- k < 0: the axis component of every gain-folded row is stretched by 1 − k;
- one shared axis on every head and neuron of every layer: the model is the original behind a mirrored interface;
- restore: logits bit-identical (torch.equal) after any edit, on every architecture, Gemma included.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from steerlab.edits.reflection import (  # noqa: E402
    Axes,
    Editor,
    Units,
    pair_axes,
    random_axes_like,
    random_units_like,
    select_top,
    sentence_axes,
    template_axes,
    unit,
)


def random_axes(d: int, layers: int, seed: int = 1, shared: bool = False) -> Axes:
    gen = torch.Generator().manual_seed(seed)
    if shared:
        a = unit(torch.randn(d, generator=gen))
        return Axes({i: a for i in range(layers)}, {i: a for i in range(layers)})
    return Axes(
        {i: unit(torch.randn(d, generator=gen)) for i in range(layers)},
        {i: unit(torch.randn(d, generator=gen)) for i in range(layers)},
    )


def all_neurons(editor: Editor) -> Units:
    return Units(neurons={i: list(range(editor.arch.d_mlp(i))) for i in range(editor.arch.n_layers)})


def all_heads(editor: Editor) -> Units:
    n = editor.arch.n_layers
    return Units(
        neurons={i: list(range(editor.arch.d_mlp(i))) for i in range(n)},
        heads={i: list(range(editor.arch.n_heads)) for i in range(n)},
    )


def block(editor: Editor, layer: int, x: Any) -> Any:
    lay = editor.arch.layers[layer]
    return lay.mlp(editor.arch.mlp_in_norm(layer)(x))


def mirror(a: Any, k: float = 2.0) -> Any:
    return torch.eye(a.shape[0]) - k * torch.outer(a, a)


def logits(model: Any, ids: list[int]) -> Any:
    with torch.no_grad():
        return model(torch.tensor([ids]), use_cache=False).logits


IDS = [5, 17, 40, 22, 9, 31, 12]


def test_reflection_conjugates_the_mlp(exact_model: Any) -> None:
    editor = Editor(exact_model)
    d = editor.arch.d_model
    axes = random_axes(d, 2)
    x = torch.randn(5, d)
    with torch.no_grad():
        before = [block(editor, i, x @ mirror(axes.reads[i])) for i in range(2)]
        editor.reflect(axes, all_neurons(editor), k=2.0)
        for i in range(2):
            assert torch.allclose(block(editor, i, x), before[i] @ mirror(axes.writes[i]), atol=1e-5)


def test_projection_removes_the_axis(exact_model: Any) -> None:
    editor = Editor(exact_model)
    d = editor.arch.d_model
    axes = random_axes(d, 2)
    x = torch.randn(5, d)
    with torch.no_grad():
        editor.reflect(axes, all_neurons(editor), k=1.0)
        for i in range(2):
            assert torch.allclose(block(editor, i, x) @ axes.writes[i], torch.zeros(5), atol=1e-5)
            gate, up, _ = editor.arch.mlp(i)
            g = editor._gain("mlp_in", i)
            for proj in (gate, up):
                assert torch.allclose((proj.weight * g) @ axes.reads[i], torch.zeros(proj.weight.shape[0]), atol=1e-5)


def test_amplify_stretches_the_axis_component(model: Any) -> None:
    editor = Editor(model)
    axes = random_axes(editor.arch.d_model, 2)
    gate = editor.arch.mlp(0)[0].weight
    g = editor._gain("mlp_in", 0)
    with torch.no_grad():
        before = ((gate * g) @ axes.reads[0]).clone()
        editor.reflect(axes, Units(neurons={0: [0, 3, 7]}), k=-2.0)
        after = (gate * g) @ axes.reads[0]
    assert torch.allclose(after[[0, 3, 7]], 3.0 * before[[0, 3, 7]], atol=1e-5)
    untouched = [i for i in range(gate.shape[0]) if i not in (0, 3, 7)]
    assert torch.equal(after[untouched], before[untouched])


def test_one_side_only(exact_model: Any) -> None:
    editor = Editor(exact_model)
    d = editor.arch.d_model
    axes = random_axes(d, 2)
    x = torch.randn(4, d)
    with torch.no_grad():
        plain = block(editor, 0, x)
        heard = block(editor, 0, x @ mirror(axes.reads[0]))
        with editor.applied(axes, all_neurons(editor), sides="reads"):
            assert torch.allclose(block(editor, 0, x), heard, atol=1e-5)
        with editor.applied(axes, all_neurons(editor), sides="writes"):
            assert torch.allclose(block(editor, 0, x), plain @ mirror(axes.writes[0]), atol=1e-5)


def test_one_shared_axis_everywhere_is_a_mirrored_interface(exact_model: Any) -> None:
    """Every head (q, k, v, o) and neuron of every layer mirrored along one axis a: fed the mirrored embeddings
    E H, the edited model's last residual is H times the original's."""
    editor = Editor(exact_model)
    d = editor.arch.d_model
    axes = random_axes(d, 2, shared=True)
    H = mirror(axes.reads[0])
    emb = exact_model.get_input_embeddings().weight.detach()[IDS][None]
    last = editor.arch.layers[-1]
    store: dict[str, Any] = {}

    def keep(module: Any, args: Any, out: Any) -> None:
        store["h"] = out[0] if isinstance(out, tuple) else out

    handle = last.register_forward_hook(keep)
    try:
        with torch.no_grad():
            exact_model(inputs_embeds=emb, use_cache=False)
            original = store["h"].clone()
            editor.reflect(axes, all_heads(editor), k=2.0)
            exact_model(inputs_embeds=emb @ H, use_cache=False)
            assert torch.allclose(store["h"], original @ H, atol=1e-4)
    finally:
        handle.remove()


def test_restore_is_bit_identical(model: Any) -> None:
    editor = Editor(model)
    before = logits(model, IDS)
    axes = random_axes(editor.arch.d_model, 2)
    editor.reflect(axes, all_heads(editor), k=2.0)
    editor.reflect(axes, Units(neurons={1: [1, 2, 3]}, heads={0: [1]}), k=-3.0, parts="qo")
    assert not torch.equal(logits(model, IDS), before)
    editor.restore()
    assert torch.equal(logits(model, IDS), before)
    assert not editor.edited


def test_k_zero_changes_nothing(model: Any) -> None:
    editor = Editor(model)
    before = logits(model, IDS)
    editor.reflect(random_axes(editor.arch.d_model, 2), all_heads(editor), k=0.0)
    assert torch.allclose(logits(model, IDS), before, atol=1e-6)


def test_gemma_is_flagged_inexact(model: Any) -> None:
    editor = Editor(model)
    assert editor.exact == (not model.arch_name.startswith("gemma"))


def test_gain_detection(model: Any) -> None:
    """Gemma 2/3 norms scale by 1 + w; the others by w."""
    editor = Editor(model)
    norm = editor.arch.mlp_in_norm(0)
    g = editor._gain("mlp_in", 0)
    w = norm.weight.detach().float()
    assert torch.allclose(g, 1 + w if model.arch_name.startswith("gemma") else w)


def test_units_parse_and_json() -> None:
    units = Units.parse(["L8.N344", "L8/n12+", "L8/n344-", "L3.H2", "L3/h0"])
    assert units.neurons == {8: [12, 344]}
    assert units.heads == {3: [0, 2]}
    assert Units.from_json(units.to_json()) == units
    assert units.count() == {"neurons": 2, "heads": 2}
    with pytest.raises(ValueError, match="not like"):
        Units.parse(["neuron 8"])


def test_bad_units_are_refused(model: Any) -> None:
    editor = Editor(model)
    axes = random_axes(editor.arch.d_model, 2)
    with pytest.raises(ValueError, match="outside"):
        editor.reflect(axes, Units(neurons={0: [10_000]}))
    with pytest.raises(ValueError, match="outside"):
        editor.reflect(axes, Units(heads={5: [0]}))
    with pytest.raises(ValueError, match="sides"):
        editor.reflect(axes, Units(neurons={0: [0]}), sides="middle")  # type: ignore[arg-type]
    assert not editor.edited


def test_select_top_and_controls(model: Any) -> None:
    editor = Editor(model)
    axes = random_axes(editor.arch.d_model, 2)
    units = select_top(editor, axes, per_layer=5)
    for layer, idx in units.neurons.items():
        load = editor.neuron_shares(layer, *axes.neuron_axes(layer)).max(0).values
        assert set(idx) == set(load.topk(5).indices.tolist())
    heads = select_top(editor, axes, per_layer=2, kind="heads", parts="q")
    assert all(len(v) == 2 for v in heads.heads.values())
    control = random_units_like(editor, units, seed=3)
    assert control.count() == units.count()
    for layer in units.neurons:
        assert not set(control.neurons[layer]) & set(units.neurons[layer])
    rand = random_axes_like(axes, seed=4)
    assert rand.layers() == axes.layers()
    assert torch.allclose(rand.reads[0].norm(), torch.tensor(1.0))
    assert torch.equal(random_axes_like(axes, seed=4).writes[1], rand.writes[1])
    assert rand.source["kind"] == "random"


def test_axis_builders(model: Any, tokenizer: Any) -> None:
    sent = sentence_axes(model, tokenizer, [("my cat sat.", "my dog sat."), ("a cat ran.", "a dog ran.")])
    assert sent.layers() == [0, 1]
    assert torch.allclose(sent.reads[1].norm(), torch.tensor(1.0), atol=1e-5)
    pairs = pair_axes(model, tokenizer, [("cat", "dog")])
    assert pairs.source == {"kind": "pairs", "method": "activations", "pairs": [["cat", "dog"]]}
    embed = pair_axes(model, tokenizer, [("a", "b")], method="embed")
    assert torch.equal(embed.reads[0], embed.reads[1])
    tmpl = template_axes(model, tokenizer, [("hi", "hi. you are a cat.")])
    assert tmpl.source["kind"] == "template"
    assert torch.isfinite(tmpl.writes[0]).all()
    with pytest.raises(ValueError, match="shared ending"):
        sentence_axes(model, tokenizer, [("ab", "ac")])


def test_units_parse_units_room_addresses() -> None:
    ids = [
        "/units/qwen3-0.6b/L8/n344+",
        "http://127.0.0.1:8770/units/qwen3-0.6b/L8/n12%2B",
        "L8·n13−",
        "L8 n14 -",
        "l20h17",
        "/units/qwen3-0.6b/L3/h5",
    ]
    units = Units.parse(ids, "Qwen/Qwen3-0.6B")
    assert units.neurons == {8: [12, 13, 14, 344]} and units.heads == {3: [5], 20: [17]}
    with pytest.raises(ValueError, match="not of the loaded"):
        Units.parse(["/units/qwen3-8b/L8/n344+"], "Qwen/Qwen3-0.6B")
    assert Units.parse(["/units/qwen3-8b/L8/n344+"]).neurons == {8: [344]}  # no model given: not checked


def test_unit_href_matches_the_units_room(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from steerlab.edits.reflection import model_slug, unit_href

    monkeypatch.setitem(sys.modules, "steerlab.atlas", None)  # not installed: the fallback rule
    assert model_slug("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b-instruct"
    assert unit_href("Qwen/Qwen3-8B", "neuron", 8, 344) == "/units/qwen3-8b/L8/n344+"
    assert unit_href("Qwen/Qwen3-8B", "head", 20, 17) == "/units/qwen3-8b/L20/h17"
    calls = []
    fake = types.ModuleType("steerlab.atlas")
    fake.model_slug = lambda m: "slug"  # type: ignore[attr-defined]
    fake.unit_href = lambda m, layer, index, pole=None: calls.append(pole) or f"/units/x/{layer}/{index}/{pole}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "steerlab.atlas", fake)
    assert unit_href("m", "neuron", 1, 2) == "/units/x/1/2/1" and unit_href("m", "head", 1, 2) == "/units/x/1/2/None"
    assert calls == [1, None]  # atlas's pole is an int (+1), a head has none
    assert model_slug("anything") == "slug"
