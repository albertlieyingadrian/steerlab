"""The inspector's numbers on the tiny model: the split adds up, the lens ends at the real output, edits touch only
what they name, an unedited run is the plain model, and the dose test's ×1 column is the normal run."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from steerlab.inspect.core import Edit, Inspector

from .conftest import PROMPT


def test_the_parts_add_up_to_the_real_logit(inspector: Inspector) -> None:
    run = inspector.run(PROMPT)
    parts = run["embed"] + sum(row["attn"] + row["mlp"] for row in run["layers"] if row["layer"] >= 0)
    assert parts == pytest.approx(run["sum"], abs=1e-9)
    assert abs(run["leftover"]) < 1e-4
    assert run["sum"] + run["leftover"] == pytest.approx(run["target"]["logit"], abs=1e-9)


def test_the_split_holds_for_any_target_and_position(inspector: Inspector) -> None:
    for target, pos in [("e", None), (7, 2), (None, 0)]:
        run = inspector.run(PROMPT, target=target, pos=pos)
        assert abs(run["leftover"]) < 1e-4
    assert inspector.run(PROMPT, pos=2)["pos"] == 2


def test_the_lens_at_the_last_layer_is_the_real_output(inspector: Inspector) -> None:
    run = inspector.run(PROMPT)
    last = run["layers"][-1]
    assert last["layer"] == inspector.n_layers - 1
    assert last["says"] == run["guesses"][0]["text"]
    assert last["target_prob"] == pytest.approx(run["target"]["prob"], abs=1e-5)
    assert run["layers"][0]["layer"] == -1  # the embedding row


def test_an_unedited_trace_is_the_plain_model_bit_for_bit(inspector: Inspector) -> None:
    ids = inspector.tokenize(PROMPT)
    with torch.no_grad():
        plain = inspector.model(torch.tensor([ids]), use_cache=False).logits[0]
    assert torch.equal(inspector.trace(ids).logits, plain)
    assert torch.equal(inspector.trace(ids, []).logits, plain)


def test_unedited_generation_is_plain_greedy_generation(inspector: Inspector) -> None:
    ids = inspector.tokenize(PROMPT)
    with torch.no_grad():
        plain = inspector.model.generate(
            torch.tensor([ids]),
            max_new_tokens=6,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=inspector.tokenizer.pad_token_id,
        )[0, len(ids) :].tolist()
    assert inspector.generate(ids, (), max_new_tokens=6) == plain


def test_an_edit_changes_only_its_neuron_and_is_removed_after(inspector: Inspector) -> None:
    ids = inspector.tokenize(PROMPT)
    base = inspector.trace(ids)
    neuron = int(base.acts[1][-1].abs().argmax())
    edited = inspector.trace(ids, [Edit(1, neuron, "off")])
    assert torch.all(edited.acts[1][:, neuron] == 0)
    others = [n for n in range(inspector.d_mlp) if n != neuron]
    assert torch.equal(edited.acts[1][:, others], base.acts[1][:, others])
    assert torch.equal(edited.acts[0], base.acts[0])  # an earlier layer cannot see a later edit
    assert not torch.equal(edited.logits, base.logits)
    assert torch.equal(inspector.trace(ids).logits, base.logits)  # the hook is gone


EXPECTED: list[tuple[str, float, Callable[[torch.Tensor], torch.Tensor]]] = [
    ("off", 0.0, lambda a: a * 0),
    ("flip", 0.0, lambda a: -a),
    ("scale", 3.0, lambda a: 3 * a),
    ("set", 1.5, lambda a: torch.full_like(a, 1.5)),
]


@pytest.mark.parametrize(("mode", "value", "expect"), EXPECTED)
def test_each_edit_mode(
    inspector: Inspector, mode: str, value: float, expect: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    ids = inspector.tokenize(PROMPT)
    base = inspector.trace(ids).acts[0][:, 5]
    edited = inspector.trace(ids, [Edit(0, 5, mode, value)]).acts[0][:, 5]  # type: ignore[arg-type]
    assert torch.allclose(edited, expect(base))


def test_bad_neurons_and_prompts_are_refused(inspector: Inspector) -> None:
    with pytest.raises(ValueError, match="layer"):
        inspector.layer(PROMPT, inspector.n_layers)
    with pytest.raises(ValueError, match="neuron"):
        inspector.neuron(PROMPT, 0, inspector.d_mlp)
    with pytest.raises(ValueError, match="at most"):
        inspector.run("x" * 100)
    with pytest.raises(ValueError, match="position"):
        inspector.run(PROMPT, pos=50)
    with pytest.raises(ValueError, match="mode"):
        Edit(0, 0, "melt").apply(torch.ones(2))  # type: ignore[arg-type]


def test_layer_view_shares_match_the_layer_bars(inspector: Inspector) -> None:
    run = inspector.run(PROMPT)
    view = inspector.layer(PROMPT, 1, k=inspector.d_mlp // 2)
    assert view["mlp"] == pytest.approx(run["layers"][-1]["mlp"], abs=1e-5)
    assert view["attn"] == pytest.approx(run["layers"][-1]["attn"], abs=1e-5)
    assert sum(h["share"] for h in view["heads"]) == pytest.approx(view["attn"], abs=1e-5)
    shares = [row["share"] for row in view["toward"]]
    assert shares == sorted(shares, reverse=True)
    for row in view["toward"][:3]:
        assert row["share"] == pytest.approx(row["act"] * row["align"], rel=1e-4)
    assert len(view["heads"]) == inspector.n_heads and view["heads"][0]["attends"]


def test_neuron_view_fires_on_every_token(inspector: Inspector) -> None:
    view = inspector.neuron(PROMPT, 1, 3, k=5)
    assert len(view["fires"]) == len(PROMPT) and len(view["promotes"]) == 5
    assert view["act"] == view["fires"][-1]["act"]
    assert view["share"] == pytest.approx(view["act"] * view["align"], rel=1e-4)


def test_find_ranks_every_neuron_by_cosine(inspector: Inspector) -> None:
    found = inspector.find("e", PROMPT, k=4)
    v = inspector.w_norm * inspector.W_U[found["word"]["id"]]
    pairs = [(layer, n) for layer in range(inspector.n_layers) for n in range(inspector.d_mlp)]
    best = max(pairs, key=lambda ln: float(torch.cosine_similarity(inspector.down(ln[0])[ln[1]], v, dim=0)))
    assert (found["toward"][0]["layer"], found["toward"][0]["neuron"]) == best
    assert all(-1 <= row["cos"] <= 1 for row in found["toward"] + found["against"])
    assert found["toward"][0]["cos"] > 0 > found["against"][0]["cos"]
    assert "share" in found["toward"][0]


def test_target_words_prefer_the_spaced_form_only_when_it_is_shorter(inspector: Inspector) -> None:
    # The character tokenizer splits " e" into two tokens, so the bare "e" wins; a leading space is kept as typed.
    assert inspector.target_id("e") == inspector.tokenize("e")[0]
    assert inspector.target_id(" e") == inspector.tokenize(" ")[0]


def test_edit_view_compares_the_same_target(inspector: Inspector) -> None:
    result = inspector.edit(PROMPT, [Edit(1, 0, "scale", 50.0)], max_new_tokens=4)
    assert result["normal"]["target"]["id"] == result["edited"]["target"]["id"]
    assert result["edits"] == ["L1.N0 ×50"]
    assert isinstance(result["edited_text"], str)


def test_dose_at_x1_is_the_normal_run_for_every_neuron(inspector: Inspector) -> None:
    dose = inspector.dose(PROMPT, 1, 3, n_random=5)
    one = dose["scales"].index(1.0)
    assert dose["curve"][one] == pytest.approx(dose["base_prob"], abs=1e-6)
    assert all(curve[one] == pytest.approx(dose["base_prob"], abs=1e-6) for curve in dose["random"])
    assert len(dose["random"]) == 5 and "L1.N3" not in dose["random_ids"]
    assert all(lo <= hi for lo, hi in zip(dose["band_lo"], dose["band_hi"], strict=True))
    zero = dose["scales"].index(0.0)
    off = inspector.edit(PROMPT, [Edit(1, 3, "off")], target=dose["target"]["id"], max_new_tokens=1)
    assert dose["curve"][zero] == pytest.approx(off["edited"]["target"]["prob"], abs=1e-5)
