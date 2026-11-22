"""The Workbench's replies: greedy equals steerlab's decode loop, steering and edits apply and are undone."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from steerlab.control import ControlState
from steerlab.decode import encode_prompt, greedy_generate, render_messages
from steerlab.engine import SteeringSession
from steerlab.workbench import edits as edit_ops
from steerlab.workbench.session import Sampling, SteerError, Workbench, parse_expression

HI = [{"role": "user", "content": "hi"}]


def test_greedy_matches_the_decode_loop(bench: Workbench) -> None:
    reply = bench.generate(HI, sampling=Sampling(max_tokens=12))
    state = ControlState(model=bench.model_id, n_layers=bench.n_layers, resolver=bench.registry)
    with SteeringSession(bench.model, bench.tokenizer, state) as session:
        text = render_messages(bench.tokenizer, HI, model_type=session.model_type)
        reference = greedy_generate(session, encode_prompt(bench.tokenizer, text, "cpu"), max_new_tokens=12)
    assert reply["token_ids"] == reference.token_ids


def test_a_reply_carries_the_instruments(bench: Workbench) -> None:
    reply = bench.generate(HI, steer="3 formal.casual", sampling=Sampling(max_tokens=4))
    assert reply["steer"][0]["source"] == {"kind": "direction", "name": "probe:formal.casual"}
    token = reply["tokens"][1]
    assert set(token["probes"]) == set(bench.bank.probes) and set(token["bands"]) == set(token["probes"])
    assert len(token["alternatives"]) == 5 and token["logprob"] <= 0
    assert token["units"] and token["units"][0]["address"].startswith("L")
    assert token["units"][0]["href"] == "/units/tiny-random-qwen2/" + token["units"][0]["address"]
    probe_layer = str(bench.bank.probes["formal.casual"].layer)
    assert token["steer"][probe_layer]["ratio"] > 0  # steered from the first generated position on
    assert reply["tokens"][0]["steer"] == {}  # the first token is chosen at the last prompt position: unsteered


def test_steering_changes_readings_and_is_undone(bench: Workbench) -> None:
    plain = bench.generate(HI, sampling=Sampling(max_tokens=6))
    pushed = bench.generate(HI, steer=[{"probe": "formal.casual", "strength": 50}], sampling=Sampling(max_tokens=6))
    assert pushed["tokens"][1]["probes"]["formal.casual"] > plain["tokens"][1]["probes"]["formal.casual"]
    assert (
        bench.state.applied == []
        and bench.generate(HI, sampling=Sampling(max_tokens=6))["token_ids"] == plain["token_ids"]
    )


def test_seeded_sampling_repeats_and_replay_matches(bench: Workbench) -> None:
    a = bench.generate(HI, sampling=Sampling(max_tokens=8, temperature=1.0, seed=7))
    b = bench.generate(HI, sampling=Sampling(max_tokens=8, temperature=1.0, seed=7))
    assert a["token_ids"] == b["token_ids"]
    replay = bench.replay(HI, a["token_ids"])
    for name, zs in replay["probes"].items():
        assert np.allclose(zs, [t["probes"][name] for t in a["tokens"]], atol=2e-3)


def test_expressions_and_bad_steering() -> None:
    assert parse_expression("0.5 formal.casual - 2 probe:x@1,3-4 + y") == [
        {"name": "formal.casual", "strength": 0.5},
        {"name": "probe:x", "strength": -2.0, "layers": [1, 3, 4]},
        {"name": "y", "strength": 1.0},
    ]


def test_unknown_names_say_what_is_loaded(bench: Workbench) -> None:
    with pytest.raises(SteerError, match="loaded"):
        bench.generate(HI, steer="2 nothing-here", sampling=Sampling(max_tokens=2))


def test_an_edit_applies_and_restores_bit_for_bit(bench: Workbench) -> None:
    weight = bench.model.model.layers[1].mlp.down_proj.weight
    before = weight.detach().clone()
    axis = torch.randn(weight.shape[0]).tolist()
    edit = {
        "type": "edit",
        "name": "e",
        "mode": "ablate",
        "axis": axis,
        "units": [{"kind": "neuron", "layer": 1, "index": i} for i in range(8)]
        + [{"kind": "head", "layer": 0, "index": 1}],
    }
    with edit_ops.applied(bench.model, [edit]) as labels:
        unit = torch.tensor(axis) / torch.tensor(axis).norm()
        assert torch.allclose(unit @ weight[:, :8], torch.zeros(8), atol=1e-5)  # nothing left along the axis
        assert labels == ["e: ablate k=1 on 9 units"]
    assert torch.equal(weight, before)
    reply = bench.generate(HI, edits=[edit], sampling=Sampling(max_tokens=3))
    assert reply["edits"] == ["e: ablate k=1 on 9 units"] and torch.equal(weight, before)
    with pytest.raises(edit_ops.EditError):
        bench.generate(HI, edits=[{**edit, "axis": [1.0]}], sampling=Sampling(max_tokens=2))
    assert torch.equal(weight, before)


def test_unit_slug_matches_the_atlas_rule() -> None:
    from steerlab.workbench.session import model_slug, unit_href

    assert model_slug("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b-instruct"
    assert model_slug("Qwen/Qwen3-8B") == "qwen3-8b"
    assert (
        unit_href("Qwen/Qwen3-8B", {"kind": "neuron", "layer": 8, "index": 344, "pole": "+"})
        == "/units/qwen3-8b/L8/n344+"
    )
    assert unit_href("Qwen/Qwen3-8B", {"kind": "head", "layer": 20, "index": 17}) == "/units/qwen3-8b/L20/h17"


def test_unit_href_passes_the_atlas_pole_as_a_signed_int(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import types

    from steerlab.workbench.session import unit_href

    calls = []

    def atlas_href(model_id: str, layer: int, index: int, pole: int | None = None) -> str:
        calls.append(pole)
        if pole is not None and pole not in (1, -1):
            raise ValueError("a neuron's pole is +1 or -1")
        return f"/units/x/L{layer}/{'h' if pole is None else 'n'}{index}{'' if pole is None else '+-'[pole < 0]}"

    real = importlib.import_module
    fake = types.SimpleNamespace(unit_href=atlas_href)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake if name == "steerlab.atlas" else real(name))
    assert unit_href("m", {"kind": "neuron", "layer": 8, "index": 344, "pole": "-"}) == "/units/x/L8/n344-"
    assert unit_href("m", {"kind": "neuron", "layer": 8, "index": 344, "pole": "+"}) == "/units/x/L8/n344+"
    assert unit_href("m", {"kind": "head", "layer": 2, "index": 1}) == "/units/x/L2/h1"
    assert calls == [-1, 1, None]
