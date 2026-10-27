"""Clamp mode: an SAE feature's activation is read every token and moved to the target, instead of a fixed vector added.

The fake SAE below has features whose encoder row and decoder vector are the same unit vector u, so a feature's
activation on h is h·u when above its threshold (else 0), and clamping to T adds (T - activation)·u to each steered
position.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sandbagkit.probing.hooks import capture_layers  # noqa: E402

from steerlab.control import ControlState, ControlValidationError, parse_control  # noqa: E402
from steerlab.decode import greedy_generate  # noqa: E402
from steerlab.engine import DirectionRegistry  # noqa: E402
from steerlab.saes.common import LoadedSAE  # noqa: E402

from .conftest import Tiny, direction_entry, unit_direction  # noqa: E402

RELEASE = "test/clamp-sae"
LAYER = 1


class UnitSAE(LoadedSAE):
    """Two features with encoder row == decoder vector (unit vectors); feature 1 has a threshold nothing reaches."""

    def __init__(self, d: int) -> None:
        self.release = RELEASE
        self.layer = LAYER
        self.units = torch.stack([unit_direction(d, 20), unit_direction(d, 21)])
        self.thresholds = torch.tensor([0.0, 1e9])

    @property
    def d_model(self) -> int:
        return int(self.units.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.units.shape[0])

    @property
    def device(self) -> torch.device:
        return self.units.device

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        pre = self._prepare(h) @ self.units.T
        return pre * (pre > self.thresholds).to(pre.dtype)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self._prepare(acts) @ self.units

    def decoder_vector(self, feature: int) -> torch.Tensor:
        return self.units[feature]

    def encoder_probe(self, feature: int) -> tuple[torch.Tensor, float, float]:
        return self.units[feature], 0.0, float(self.thresholds[feature])


def clamp_entry(feature: int, target: float, **extra: Any) -> dict[str, Any]:
    return {
        "source": {"kind": "sae", "release": RELEASE, "layer": LAYER, "feature": feature},
        "strength": target,
        "mode": "clamp",
        **extra,
    }


def clamp_state(t: Tiny, sae: UnitSAE | None) -> tuple[ControlState, list[str]]:
    registry = DirectionRegistry()
    if sae is not None:
        registry.add_sae(sae)
    registry.directions.update(t.registry.directions)
    messages: list[str] = []
    return ControlState(t.name, t.n_layers, registry, on_message=messages.append), messages


def test_clamp_sets_the_feature_activation_to_the_target_on_generated_positions_only(tiny: Tiny) -> None:
    sae = UnitSAE(tiny.d_model)
    prompt = tiny.prompt_ids("x")
    full = torch.cat([prompt, torch.tensor([[10, 11, 12]])], dim=1)
    with capture_layers(tiny.model, [LAYER]) as plain, torch.no_grad():
        tiny.model(input_ids=full)
        plain_out = plain[LAYER].clone()
    state, messages = clamp_state(tiny, sae)
    state.apply(tiny.control([clamp_entry(0, 7.5)]))
    assert messages == [] and state.layers == [LAYER] and state.deltas == {}
    with tiny.session(state) as session, capture_layers(tiny.model, [LAYER]) as steered, torch.no_grad():
        session.prompt_length = prompt.shape[1]
        session.position_offset = 0
        tiny.model(input_ids=full)
        out = steered[LAYER]
    n_prompt = prompt.shape[1]
    assert torch.equal(out[:, :n_prompt], plain_out[:, :n_prompt])
    before = plain_out[0, n_prompt:].float() @ sae.units[0]
    after = out[0, n_prompt:].float() @ sae.units[0]
    # The SAE's activation (h·u where positive, else 0) is replaced by 7.5; a negative h·u keeps its negative part.
    assert torch.allclose(after, 7.5 + before.clamp(max=0), atol=1e-4)
    assert torch.allclose(sae.encode(out[0, n_prompt:])[:, 0], 7.5 + before.clamp(max=0), atol=1e-4)
    assert session.last_stats[LAYER][0] > 0


def test_clamp_adds_less_where_the_feature_already_fires(tiny_qwen3: Tiny) -> None:
    """The point of clamping: the push is the target minus what is already there, so it varies by position."""
    t = tiny_qwen3
    sae = UnitSAE(t.d_model)
    state, _ = clamp_state(t, sae)
    state.apply(t.control([clamp_entry(0, 5.0)]))
    hidden = torch.stack([2.0 * sae.units[0], torch.zeros(t.d_model)]).unsqueeze(0)  # fires at 2, then silent
    with t.session(state) as session:
        out = session._clamped(hidden, session._clamps[LAYER])
    assert torch.allclose((out - hidden).norm(dim=-1).flatten(), torch.tensor([3.0, 5.0]), atol=1e-5)


def test_clamping_a_silent_feature_to_zero_is_the_unsteered_model(tiny: Tiny) -> None:
    """Feature 1 never fires (huge threshold), so clamping it to 0 adds exactly zero: same tokens as no steering."""
    prompt = tiny.prompt_ids("Count to ten")
    plain, _ = clamp_state(tiny, None)
    with tiny.session(plain) as session:
        reference = greedy_generate(session, prompt, max_new_tokens=10).token_ids
    state, _ = clamp_state(tiny, UnitSAE(tiny.d_model))
    state.apply(tiny.control([clamp_entry(1, 0.0)]))
    with tiny.session(state) as session:
        assert greedy_generate(session, prompt, max_new_tokens=10).token_ids == reference


def test_a_strong_clamp_changes_the_tokens(tiny: Tiny) -> None:
    prompt = tiny.prompt_ids("Count to ten")
    plain, _ = clamp_state(tiny, None)
    with tiny.session(plain) as session:
        reference = greedy_generate(session, prompt, max_new_tokens=10).token_ids
    state, _ = clamp_state(tiny, UnitSAE(tiny.d_model))
    state.apply(tiny.control([clamp_entry(0, 60.0)]))
    with tiny.session(state) as session:
        assert greedy_generate(session, prompt, max_new_tokens=10).token_ids != reference


def test_clamp_and_add_entries_combine_on_one_layer(tiny_qwen3: Tiny) -> None:
    t = tiny_qwen3
    state, messages = clamp_state(t, UnitSAE(t.d_model))
    state.apply(t.control([direction_entry("random", 2.0), clamp_entry(0, 3.0)]))
    assert messages == []
    assert set(state.deltas) == {LAYER} and set(state.clamps) == {LAYER}
    assert [entry.mode for entry in state.applied] == ["add", "clamp"]
    clamped = state.applied[1]
    assert clamped.norm == pytest.approx(3.0) and "clamped to 3.0" in clamped.describe()
    assert clamped.to_dict()["mode"] == "clamp"


def test_clamp_entries_that_cannot_work_are_skipped_with_a_reason(tiny_qwen3: Tiny) -> None:
    t = tiny_qwen3
    state, messages = clamp_state(t, None)
    state.apply(
        t.control(
            [
                {**direction_entry("random", 1.0), "mode": "clamp"},
                clamp_entry(0, 1.0),
                clamp_entry(0, 1.0, layers=[0]),
            ]
        )
    )
    assert state.applied == [] and state.clamps == {}
    assert "clamp needs an SAE feature" in messages[0]
    assert "--sae-layer 1" in messages[1]
    assert "only sit at its SAE's layer 1" in messages[2]


def test_mode_is_absent_by_default_and_rejects_anything_else() -> None:
    source = '{"kind": "direction", "name": "x"}'
    assert (
        parse_control(f'{{"model": "m", "entries": [{{"source": {source}, "strength": 1}}]}}').entries[0].mode is None
    )
    with pytest.raises(ControlValidationError, match="mode"):
        parse_control(f'{{"model": "m", "entries": [{{"source": {source}, "strength": 1, "mode": "multiply"}}]}}')
