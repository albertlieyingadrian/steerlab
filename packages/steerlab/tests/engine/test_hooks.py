"""The two invariants every later unit relies on: strength 0 is the unsteered model, and prompt positions are
never steered, under both the cached decode loop and a no-cache teacher-forced pass."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sandbagkit.probing.hooks import capture_layers, count_forward_hooks  # noqa: E402

from steerlab.decode import forced_logits, greedy_generate  # noqa: E402

from .conftest import Tiny, direction_entry  # noqa: E402


def unsteered_tokens(t: Tiny, prompt: torch.Tensor, n: int) -> list[int]:
    """Plain greedy decoding with transformers' generate, the reference the engine must match at strength 0."""
    with torch.no_grad():
        out = t.model.generate(
            prompt, max_new_tokens=n, do_sample=False, repetition_penalty=1.0, pad_token_id=t.tokenizer.pad_token_id
        )
    ids: list[int] = out[0, prompt.shape[1] :].tolist()
    return ids


def test_strength_zero_is_token_for_token_identical_to_no_hook(tiny: Tiny) -> None:
    """On both tiny models: all strengths 0 -> the same tokens as no hook, and no hook left behind."""
    prompt = tiny.prompt_ids("Count to ten")
    reference = unsteered_tokens(tiny, prompt, 12)
    state = tiny.state()
    state.apply(tiny.control([direction_entry("random", 0.0), direction_entry("random0", 0.0, layers=[0, 1])]))
    assert state.layers == [0, 1]  # entries are live, their strength is just zero
    before = count_forward_hooks(tiny.model)
    with tiny.session(state) as session:
        assert count_forward_hooks(tiny.model) == before + 2
        steered = greedy_generate(session, prompt, max_new_tokens=12)
    assert count_forward_hooks(tiny.model) == before
    assert steered.token_ids == reference
    assert all(entry.applied for entry in steered.timeline)  # the zero entries were in force on every token
    assert all(norm == 0.0 for entry in steered.timeline for norm in entry.added_norm.values())


def test_nonzero_strength_changes_the_tokens(tiny: Tiny) -> None:
    """The complement of the identity test, so it is not passing vacuously."""
    prompt = tiny.prompt_ids("Count to ten")
    reference = unsteered_tokens(tiny, prompt, 12)
    state = tiny.state()
    state.apply(tiny.control([direction_entry("random", 40.0)]))
    with tiny.session(state) as session:
        steered = greedy_generate(session, prompt, max_new_tokens=12)
    assert steered.token_ids != reference


def test_prompt_positions_are_untouched_during_prefill(tiny: Tiny) -> None:
    """Capturing the steered layer's output on the prompt alone equals the unsteered prefill, at any strength."""
    prompt = tiny.prompt_ids("The prompt is never steered")
    with capture_layers(tiny.model, [0, 1]) as plain, torch.no_grad():
        tiny.model(input_ids=prompt)
        plain_out = {layer: hidden.clone() for layer, hidden in plain.items()}
    state = tiny.state()
    state.apply(tiny.control([direction_entry("random", 25.0, layers=[0, 1])]))
    with tiny.session(state) as session, capture_layers(tiny.model, [0, 1]) as steered, torch.no_grad():
        session.prompt_length = prompt.shape[1]
        session.position_offset = 0
        tiny.model(input_ids=prompt)
        for layer in (0, 1):
            assert torch.equal(steered[layer], plain_out[layer]), f"layer {layer} prompt positions were steered"


def test_generated_positions_are_steered_and_the_prompt_is_not_in_a_no_cache_pass(tiny: Tiny) -> None:
    """Teacher forcing without a cache: positions >= prompt length move by exactly the added vector at the hooked
    layer; every prompt position (including position 0, Gemma's BOS slot) is bit-identical."""
    prompt = tiny.prompt_ids("x")
    generated = torch.tensor([[10, 11, 12]])
    full = torch.cat([prompt, generated], dim=1)
    layer = 1
    with capture_layers(tiny.model, [layer]) as plain, torch.no_grad():
        tiny.model(input_ids=full)
        plain_out = plain[layer].clone()
    state = tiny.state()
    state.apply(tiny.control([direction_entry("random", 3.0)]))  # layer defaults to the direction's own layer, 1
    with tiny.session(state) as session, capture_layers(tiny.model, [layer]) as steered, torch.no_grad():
        session.prompt_length = prompt.shape[1]
        session.position_offset = 0
        tiny.model(input_ids=full)
        out = steered[layer]
    n_prompt = prompt.shape[1]
    assert torch.equal(out[:, :n_prompt], plain_out[:, :n_prompt])
    expected = plain_out[:, n_prompt:] + 3.0 * tiny.registry.directions["random"].unit
    assert torch.allclose(out[:, n_prompt:], expected, atol=1e-6)


def test_position_zero_is_never_modified_even_for_a_one_token_prompt(tiny_gemma3: Tiny) -> None:
    t = tiny_gemma3
    prompt = torch.tensor([[2]])  # one token: the BOS slot
    state = t.state()
    state.apply(t.control([direction_entry("random0", 9.0, layers=[0])]))
    with capture_layers(t.model, [0]) as plain, torch.no_grad():
        t.model(input_ids=torch.tensor([[2, 5, 6]]))
        plain_out = plain[0].clone()
    with t.session(state) as session, capture_layers(t.model, [0]) as steered, torch.no_grad():
        session.prompt_length = prompt.shape[1]
        session.position_offset = 0
        t.model(input_ids=torch.tensor([[2, 5, 6]]))
        out = steered[0]
    assert torch.equal(out[:, 0], plain_out[:, 0])
    assert not torch.equal(out[:, 1:], plain_out[:, 1:])


def test_forced_logits_match_the_cached_decode_step_by_step(tiny: Tiny) -> None:
    """KV parity (the Neuronpedia #163 class): the cached loop's per-step logits equal a no-cache teacher-forced
    pass over the same tokens, with a fixed nonzero control in force from the start."""
    prompt = tiny.prompt_ids("Parity")
    state = tiny.state()
    state.apply(tiny.control([direction_entry("random", 4.0, layers=[0, 1]), direction_entry("random0", -2.0)]))
    with tiny.session(state) as session:
        cached = greedy_generate(session, prompt, max_new_tokens=8)
        forced = forced_logits(session, prompt, cached.token_ids)
    assert len(cached.step_logits) == len(cached.token_ids) == forced.shape[0]
    for step, logits in enumerate(cached.step_logits):
        assert torch.allclose(logits, forced[step], atol=1e-4), f"step {step} differs"
    # and the teacher-forced argmaxes reproduce the greedy tokens
    assert forced.argmax(dim=-1).tolist() == cached.token_ids
