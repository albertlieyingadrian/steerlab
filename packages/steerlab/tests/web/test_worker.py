"""The worker on the tiny model: live control changes, cancellation, queueing, prompt, redo, clear and swap.

A visitor's sink runs on the worker thread, so a test can act "while the reply is at token N" by moving a fader from
inside the sink when token N arrives; the change lands before token N + 1, exactly like an admin's click.
"""

from __future__ import annotations

from typing import Any

import pytest

from steerlab.web import worker as worker_module
from steerlab.web.conversations import MessageRejected
from steerlab.web.worker import Worker

from .conftest import SAE_RELEASE, SYSTEM, Visitor, wait_until

torch = pytest.importorskip("torch")


def send(worker: Worker, conversation: str, text: str = "Hello there", **kwargs: Any) -> Visitor:
    visitor = Visitor(**kwargs)
    worker.submit(conversation, text, visitor)
    return visitor


def markers(worker: Worker, conversation: str) -> list[dict[str, Any]]:
    broker: Any = worker.broker
    return [event for event in broker.of("marker") if event["conversation"] == conversation]


def test_a_fader_moved_at_token_5_steers_from_token_6(make_worker: Any) -> None:
    """The change applies from the next generated token, and the mirror gets a marker there."""
    worker = make_worker(max_new_tokens=12)
    baseline = send(worker, "a").wait()
    assert baseline.end["reason"] in ("eos", "length") and len(baseline.text) > 6

    def at_token_5(event: dict[str, Any]) -> None:
        if event["type"] == "token" and event["index"] == 5:
            worker.set_strength("random", 40.0)

    worker.new_conversation("a")
    steered = send(worker, "a", on_event=at_token_5).wait()
    [marker] = markers(worker, "a")
    assert marker["token_index"] == 6
    assert marker["applied"][0]["source"] == {"kind": "direction", "name": "random"}
    assert marker["applied"][0]["strength"] == 40.0
    assert steered.text[:6] == baseline.text[:6] and steered.text != baseline.text


def test_a_change_between_replies_marks_token_0_of_the_next(make_worker: Any) -> None:
    worker = make_worker()
    send(worker, "a").wait()
    worker.set_strength("random", 2.0)
    wait_until(lambda: worker.state()["directions"][0]["strength"] == 2.0)
    send(worker, "b").wait()
    assert [marker["token_index"] for marker in markers(worker, "b")] == [0]


def test_with_every_fader_at_0_the_reply_matches_plain_greedy_decoding(make_worker: Any) -> None:
    """The server adds nothing of its own; strength 0 is the unsteered model."""
    from steerlab.decode import encode_prompt, greedy_generate, render_messages
    from steerlab.engine import SteeringSession

    from .conftest import tiny_session

    worker = make_worker(max_new_tokens=10)
    worker.set_strength("random", 0.0)
    reply = send(worker, "a", "Hi").wait()
    backend, state = tiny_session("tiny/qwen3", print, directions=True, sae=False)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hi"}]
    with SteeringSession(backend.model, backend.tokenizer, state) as session:
        ids = encode_prompt(session.tokenizer, render_messages(session.tokenizer, messages, model_type="qwen3"), "cpu")
        direct = greedy_generate(session, ids, max_new_tokens=10)
    assert reply.text == direct.text
    assert len([event for event in reply.events if event["type"] == "token"]) == len(direct.token_ids)


def test_stop_ends_the_reply_within_one_token_and_keeps_the_partial_turn(make_worker: Any) -> None:
    worker = make_worker(max_new_tokens=40)

    def stop_at_3(event: dict[str, Any]) -> None:
        if event["type"] == "token" and event["index"] == 3:
            worker.stop_reply()

    visitor = send(worker, "a", "Hello", on_event=stop_at_3).wait()
    tokens = [event["index"] for event in visitor.events if event["type"] == "token"]
    assert tokens == [0, 1, 2, 3] and visitor.end["reason"] == "stopped"
    assert worker.store.transcript("a") == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": visitor.text},
    ]


def test_a_visitor_who_leaves_mid_reply_frees_the_queue_and_their_turn(make_worker: Any) -> None:
    """The reply stops within one token, the user turn is removed, and the next visitor starts."""
    worker = make_worker(max_new_tokens=40)
    jobs: dict[str, Any] = {}

    def leave_at_2(event: dict[str, Any]) -> None:
        if event["type"] == "token" and event["index"] == 2:
            worker.disconnect(jobs["a"])

    first = Visitor(on_event=leave_at_2)
    jobs["a"] = worker.submit("a", "Hello", first)
    second = send(worker, "b")
    first.wait()
    second.wait()
    assert first.end["reason"] == "disconnected"
    assert max(event["index"] for event in first.events if event["type"] == "token") == 2
    assert worker.store.transcript("a") == []
    assert second.end["reason"] in ("eos", "length")


def test_visitors_are_served_in_order_and_one_message_each(make_worker: Any) -> None:
    worker = make_worker(max_new_tokens=20)
    started: list[str] = []
    first = send(worker, "a", on_event=lambda e: e["type"] == "start" and started.append("a"))
    second = send(worker, "b", on_event=lambda e: e["type"] == "start" and started.append("b"))
    with pytest.raises(MessageRejected, match="Wait for the reply"):
        worker.submit("a", "again", Visitor())
    first.wait()
    second.wait()
    assert started == ["a", "b"]
    assert any(event == {"type": "queued", "position": 2} for event in second.events)


def test_a_prompt_change_reaches_queued_replies_but_not_the_one_streaming(
    make_worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import steerlab.decode

    systems: list[str] = []
    render = steerlab.decode.render_messages

    def spy(tokenizer: Any, messages: Any, *, model_type: str) -> str:
        systems.append(messages[0]["content"])
        return render(tokenizer, messages, model_type=model_type)

    monkeypatch.setattr(steerlab.decode, "render_messages", spy)
    worker = make_worker(max_new_tokens=20)
    jobs: list[Visitor] = []

    def change_prompt(event: dict[str, Any]) -> None:
        if event["type"] == "token" and event["index"] == 1:
            jobs.append(send(worker, "b"))
            jobs.append(send(worker, "c"))
            worker.set_prompt("Answer like a pirate.")

    send(worker, "a", on_event=change_prompt).wait()
    for job in jobs:
        job.wait()
    assert systems == [SYSTEM, "Answer like a pirate.", "Answer like a pirate."]
    worker.set_prompt(None)
    wait_until(lambda: worker.state()["prompt"] == SYSTEM)


def test_redo_regenerates_the_last_reply_with_the_current_faders(make_worker: Any) -> None:
    worker = make_worker()
    first = send(worker, "a").wait()
    worker.set_strength("random", 40.0)
    worker.redo("a")
    wait_until(lambda: len(worker.broker.of("end")) == 2)  # type: ignore[attr-defined]
    transcript = worker.store.transcript("a")
    assert [turn["role"] for turn in transcript] == ["user", "assistant"]
    assert transcript[1]["content"] != first.text
    assert markers(worker, "a")[0]["token_index"] == 0
    assert worker.status("a")["version"] == 2  # bumped at the redo's start and end


def test_clear_while_idle_empties_the_conversation_and_bumps_its_version(make_worker: Any) -> None:
    worker = make_worker()
    send(worker, "a").wait()
    worker.set_strength("random", 1.0)
    before = worker.status("a")["version"]
    worker.clear()
    wait_until(lambda: worker.status("a")["version"] > before)
    assert worker.store.transcript("a") == []
    assert worker.state()["directions"][0]["strength"] == 0.0


def test_a_swap_mid_reply_resets_everyone_and_loads_the_new_model(make_worker: Any, unloaded: list[str]) -> None:
    """On success the reply ends with reset, the queued visitor too, faders and conversations reset."""
    worker = make_worker(max_new_tokens=40)
    worker.set_strength("random", 1.0)
    queued: list[Visitor] = []

    def swap_at_2(event: dict[str, Any]) -> None:
        if event["type"] == "token" and event["index"] == 2:
            queued.append(send(worker, "b"))
            worker.set_strength("random", 3.0)  # posted just before the swap: must not steer the new model
            worker.swap("plain")

    active = send(worker, "a", on_event=swap_at_2).wait()
    queued[0].wait()
    assert active.end["reason"] == "reset" and queued[0].end["reason"] == "reset"
    wait_until(lambda: worker.state()["model"] == "plain" and worker.state()["status"] == "ready")
    state = worker.state()
    assert state["directions"] == [] and state["sae"] is None
    assert unloaded == ["tiny/qwen3"]
    assert worker.store.transcript("a") == []
    reply = send(worker, "c").wait()
    assert reply.end["reason"] in ("eos", "length")
    assert markers(worker, "c") == []


def test_a_failed_swap_reports_why_and_restores_the_previous_model(make_worker: Any) -> None:
    """Reset on failure."""
    worker = make_worker()
    worker.swap("broken")
    wait_until(lambda: any(e["status"] == "restoring" for e in worker.broker.of("state")))  # type: ignore[attr-defined]
    wait_until(lambda: worker.state()["status"] == "ready")
    assert worker.state()["model"] == "tiny"
    notices = [event["message"] for event in worker.broker.of("notice")]  # type: ignore[attr-defined]
    assert any("Could not load broken" in message and "corrupt" in message for message in notices)
    assert send(worker, "a").wait().end["reason"] in ("eos", "length")


def test_a_swap_to_a_key_outside_the_catalog_is_refused(make_worker: Any) -> None:
    worker = make_worker()
    for key in ("Qwen/Qwen3-8B", "/tmp/model", ""):
        with pytest.raises(ValueError, match="not a downloaded model"):
            worker.swap(key)
    assert worker.state()["model"] == "tiny"


def test_unknown_sources_and_bad_strengths_are_rejected(make_worker: Any) -> None:
    worker = make_worker()
    with pytest.raises(ValueError, match="unknown direction"):
        worker.set_strength("korean", 1.0)
    with pytest.raises(ValueError, match="finite"):
        worker.set_strength("random", float("nan"))
    with pytest.raises(ValueError, match="add it first"):
        worker.set_strength("feature:3", 1.0)


def test_an_sae_feature_can_be_added_steered_and_removed(make_worker: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = make_worker()
    state = worker.state()
    assert state["sae"] == {"release": SAE_RELEASE, "layer": 1, "features": 6}
    worker.add_feature(3)
    worker.add_feature(0)
    assert worker.state()["features"] == [
        {"id": 3, "label": "Bridges", "strength": 0.0, "suggested": None, "mode": "add"},
        {"id": 0, "label": None, "strength": 0.0, "suggested": None, "mode": "add"},
    ]
    with pytest.raises(ValueError, match="from 0 to 5"):
        worker.add_feature(6)
    monkeypatch.setattr(worker_module, "MAX_FEATURES", 2)
    with pytest.raises(ValueError, match="at most 2"):
        worker.add_feature(1)
    worker.set_strength("feature:3", 4.0)
    wait_until(lambda: worker.state()["features"][0]["strength"] == 4.0)
    send(worker, "a").wait()
    [marker] = markers(worker, "a")
    assert marker["applied"][0]["source"] == {"kind": "sae", "release": SAE_RELEASE, "layer": 1, "feature": 3}
    worker.remove_feature(3)
    assert [feature["id"] for feature in worker.state()["features"]] == [0]


def test_a_model_without_an_sae_has_no_features_to_add(make_worker: Any) -> None:
    worker = make_worker(model="plain")
    with pytest.raises(ValueError, match="no SAE"):
        worker.add_feature(1)


def test_a_conversation_past_the_token_cap_is_refused(make_worker: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "MAX_PROMPT_TOKENS", 60)
    worker = make_worker()
    visitor = send(worker, "a", "x" * 80).wait()
    assert visitor.end["reason"] == "full"
    assert {"type": "error", "message": "This conversation is full. Start a new one."} in visitor.events
    assert worker.store.transcript("a") == []


def test_the_admin_snapshot_has_the_reply_in_progress(make_worker: Any) -> None:
    worker = make_worker(max_new_tokens=30)
    snapshots: list[Any] = []

    def snap_at_4(event: dict[str, Any]) -> None:
        if event["type"] == "token" and event["index"] == 4:
            snapshots.append(worker.broker.snapshot())

    send(worker, "a", "Hello", on_event=snap_at_4).wait()
    [conversation] = snapshots[0]["conversations"]
    assert conversation["label"] == 1
    user, reply = conversation["turns"]
    assert user == {"role": "user", "text": "Hello"}
    assert [index for index, _ in reply["tokens"]] == [0, 1, 2, 3, 4] and reply["finish"] is None


def test_a_failure_outside_the_reply_loop_does_not_stop_the_worker(
    make_worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat template that refuses a history, or a fader change that cannot apply, is reported and skipped."""
    import steerlab.decode

    worker = make_worker()
    render = steerlab.decode.render_messages
    calls = {"n": 0}

    def refuse_once(tokenizer: Any, messages: Any, *, model_type: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("Conversation roles must alternate (test)")
        return render(tokenizer, messages, model_type=model_type)

    monkeypatch.setattr(steerlab.decode, "render_messages", refuse_once)
    failed = send(worker, "a").wait()
    assert failed.end["reason"] == "failed" and worker.store.transcript("a") == []

    def broken_apply(control: Any) -> None:
        raise RuntimeError("resolver crashed (test)")

    monkeypatch.setattr(worker.poller.state, "apply", broken_apply)
    worker.set_strength("random", 1.0)

    def notices() -> list[str]:
        return [event["message"] for event in worker.broker.of("notice")]  # type: ignore[attr-defined]

    wait_until(lambda: any("resolver crashed" in message for message in notices()))
    monkeypatch.undo()
    assert send(worker, "b").wait().end["reason"] in ("eos", "length")


def test_a_preset_sets_the_prompt_suggestions_and_features_of_each_model(make_worker: Any) -> None:
    """On load: the model's own prompt, a suggested strength per direction, its features on the desk (and again after
    Clear all); a swap brings the next model's prompt."""
    from steerlab.demo import Preset, PresetFeature

    presets = {
        "tiny": Preset(prompt="P-tiny", suggest={"random": 1.5}, features=(PresetFeature(4, "Golden Gate", 16.0),)),
        "plain": Preset(prompt="P-plain"),
    }
    worker = make_worker(presets=presets, prompt_for=lambda key: f"prompt for {key}")
    state = worker.state()
    assert state["prompt"] == "prompt for tiny" and state["default_prompt"] == "prompt for tiny"
    assert state["directions"][0]["suggested"] == 1.5
    assert state["features"] == [{"id": 4, "label": "Golden Gate", "strength": 0.0, "suggested": 16.0, "mode": "add"}]
    worker.clear()
    wait_until(lambda: any(e["message"].startswith("Cleared") for e in worker.broker.of("notice")))  # type: ignore[attr-defined]
    assert [feature["id"] for feature in worker.state()["features"]] == [4]
    worker.swap("plain")
    wait_until(lambda: worker.state()["model"] == "plain" and worker.state()["status"] == "ready")
    assert worker.state()["prompt"] == "prompt for plain" and worker.state()["features"] == []


def test_a_preset_feature_in_clamp_mode_reaches_the_control_file_as_a_clamp(make_worker: Any) -> None:
    """A clamped preset feature shows as one on the desk, and moving its fader writes a clamp entry, not an add."""
    from steerlab.demo import Preset, PresetFeature

    presets = {"tiny": Preset(prompt="P", features=(PresetFeature(4, "GG", 4.0, "clamp"), PresetFeature(5, "x", 8.0)))}
    worker = make_worker(presets=presets)
    assert [(f["id"], f["mode"]) for f in worker.state()["features"]] == [(4, "clamp"), (5, "add")]
    worker.set_strength("feature:4", 4.0)
    worker.set_strength("feature:5", 8.0)
    control = worker._control_file()
    assert control is not None
    assert [(entry.source.feature, entry.mode) for entry in control.entries] == [(4, "clamp"), (5, None)]  # type: ignore[union-attr]
