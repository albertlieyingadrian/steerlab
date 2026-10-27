"""The decode loop: events mode, the control-file watcher mid-run, the timeline, and the CLI's generate and chat."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sandbagkit.probing.steering import SteeringVector  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from steerlab.control import ControlWatcher  # noqa: E402
from steerlab.decode import Event, chat, greedy_generate, render_messages  # noqa: E402

from .conftest import Tiny, direction_entry  # noqa: E402

runner = CliRunner()


def write_atomically(path: Path, data: Any) -> None:
    """How writers are told to update the control file: temp file, then rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


# ---- events mode ----


def test_an_event_at_token_five_changes_the_next_logits_and_is_recorded(tiny: Tiny) -> None:
    prompt = tiny.prompt_ids("Tell me a story")
    with tiny.session() as session:
        plain = greedy_generate(session, prompt, max_new_tokens=10)
    events = [Event(token_index=5, control=tiny.control([direction_entry("random", 3.0)]))]
    with tiny.session() as session:
        steered = greedy_generate(session, prompt, max_new_tokens=10, events=events)
    # Before the event: nothing applied, and the same logits as the unsteered run.
    for i in range(5):
        assert steered.timeline[i].applied == []
        assert torch.equal(steered.step_logits[i], plain.step_logits[i])
    # At the event: the entry is recorded with its added norm (3 × a unit vector), and the logits move.
    at = steered.timeline[5]
    assert at.token_index == 5 and at.changed is True
    assert [entry.strength for entry in at.applied] == [3.0]
    assert at.applied[0].layers == [1]
    assert at.added_norm[1] == pytest.approx(3.0, rel=1e-5)
    assert at.residual_norm[1] > 0
    assert not torch.equal(steered.step_logits[5], plain.step_logits[5])
    assert steered.timeline[6].changed is False and steered.timeline[6].applied


def test_two_entries_naming_the_same_source_sum(tiny: Tiny) -> None:
    state = tiny.state()
    state.apply(tiny.control([direction_entry("random", 1.5), direction_entry("random", 2.5)]))
    assert torch.allclose(state.deltas[1], 4.0 * tiny.registry.directions["random"].unit)
    assert len(state.applied) == 2


# ---- the watcher mid-run ----


def test_truncated_control_file_keeps_the_previous_state_and_warns_once(tiny: Tiny, tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    write_atomically(path, tiny.control([direction_entry("random", 2.0)]))
    state = tiny.state()
    watcher = ControlWatcher(path, state)
    assert watcher.poll() is True
    assert [e.strength for e in state.applied] == [2.0]

    path.write_text("")  # a writer that truncates before writing
    assert watcher.poll() is False
    assert [e.strength for e in state.applied] == [2.0]
    warnings = [m for m in tiny.messages if "control.json" in m and "last valid" in m]
    assert len(warnings) == 1
    path.write_text("")
    later = os.stat(path).st_mtime_ns + 10**6
    os.utime(path, ns=(later, later))
    assert watcher.poll() is False
    assert len([m for m in tiny.messages if "last valid" in m]) == 1  # the same failure is reported once

    write_atomically(path, tiny.control([direction_entry("random", 5.0)]))
    assert watcher.poll() is True
    assert [e.strength for e in state.applied] == [5.0]


def test_a_file_naming_the_other_model_rejects_all_entries_with_one_message(tiny: Tiny, tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    write_atomically(path, {"model": "google/gemma-3-1b-it", "entries": [direction_entry("random", 2.0)]})
    state = tiny.state()
    watcher = ControlWatcher(path, state)
    watcher.poll()
    assert state.applied == [] and state.layers == []
    mismatch = [m for m in tiny.messages if "google/gemma-3-1b-it" in m and tiny.name in m]
    assert len(mismatch) == 1
    with tiny.session(state) as session:  # and generation continues unsteered
        result = greedy_generate(session, tiny.prompt_ids(), max_new_tokens=3)
    assert all(entry.applied == [] for entry in result.timeline)


def test_an_entry_for_an_unloaded_sae_layer_is_skipped_and_names_the_flag(tiny: Tiny) -> None:
    state = tiny.state()
    sae_entry = {
        "source": {"kind": "sae", "release": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer": 1, "feature": 7},
        "strength": 1.0,
    }
    state.apply(tiny.control([sae_entry, direction_entry("random", 1.0)]))
    assert [e.strength for e in state.applied] == [1.0] and state.applied[0].source["kind"] == "direction"
    skipped = [m for m in tiny.messages if "--sae-layer 1" in m and "feature 7" in m]
    assert len(skipped) == 1


def test_a_control_file_that_does_not_exist_yet_is_reported_once(tiny: Tiny, tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    state = tiny.state()
    watcher = ControlWatcher(path, state)
    assert watcher.poll() is False and watcher.poll() is False
    missing = [m for m in tiny.messages if "control.json" in m and "does not exist yet" in m]
    assert len(missing) == 1
    write_atomically(path, tiny.control([direction_entry("random", 1.0)]))
    assert watcher.poll() is True
    path.unlink()
    assert watcher.poll() is False
    assert any("disappeared" in m for m in tiny.messages)
    assert len([m for m in tiny.messages if "does not exist yet" in m]) == 1  # not repeated after it vanished


class Exploding:
    """A resolver that raises something other than SourceUnavailableError (the device-mismatch class of bug)."""

    def __call__(self, source: Any) -> Any:
        raise RuntimeError("Expected all tensors to be on the same device")


def test_a_resolver_crash_keeps_the_last_valid_state_in_both_the_watcher_and_events(tiny: Tiny, tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    write_atomically(path, tiny.control([direction_entry("random", 2.0)]))
    state = tiny.state()
    watcher = ControlWatcher(path, state)
    assert watcher.poll() is True
    state.resolver = Exploding()
    write_atomically(path, tiny.control([direction_entry("random", 5.0)]))
    assert watcher.poll() is False
    assert [e.strength for e in state.applied] == [2.0]
    crashes = [m for m in tiny.messages if "same device" in m and "last valid" in m]
    assert len(crashes) == 1

    events = [Event(token_index=1, control=tiny.control([direction_entry("random", 7.0)]))]
    with tiny.session(state) as session:
        result = greedy_generate(session, tiny.prompt_ids(), max_new_tokens=3, events=events)
    assert all([e.strength for e in entry.applied] == [2.0] for entry in result.timeline)
    assert any("event at token 1" in m and "same device" in m for m in tiny.messages)


def test_a_live_file_change_takes_effect_on_the_next_token(tiny: Tiny, tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    write_atomically(path, tiny.control([]))
    state = tiny.state()
    watcher = ControlWatcher(path, state)
    prompt = tiny.prompt_ids("Live")
    seen: list[int] = []

    def on_token(index: int, text: str) -> None:
        seen.append(index)
        if index == 3:
            write_atomically(path, tiny.control([direction_entry("random", 3.0)]))

    with tiny.session(state) as session:
        result = greedy_generate(session, prompt, max_new_tokens=8, watcher=watcher, on_token=on_token)
    assert [e.applied == [] for e in result.timeline[:4]] == [True] * 4
    assert result.timeline[4].changed is True and result.timeline[4].added_norm[1] == pytest.approx(3.0, rel=1e-5)
    assert seen == list(range(len(result.token_ids)))


# ---- chat template rules ----


def test_gemma_folds_the_system_prompt_into_the_first_user_turn(tiny_gemma3: Tiny, tiny_qwen3: Tiny) -> None:
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    gemma = render_messages(tiny_gemma3.tokenizer, messages, model_type="gemma3_text")
    assert "<|im_start|>system" not in gemma
    assert gemma.index("Be brief.") < gemma.index("Hi")
    qwen = render_messages(tiny_qwen3.tokenizer, messages, model_type="qwen3")
    assert qwen.startswith("<|im_start|>system\nBe brief.")


class RecordingTokenizer:
    """Records the keyword arguments `render_messages` passes to the chat template."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "rendered"


def test_render_messages_takes_the_thinking_rule_from_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """One rule, one place: decode and directions both defer to `defaults.template_kwargs_for`."""
    from steerlab import defaults, directions

    tokenizer = RecordingTokenizer()
    render_messages(tokenizer, [{"role": "user", "content": "Hi"}], model_type="qwen3")
    assert tokenizer.kwargs.get("enable_thinking") is False
    render_messages(tokenizer, [{"role": "user", "content": "Hi"}], model_type="gemma3_text")
    assert "enable_thinking" not in tokenizer.kwargs

    monkeypatch.setattr(defaults, "template_kwargs_for", lambda name: {"marker": name})
    render_messages(tokenizer, [{"role": "user", "content": "Hi"}], model_type="qwen3")
    assert tokenizer.kwargs.get("marker") == "qwen3"
    assert directions.template_kwargs_for("Qwen/Qwen3-1.7B") == {"marker": "Qwen/Qwen3-1.7B"}


# ---- the CLI: generate --json, chat ----


@pytest.fixture
def saved_tiny(tiny_qwen3: Tiny, tmp_path: Path) -> tuple[Path, Path]:
    """The tiny Qwen3-shaped model on disk (so `--model <path>` loads it) and a saved random unit direction."""
    model_dir = tmp_path / "tiny-model"
    tiny_qwen3.model.save_pretrained(model_dir)
    tiny_qwen3.tokenizer.save_pretrained(model_dir)
    unit = tiny_qwen3.registry.directions["random"].unit.numpy().astype("float64")
    SteeringVector(unit, layer=1).save(tmp_path / "random")
    return model_dir, tmp_path / "random.npz"


def test_generate_json_returns_text_and_timeline(saved_tiny: tuple[Path, Path], tmp_path: Path) -> None:
    from steerlab.cli_engine import app

    model_dir, direction = saved_tiny
    events = tmp_path / "events.json"
    control = {"model": str(model_dir), "entries": [direction_entry("random", 3.0)]}
    events.write_text(json.dumps([{"token_index": 2, "control": control}]))
    result = runner.invoke(
        app,
        [
            "generate",
            "Hello",
            "--model",
            str(model_dir),
            "--direction",
            f"random={direction}",
            "--events",
            str(events),
            "--max-new-tokens",
            "6",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload) >= {"model", "text", "token_ids", "timeline"}
    assert isinstance(payload["text"], str)
    timeline = payload["timeline"]
    assert len(timeline) == len(payload["token_ids"]) <= 6
    assert timeline[0]["applied"] == [] and timeline[2]["changed"] is True
    assert timeline[2]["applied"][0]["source"] == {"kind": "direction", "name": "random"}
    assert timeline[2]["added_norm"]["1"] == pytest.approx(3.0, rel=1e-4)
    assert not any(message.startswith("loaded ") for message in payload["messages"])  # progress goes to stderr


def test_generate_without_json_streams_text_and_a_change_marker(saved_tiny: tuple[Path, Path], tmp_path: Path) -> None:
    from steerlab.cli_engine import app

    model_dir, direction = saved_tiny
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"model": str(model_dir), "entries": [direction_entry("random", 2.0)]}))
    args = ["generate", "Hello", "--model", str(model_dir), "--direction", f"random={direction}"]
    result = runner.invoke(app, [*args, "--control", str(control)])
    assert result.exit_code == 0, result.output
    assert "token 0" in result.output and "strength=2.0" in result.output and "[1]" in result.output


def test_a_direction_from_another_model_is_rejected_before_generation(
    saved_tiny: tuple[Path, Path], tmp_path: Path
) -> None:
    from steerlab.cli_engine import app

    model_dir, _ = saved_tiny
    wide = torch.randn(16, generator=torch.Generator().manual_seed(3)).double().numpy()
    SteeringVector(wide, layer=1, metadata={"model": "other/model-16"}).save(tmp_path / "wide")
    args = ["generate", "Hello", "--model", str(model_dir), "--direction", f"wide={tmp_path / 'wide.npz'}"]
    result = runner.invoke(app, [*args, "--max-new-tokens", "2"])
    assert result.exit_code == 2, result.output
    assert "other/model-16" in result.output and "16" in result.output and "32" in result.output


def test_a_direction_keeps_its_class_gap_as_the_scale(saved_tiny: tuple[Path, Path], tmp_path: Path) -> None:
    """Strength is in calibrated units, so a saved class gap of 5 at strength 2 adds norm 10."""
    from steerlab.cli_engine import app

    model_dir, _ = saved_tiny
    unit = torch.randn(32, generator=torch.Generator().manual_seed(4))
    SteeringVector((unit / unit.norm()).double().numpy(), layer=1, metadata={"class_gap": 5.0}).save(tmp_path / "gap")
    events = tmp_path / "events.json"
    control = {"model": str(model_dir), "entries": [direction_entry("gap", 2.0)]}
    events.write_text(json.dumps([{"token_index": 0, "control": control}]))
    args = ["generate", "Hello", "--model", str(model_dir), "--direction", f"gap={tmp_path / 'gap.npz'}"]
    result = runner.invoke(app, [*args, "--events", str(events), "--max-new-tokens", "3", "--json"])
    assert result.exit_code == 0, result.output
    timeline = json.loads(result.stdout)["timeline"]
    assert timeline[0]["applied"][0]["scale"] == 5.0
    assert timeline[1]["added_norm"]["1"] == pytest.approx(10.0, rel=1e-4)  # token 0 is the unsteered prefill


def test_generate_json_carries_the_engine_messages(saved_tiny: tuple[Path, Path], tmp_path: Path) -> None:
    """An agent reading the JSON must see why its entries were ignored, not just unsteered text with exit 0."""
    from steerlab.cli_engine import app

    model_dir, direction = saved_tiny
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"model": "google/gemma-3-1b-it", "entries": [direction_entry("random", 2.0)]}))
    args = ["generate", "Hello", "--model", str(model_dir), "--direction", f"random={direction}"]
    result = runner.invoke(app, [*args, "--control", str(control), "--max-new-tokens", "3", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert all(entry["applied"] == [] for entry in payload["timeline"])
    mismatch = [m for m in payload["messages"] if "google/gemma-3-1b-it" in m and str(model_dir) in m]
    assert len(mismatch) == 1


def test_generate_with_a_missing_control_file_exits_two(saved_tiny: tuple[Path, Path], tmp_path: Path) -> None:
    from steerlab.cli_engine import app

    model_dir, _ = saved_tiny
    missing = tmp_path / "nope.json"
    result = runner.invoke(app, ["generate", "Hello", "--model", str(model_dir), "--control", str(missing), "--json"])
    assert result.exit_code == 2, result.output
    assert "nope.json" in result.output


def test_chat_loop_restarts_a_reply_and_quits(tiny_qwen3: Tiny) -> None:
    out: list[str] = []
    lines = iter(["Hello there", "r", "Second turn", "q"])
    with tiny_qwen3.session() as session:
        history = chat(session, input_fn=lambda _prompt: next(lines), write=out.append, max_new_tokens=4)
    text = "".join(out)
    assert "restart" in text.lower()
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "user", "assistant"]  # the restarted reply replaced the first one


def test_chat_loop_clear_starts_a_fresh_conversation(tiny_qwen3: Tiny) -> None:
    out: list[str] = []
    lines = iter(["Hello there", "c", "r", "New question", "q"])
    with tiny_qwen3.session() as session:
        history = chat(session, input_fn=lambda _prompt: next(lines), write=out.append, max_new_tokens=4)
    text = "".join(out)
    assert "history cleared" in text
    assert "nothing to restart yet" in text  # `r` after `c` has no reply to redo
    assert [m["content"] for m in history if m["role"] == "user"] == ["New question"]


def test_chat_cli_reads_stdin(saved_tiny: tuple[Path, Path]) -> None:
    from steerlab.cli_engine import app

    model_dir, _ = saved_tiny
    result = runner.invoke(app, ["chat", "--model", str(model_dir), "--max-new-tokens", "3"], input="Hi\nq\n")
    assert result.exit_code == 0, result.output
    assert "you>" in result.output


# ---- the real model (slow; needs Qwen/Qwen3-1.7B in the Hub cache) ----


@pytest.fixture
def real_qwen_direction(tmp_path: Path) -> Path:
    """A fixed random unit direction in Qwen3-1.7B's residual space (d_model 2048) at the default layer."""
    from steerlab import defaults, hub

    if hub.cached_bytes(defaults.QWEN_MODEL) is None:
        pytest.skip(f"{defaults.QWEN_MODEL} is not in the Hub cache; download it first (`steerlab doctor` shows sizes)")
    g = torch.Generator().manual_seed(2024)
    vector = torch.randn(2048, generator=g)
    SteeringVector((vector / vector.norm()).double().numpy(), layer=defaults.QWEN_SCOPE_LAYER).save(tmp_path / "rnd")
    return tmp_path / "rnd.npz"


@pytest.mark.slow
def test_generate_json_on_real_qwen3_records_the_event(real_qwen_direction: Path, tmp_path: Path) -> None:
    from steerlab import defaults
    from steerlab.cli_engine import app

    layer = defaults.QWEN_SCOPE_LAYER
    control = {"model": defaults.QWEN_MODEL, "entries": [direction_entry("rnd", 3.0, layers=[layer])]}
    events = tmp_path / "events.json"
    events.write_text(json.dumps([{"token_index": 5, "control": control}]))
    result = runner.invoke(
        app,
        [
            "generate",
            "Tell me about the city of Paris in two sentences.",
            "--model",
            "qwen",
            "--direction",
            f"rnd={real_qwen_direction}",
            "--events",
            str(events),
            "--max-new-tokens",
            "24",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # stays valid JSON: engine messages go to stderr
    assert payload["model"] == defaults.QWEN_MODEL
    assert "<think>" not in payload["text"]  # thinking is off
    timeline = payload["timeline"]
    assert len(timeline) == len(payload["token_ids"]) > 5
    assert all(entry["applied"] == [] for entry in timeline[:5])
    assert timeline[5]["changed"] is True
    assert timeline[5]["applied"][0]["layers"] == [layer]
    assert timeline[5]["added_norm"][str(layer)] == pytest.approx(3.0, rel=1e-2)  # bfloat16 on MPS
    assert timeline[5]["residual_norm"][str(layer)] > 0


@pytest.mark.slow
def test_chat_cli_on_real_qwen3(real_qwen_direction: Path) -> None:
    from steerlab.cli_engine import app

    result = runner.invoke(app, ["chat", "--model", "qwen", "--max-new-tokens", "8"], input="Say hello.\nq\n")
    assert result.exit_code == 0, result.output
    assert "model>" in result.output and "<think>" not in result.output
