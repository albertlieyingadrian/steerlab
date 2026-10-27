"""The library loader: load from a path or the local cache, unload, discover directions, and the model catalog.

The tiny Qwen3-shaped model is saved to a temporary folder, and for the cache tests into a fake Hugging Face cache laid
out like the real one (`models--Qwen--Qwen3-8B/snapshots/<rev>/`), so nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sandbagkit.backends.tiny import tiny_backend  # noqa: E402
from sandbagkit.probing.steering import SteeringVector  # noqa: E402

from steerlab import demo, loading  # noqa: E402
from steerlab.control import DirectionSource  # noqa: E402

QWEN8B = "Qwen/Qwen3-8B"
QWEN8B_SAE = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"


def save_tiny(folder: Path) -> Path:
    backend = tiny_backend(seed=0, variant="qwen3")
    backend.model.save_pretrained(folder)
    backend.tokenizer.save_pretrained(folder)
    return folder


def save_direction(path: Path, width: int = 32, *, metadata: dict[str, Any] | None = None) -> Path:
    unit = torch.randn(width, generator=torch.Generator().manual_seed(1))
    SteeringVector((unit / unit.norm()).double().numpy(), layer=1, metadata=metadata or {}).save(path)
    return path


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return cache


def snapshot(cache: Path, repo_id: str) -> Path:
    folder = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc123"
    folder.mkdir(parents=True)
    return folder


def test_a_saved_model_loads_with_its_direction(tmp_path: Path) -> None:
    model_dir = save_tiny(tmp_path / "model")
    direction = save_direction(tmp_path / "random")
    messages: list[str] = []
    backend, state = loading.load_session(str(model_dir), [], None, [f"random={direction}"], on_message=messages.append)
    assert state.resolver(DirectionSource(kind="direction", name="random")).layer == 1
    assert state.model == str(model_dir) and state.n_layers == backend.model.config.num_hidden_layers
    assert any("loaded direction 'random'" in message for message in messages)


def test_a_direction_of_another_width_is_a_plain_error(tmp_path: Path) -> None:
    model_dir = save_tiny(tmp_path / "model")
    wide = save_direction(tmp_path / "wide", 16, metadata={"model": "other/model-16"})
    with pytest.raises(loading.LoadError, match=r"width 16 \(built on other/model-16\).*width 32"):
        loading.load_session(str(model_dir), [], None, [f"wide={wide}"], on_message=print)


def test_local_only_loads_from_the_cache_and_refuses_what_is_not_there(fake_hub: Path) -> None:
    with pytest.raises(loading.LoadError, match="not in the Hugging Face cache"):
        loading.load_session("qwen8b", [], None, [], on_message=print, local_only=True)
    save_tiny(snapshot(fake_hub, QWEN8B))
    backend, state = loading.load_session("qwen8b", [], None, [], on_message=print, local_only=True)
    assert state.model == QWEN8B  # the Hub id, not the snapshot path, so control entries keep naming the model
    with pytest.raises(loading.LoadError, match=r"layer 18 is not in the Hugging Face cache"):
        loading.load_session("qwen8b", [18], None, [], on_message=print, local_only=True)
    loading.unload(backend, state)


def test_directions_are_the_npz_files_that_have_their_json(tmp_path: Path) -> None:
    folder = tmp_path / "qwen8b"
    folder.mkdir()
    for name in ("chinese", "golden_gate"):
        (folder / f"{name}.npz").write_bytes(b"")
        (folder / f"{name}.json").write_text("{}")
    (folder / "half.npz").write_bytes(b"")  # no .json: an interrupted build
    (folder / "notes.txt").write_text("not a direction")
    specs = loading.discover_directions(tmp_path, "qwen8b")
    assert specs == [f"chinese={folder / 'chinese'}", f"golden_gate={folder / 'golden_gate'}"]
    assert loading.discover_directions(tmp_path, "gemma2-2b") == []


def test_unload_clears_the_steering_state_and_the_weights(tmp_path: Path) -> None:
    model_dir = save_tiny(tmp_path / "model")
    direction = save_direction(tmp_path / "random")
    backend, state = loading.load_session(str(model_dir), [], None, [f"random={direction}"], on_message=print)
    entry = {"source": {"kind": "direction", "name": "random"}, "strength": 2}
    state.apply(json.dumps({"model": state.model, "entries": [entry]}))
    assert state.deltas
    loading.unload(backend, state)
    assert backend.model is None
    assert state.deltas == {} and state.applied == []
    assert state.resolver.directions == {}  # type: ignore[attr-defined]


def test_the_catalog_lists_cached_models_with_their_directions_and_sae(fake_hub: Path, tmp_path: Path) -> None:
    qwen8b = snapshot(fake_hub, QWEN8B)
    (qwen8b / "config.json").write_text("{}")
    (qwen8b / "model.safetensors").write_bytes(b"x")
    gemma = snapshot(fake_hub, "google/gemma-2-2b-it")
    (gemma / "config.json").write_text("{}")  # config only: an unfinished download is not offered
    llama = snapshot(fake_hub, "meta-llama/Llama-3.1-8B-Instruct")
    (llama / "config.json").write_text("{}")
    (llama / "model-00001-of-00004.safetensors").write_bytes(b"x")
    root = tmp_path / "directions"
    (root / "qwen8b").mkdir(parents=True)
    for name in ("chinese", "refusal"):
        (root / "qwen8b" / f"{name}.npz").write_bytes(b"")
        (root / "qwen8b" / f"{name}.json").write_text("{}")

    entries = {entry.key: entry for entry in loading.catalog(root)}
    assert set(entries) == {"qwen8b", "llama8b"}
    # The tuned directions shipped with steerlab, plus the saved concept steerlab does not ship (refusal).
    assert entries["qwen8b"].direction_names == ["chinese", "golden_gate", "refusal"]
    shipped = {spec.partition("=")[0]: spec.partition("=")[2] for spec in entries["qwen8b"].direction_specs}
    assert Path(shipped["chinese"]).parent == demo.DIRECTIONS_ROOT / "qwen8b"  # shipped wins over the saved one
    assert Path(shipped["refusal"]).parent == root / "qwen8b"
    assert entries["qwen8b"].sae is None  # the SAE file is not cached yet
    assert entries["llama8b"].direction_names == ["chinese", "golden_gate"] and entries["llama8b"].sae is None

    (snapshot(fake_hub, QWEN8B_SAE) / "layer18.sae.pt").write_bytes(b"x")
    qwen_entry = next(entry for entry in loading.catalog(root) if entry.key == "qwen8b")
    assert qwen_entry.sae == loading.SAEInfo(QWEN8B_SAE, 18)
    assert qwen_entry.to_dict() == {
        "key": "qwen8b",
        "model": QWEN8B,
        "directions": ["chinese", "golden_gate", "refusal"],
        "sae": {"release": QWEN8B_SAE, "layer": 18},
    }


def test_the_shipped_demo_set_is_complete_and_loads() -> None:
    """Every preset names a shipped prompt and shipped directions, and each direction records its tuned layer."""
    from sandbagkit.probing.steering import SteeringVector

    assert set(demo.prompts()) == {"S2", "S3", "S4"}
    assert demo.prompt_for("llama8b") == demo.prompts()["S2"] and demo.prompt_for("qwen") == demo.prompts()["S4"]
    for key, preset in demo.presets().items():
        assert preset.prompt in demo.prompts()
        for concept in preset.suggest:
            vector = SteeringVector.load(demo.DIRECTIONS_ROOT / key / concept)
            assert vector.layer is not None and vector.metadata["concept"] == concept
    assert SteeringVector.load(demo.DIRECTIONS_ROOT / "llama8b" / "chinese").layer == 15
    assert SteeringVector.load(demo.DIRECTIONS_ROOT / "llama8b" / "golden_gate").layer == 11
    llama_features = demo.presets()["llama8b"].features
    assert [(feature.id, feature.mode, feature.suggest) for feature in llama_features] == [(54543, "clamp", 3.0)]


def test_display_path_never_shows_a_full_home_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    shipped = demo.DIRECTIONS_ROOT / "qwen8b" / "chinese"
    assert loading.display_path(shipped) == "steerlab demo set: qwen8b/chinese"
    assert (
        loading.display_path(tmp_path / "outputs" / "directions" / "qwen" / "chinese")
        == "outputs/directions/qwen/chinese"
    )
    elsewhere = Path.home() / "somewhere-else" / "chinese"  # tmp_path is not under the home folder
    assert loading.display_path(elsewhere) == "~/somewhere-else/chinese"
