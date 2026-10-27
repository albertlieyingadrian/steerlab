"""Package-level smoke tests: version, CLI, the light base install, and `doctor` offline."""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import steerlab
from steerlab.cli import app

runner = CliRunner()

HEAVY_MODULES = [
    "torch",
    "transformers",
    "safetensors",
    "sae_lens",
    "transformer_lens",
    "sandbagkit",
    "fastapi",
    "uvicorn",
]

FAKE_TOKEN = "hf_FAKE_TOKEN_FOR_TESTS_do_not_print_0123456789"


def test_version_comes_from_metadata():
    assert steerlab.__version__ == importlib.metadata.version("steerlab")


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"steerlab {steerlab.__version__}"


def test_python_dash_m():
    out = subprocess.run([sys.executable, "-m", "steerlab", "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == f"steerlab {steerlab.__version__}"


LIGHT_MODULES = [
    "steerlab.cli",
    "steerlab.hub",
    "steerlab.saes.common",
    "steerlab.saes.qwen_scope",
    "steerlab.saes.gemma_scope",
    "steerlab.monitors",
    "steerlab.monitors.mcq",
    "steerlab.monitors.models",
]
"""Modules that must import without torch, transformers or the web stack."""


def test_base_import_does_not_load_heavy_dependencies():
    # A fresh interpreter, because this test session may already have imported torch elsewhere.
    # Every light module this install has (the monitors are not in every install).
    light = [m for m in LIGHT_MODULES if m.split(".")[1] != "monitors" or importlib.util.find_spec("steerlab.monitors")]
    code = (
        "import sys, json, importlib\n"
        f"for name in {light!r}: importlib.import_module(name)\n"
        f"print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []


def test_doctor_runs_on_the_base_install_without_torch(tmp_path: Path):
    # pyproject promises `doctor` works on the torch-free base install. A fresh interpreter with torch blocked
    # (`sys.modules["torch"] = None` makes every `import torch` raise) and an empty Hub cache must still exit 0.
    (tmp_path / "hub").mkdir()
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from typer.testing import CliRunner\n"
        "from steerlab.cli import app\n"
        "result = CliRunner().invoke(app, ['doctor'])\n"
        "print(result.output)\n"
        "if result.exception and not isinstance(result.exception, SystemExit):\n"
        "    raise result.exception\n"
        "sys.exit(result.exit_code)\n"
    )
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_TOKEN": FAKE_TOKEN}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    for repo_id in ("Qwen/Qwen3-1.7B", "google/gemma-3-1b-it"):
        matching = [line for line in out.stdout.splitlines() if repo_id in line]
        assert matching and "not cached" in matching[0], out.stdout
    assert "layer14.sae.pt" in out.stdout and "params.safetensors" in out.stdout
    assert 'n/a (install "steerlab[models]")' in out.stdout
    assert FAKE_TOKEN not in out.stdout


@pytest.fixture
def empty_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    hub = tmp_path / "hub"
    hub.mkdir()
    steer = tmp_path / "steerlab"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("STEERLAB_CACHE", str(steer))
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    return hub, steer


def test_doctor_with_no_cache_lists_models_as_not_cached_with_sizes(empty_caches):
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    out = result.output
    for line_start, size in [
        ("Qwen/Qwen3-1.7B", "4.0 GB"),
        ("google/gemma-3-1b-it", "2.0 GB"),
        ("Qwen/Qwen3-8B", "16.4 GB"),
        ("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "537 MB"),
        ("google/gemma-scope-2-1b-it", "151 MB"),
    ]:
        matching = [line for line in out.splitlines() if line_start in line]
        assert matching, f"{line_start} missing from doctor output:\n{out}"
        assert "not cached" in matching[0] and size in matching[0], matching[0]
    assert "judge" in out
    assert "directions" in out
    # Offline by default: Gemma access is reported as unchecked, not looked up.
    assert "--online" in out


def test_doctor_never_prints_the_token(empty_caches):
    result = runner.invoke(app, ["doctor"])
    assert FAKE_TOKEN not in result.output
    assert "hf token: present" in result.output


def test_doctor_honours_the_repo_token_variable_and_never_prints_it(empty_caches, monkeypatch: pytest.MonkeyPatch):
    # The repo's .env names the token HUGGINGFACE_TOKEN; huggingface_hub only reads HF_TOKEN.
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HF_TOKEN_PATH", str(empty_caches[0] / "no-such-token"))  # so "present" comes from the var
    monkeypatch.setenv("HUGGINGFACE_TOKEN", FAKE_TOKEN + "_repo")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert FAKE_TOKEN not in result.output
    assert "hf token: present" in result.output


def test_doctor_reports_absent_token(empty_caches, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN_PATH", str(empty_caches[0] / "no-such-token"))
    result = runner.invoke(app, ["doctor"])
    assert "hf token: absent" in result.output


def _fake_snapshot_file(hub: Path, repo_id: str, filename: str, size: int) -> None:
    """Lay out one file in the Hub cache format: blobs/<hash> + snapshots/<rev>/<filename> symlink."""
    repo_dir = hub / ("models--" + repo_id.replace("/", "--"))
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    blob = blobs / f"blob{abs(hash(filename))}"
    blob.write_bytes(b"\0" * size)
    target = repo_dir / "snapshots" / "abc123" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(blob, target)


def test_doctor_reports_cached_repos_and_sae_layers_from_the_cache_layout(empty_caches):
    hub, _ = empty_caches
    _fake_snapshot_file(hub, "Qwen/Qwen3-1.7B", "model.safetensors", 3 * 1024 * 1024)
    _fake_snapshot_file(hub, "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer14.sae.pt", 2 * 1024 * 1024)
    _fake_snapshot_file(hub, "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer3.sae.pt", 1024 * 1024)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    qwen = next(line for line in lines if line.strip().startswith("Qwen/Qwen3-1.7B "))
    assert "cached" in qwen and "not cached" not in qwen and "3.0 MB" in qwen
    sae = next(line for line in lines if "layer14.sae.pt" in line)
    assert "cached" in sae and "not cached" not in sae and "2.0 MB" in sae
    assert "layer3.sae.pt" in result.output  # other cached layers of the release are listed too


def _fake_direction(out_dir: Path, model_key: str, concept: str) -> None:
    """Lay out one saved direction as sandbagkit's `SteeringVector.save` does: `<model>/<concept>.npz` + `.json`."""
    base = out_dir / model_key / concept
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".npz").write_bytes(b"PK\x05\x06" + b"\0" * 18)  # an empty zip; doctor never opens it
    description = {"kind": "steering_vector", "metadata": {"class_gap": 12.3, "cosine_agreement": 0.81, "d_prime": 3.2}}
    base.with_suffix(".json").write_text(json.dumps(description), encoding="utf-8")


def test_doctor_reports_directions_from_where_directions_build_writes_them(
    empty_caches, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # `directions build` writes to directions.DEFAULT_OUTPUT_DIR (relative to the repo root the lessons run from), so
    # that is where doctor must look: per model key, present or missing per concept.
    from steerlab import directions

    monkeypatch.chdir(tmp_path)
    _fake_direction(tmp_path / directions.DEFAULT_OUTPUT_DIR, "qwen", "chinese")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert any(str(directions.DEFAULT_OUTPUT_DIR) in line for line in lines), result.output
    chinese = next(line for line in lines if "qwen" in line and "chinese" in line)
    assert "present" in chinese and "missing" not in chinese, chinese
    refusal = next(line for line in lines if "qwen" in line and "refusal" in line)
    assert "missing" in refusal, refusal
    gemma_chinese = next(line for line in lines if "gemma" in line and "chinese" in line)
    assert "missing" in gemma_chinese, gemma_chinese


def test_core_sub_commands_are_registered_and_optional_ones_when_their_module_is_installed():
    from steerlab import cli

    names = {info.name for info in app.registered_groups}
    assert {"schema", "control", "directions"} <= names
    for module_name, _attribute, command, _help in cli.OPTIONAL_COMMANDS:
        if command is not None:
            assert (command in names) == (cli.optional_module(module_name) is not None), command


def test_an_absent_sub_command_module_is_skipped_but_a_broken_one_raises(monkeypatch: pytest.MonkeyPatch):
    from steerlab import cli

    assert cli.optional_module("steerlab.cli_does_not_exist") is None

    def broken(name: str):
        raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    monkeypatch.setattr(cli.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        cli.optional_module("steerlab.cli_web")


def test_directions_dir_defaults_to_outputs_and_follows_the_environment(monkeypatch: pytest.MonkeyPatch):
    from steerlab import defaults

    monkeypatch.delenv(defaults.DIRECTIONS_ENV, raising=False)
    assert defaults.directions_dir() == Path("outputs/directions")
    monkeypatch.setenv(defaults.DIRECTIONS_ENV, "elsewhere/dirs")
    assert defaults.directions_dir() == Path("elsewhere/dirs")
