"""The control file schema: pydantic models, the published JSON Schema, `schema control` and `control validate`.

No torch: the control module must import on the light base install so the schema can be printed without a model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from steerlab import control
from steerlab.cli_engine import app

runner = CliRunner()

EXAMPLE: dict[str, Any] = {
    "model": "Qwen/Qwen3-1.7B",
    "entries": [
        {
            "source": {
                "kind": "sae",
                "release": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
                "layer": 14,
                "feature": 1234,
            },
            "strength": 2.0,
            "layers": [14],
        },
        {"source": {"kind": "direction", "name": "chinese"}, "strength": -1.0},
    ],
}


def test_the_example_parses_and_layers_default_to_none() -> None:
    parsed = control.ControlFile.model_validate(EXAMPLE)
    assert parsed.model == "Qwen/Qwen3-1.7B"
    assert isinstance(parsed.entries[0].source, control.SAESource) and parsed.entries[0].layers == [14]
    assert isinstance(parsed.entries[1].source, control.DirectionSource) and parsed.entries[1].layers is None


def test_entries_may_be_omitted_meaning_no_steering() -> None:
    assert control.parse_control('{"model": "m"}').entries == []


@pytest.mark.parametrize(
    "bad",
    [
        {"model": "m", "entries": [{"source": {"kind": "sae", "release": "r", "layer": 1}, "strength": 1}]},
        {"model": "m", "entries": [{"source": {"kind": "direction"}, "strength": 1}]},
        {"model": "m", "entries": [{"source": {"kind": "other", "name": "x"}, "strength": 1}]},
        {"model": "m", "entries": [{"source": {"kind": "direction", "name": "x"}}]},
        {"model": "m", "entries": [{"source": {"kind": "direction", "name": "x"}, "strength": "two"}]},
        {"model": "m", "entries": [], "extra": 1},
        {"entries": []},
    ],
)
def test_malformed_files_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(control.ControlValidationError):
        control.parse_control(json.dumps(bad))


def test_truncated_json_is_a_validation_error_too() -> None:
    with pytest.raises(control.ControlValidationError):
        control.parse_control("")


def test_the_committed_schema_matches_the_models() -> None:
    # The drift guard needs nothing beyond pydantic, so it must never sit behind an importorskip: CI's dev group has no
    # jsonschema, and a skipped guard would let the committed file fall out of date silently.
    committed = json.loads(control.SCHEMA_PATH.read_text())
    assert committed == control.control_json_schema(), "regenerate src/steerlab/schemas/control.schema.json"


def test_the_committed_schema_validates_the_example() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    committed = json.loads(control.SCHEMA_PATH.read_text())
    jsonschema.validate(EXAMPLE, committed)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"model": "m", "entries": [{"source": {"kind": "direction"}, "strength": 1}]}, committed)


def test_schema_control_prints_the_schema() -> None:
    result = runner.invoke(app, ["schema", "control"])
    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert printed == control.control_json_schema()
    assert "rename" in json.dumps(printed).lower()  # writers are told to write via temp file and rename


def test_control_validate_reports_a_bad_layer_index(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(EXAMPLE))
    bad["entries"][0]["layers"] = [99]
    path = tmp_path / "control.json"
    path.write_text(json.dumps(bad))
    result = runner.invoke(app, ["control", "validate", str(path)])
    assert result.exit_code == 1, result.output
    assert "layer 99" in result.output and "0..27" in result.output

    good = tmp_path / "good.json"
    good.write_text(json.dumps(EXAMPLE))
    ok = runner.invoke(app, ["control", "validate", str(good)])
    assert ok.exit_code == 0, ok.output
    assert "ok" in ok.output.lower() and "2 entries" in ok.output


def test_control_validate_reports_json_errors(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text('{"model": ')
    result = runner.invoke(app, ["control", "validate", str(path)])
    assert result.exit_code == 1
    assert "control.json" in result.output


def test_control_and_cli_modules_import_without_torch() -> None:
    code = "import sys, steerlab.control, steerlab.cli_engine; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_control_set_creates_the_file_and_prints_before_and_after(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    result = runner.invoke(app, ["control", "set", "chinese=1.5", "--file", str(path)])
    assert result.exit_code == 0, result.output
    assert "before: (no control.json yet)" in result.output
    written = control.load_control(path)
    assert written.model == "Qwen/Qwen3-1.7B"
    assert [(e.source.describe(), e.strength) for e in written.entries] == [("direction 'chinese'", 1.5)]
    assert f"after:  {path.read_text().strip()}" in result.output
    assert not (tmp_path / "control.json.tmp").exists()


def test_control_set_moves_one_fader_and_keeps_the_others(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps(EXAMPLE))
    result = runner.invoke(app, ["control", "set", "chinese=0.5", "refusal=1", "--file", str(path)])
    assert result.exit_code == 0, result.output
    written = control.load_control(path)
    assert [e.strength for e in written.entries] == [2.0, 0.5, 1.0]
    assert written.entries[0].layers == [14]
    assert '"strength": -1.0' in result.output  # the before line shows the old file


def test_control_set_with_another_model_starts_over(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps(EXAMPLE))
    result = runner.invoke(app, ["control", "set", "chinese=1", "--model", "gemma", "--file", str(path)])
    assert result.exit_code == 0, result.output
    written = control.load_control(path)
    assert written.model == "google/gemma-3-1b-it"
    assert len(written.entries) == 1


@pytest.mark.parametrize("bad", ["chinese", "chinese=", "=1.5", "chinese=lots"])
def test_control_set_rejects_a_malformed_setting_without_writing(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "control.json"
    result = runner.invoke(app, ["control", "set", bad, "--file", str(path)])
    assert result.exit_code == 1
    assert "NAME=STRENGTH" in result.output
    assert not path.exists()


def test_control_off_clears_the_entries_and_keeps_the_model(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps(EXAMPLE))
    result = runner.invoke(app, ["control", "off", "--file", str(path)])
    assert result.exit_code == 0, result.output
    written = control.load_control(path)
    assert (written.model, written.entries) == ("Qwen/Qwen3-1.7B", [])


def test_control_set_names_an_sae_feature_of_the_models_own_release(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    result = runner.invoke(app, ["control", "set", "feature:24946=4", "--file", str(path)])
    assert result.exit_code == 0, result.output
    source = control.load_control(path).entries[0].source
    assert source.describe() == "sae Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 L14 feature 24946"
    result = runner.invoke(app, ["control", "set", "feature:24946=2", "--sae-layer", "20", "--file", str(path)])
    assert result.exit_code == 0, result.output
    layers = [(e.source.layer, e.strength) for e in control.load_control(path).entries if e.source.kind == "sae"]
    assert layers == [(14, 4.0), (20, 2.0)]  # another layer is another fader


def test_control_set_rejects_a_feature_that_is_not_a_number(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    result = runner.invoke(app, ["control", "set", "feature:chinese=4", "--file", str(path)])
    assert result.exit_code == 1
    assert "whole number" in result.output
    assert not path.exists()


def test_control_set_feature_on_an_extra_gemma_model_uses_its_gemma_scope_release(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    result = runner.invoke(app, ["control", "set", "feature:7=2", "--model", "gemma2-2b", "--file", str(path)])
    assert result.exit_code == 0, result.output
    written = control.load_control(path)
    assert written.model == "google/gemma-2-2b-it"
    assert written.entries[0].source.describe() == "sae google/gemma-scope-2b-pt-res L12 feature 7"


def test_control_off_can_switch_the_files_model_so_a_bigger_model_reads_it(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps(EXAMPLE))
    result = runner.invoke(app, ["control", "off", "--model", "qwen8b", "--file", str(path)])
    assert result.exit_code == 0, result.output
    assert control.load_control(path).model == "Qwen/Qwen3-8B"
    result = runner.invoke(app, ["control", "set", "chinese=1.5", "--file", str(path)])
    assert control.load_control(path).model == "Qwen/Qwen3-8B"  # set keeps the file's model
