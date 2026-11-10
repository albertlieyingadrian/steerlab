"""The static explorer: `export` writes a page marked to read data.json, and data.json holds every saved view.
Also `steerlab inspect --export`, with the model load stubbed by the tiny inspector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from steerlab.cli import app
from steerlab.inspect import core, export
from steerlab.inspect.core import Inspector

PRESETS = [{"id": "a", "label": "A", "text": "hello wor", "note": "first"}, {"id": "b", "label": "B", "text": "abc ab"}]


def test_rounded_keeps_four_significant_figures() -> None:
    assert export.rounded({"x": [1.234567, 1.234567e-5, 3], "y": "s"}) == {"x": [1.235, 1.235e-05, 3], "y": "s"}


def test_export_writes_the_page_and_every_saved_view(inspector: Inspector, tmp_path: Path) -> None:
    path = export.export(inspector, tmp_path, PRESETS, on_message=lambda _: None)
    page = (tmp_path / "index.html").read_text()
    assert '<html lang="en" data-source="data.json">' in page
    assert (tmp_path / "static" / "inspect.js").exists() and (tmp_path / "static" / "inspect.css").exists()
    data = json.loads(path.read_text())
    assert data["model"] == "tiny-random-qwen2" and [p["id"] for p in data["presets"]] == ["a", "b"]
    preset = data["presets"][0]
    assert set(preset["layers"]) == {"0", "1"}
    listed = {row["id"] for view in preset["layers"].values() for row in view["toward"] + view["against"]}
    assert listed <= set(preset["neurons"])  # every neuron the page lists can be opened
    assert len(preset["edits"]) == 7 and preset["edits"][-1]["labels"][0].endswith("off")
    top = export.strongest(preset["layers"], 1)[0]
    assert top["id"] in preset["doses"] and preset["find"]["word"]["id"] == preset["run"]["target"]["id"]


def test_steerlab_inspect_export(inspector: Inspector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core.Inspector, "load", classmethod(lambda cls, model, local_only=True: inspector))
    monkeypatch.setattr(export.export, "__defaults__", (PRESETS[:1],))
    result = CliRunner().invoke(app, ["inspect", "--export", str(tmp_path / "site")])
    assert result.exit_code == 0, result.output
    assert "wrote" in result.output and (tmp_path / "site" / "data.json").exists()
