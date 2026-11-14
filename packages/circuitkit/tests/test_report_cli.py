"""HTML reports render the key facts, and the CLI runs end to end on the toy model and the bundled graphs."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from circuitkit import __version__
from circuitkit.cli import app
from circuitkit.diff import diff_graphs, load_summaries
from circuitkit.labels import ScriptedLLM, calibration, label_features
from circuitkit.report import render_diff, render_labels, render_validate, save_html
from circuitkit.validate import logit_diff, validate
from circuitkit.validate.toy import demo, demo_nodes

from .test_labels import _feature, oracle

runner = CliRunner()


def test_render_validate(tmp_path) -> None:
    b = demo()
    rep = validate(
        b,
        "dallas",
        demo_nodes(b),
        metric=logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento")),
        counterfactual="oakland",
        n_controls=5,
    )
    page = render_validate(rep)
    assert "<!doctype html>" in page and "switch" in page and "readout" in page and "Dark mode" in page
    assert save_html(page, tmp_path / "v.html").exists()


def test_render_diff(data_dir) -> None:
    g = load_summaries(data_dir / "rl_hack_graphs.json.gz")
    page = render_diff(diff_graphs(g["installed/before"], g["installed/after"]))
    assert "21:154092" in page and "trained on the base model" in page


def test_render_labels() -> None:
    labs = label_features(ScriptedLLM(oracle), [_feature()])
    page = render_labels(labs, calibration(labs))
    assert "the word Paris" in page and "Calibration" in page


def test_cli_version() -> None:
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and __version__ in r.output


def test_cli_validate_demo(tmp_path) -> None:
    r = runner.invoke(
        app,
        [
            "validate",
            "--demo",
            "--n-controls",
            "20",
            "--json",
            str(tmp_path / "v.json"),
            "--html",
            str(tmp_path / "v.html"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "switch" in r.output
    assert json.loads((tmp_path / "v.json").read_text())["groups"]


def test_cli_validate_needs_inputs() -> None:
    r = runner.invoke(app, ["validate"])
    assert r.exit_code != 0


def test_cli_diff(data_dir, tmp_path) -> None:
    r = runner.invoke(
        app,
        [
            "diff",
            str(data_dir / "rl_hack_graphs.json.gz"),
            "--before",
            "installed/before",
            "--after",
            "installed/after",
            "--html",
            str(tmp_path / "d.html"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "21:154092" in r.output and "validity: ok" in r.output
    assert (tmp_path / "d.html").exists()


def test_cli_label_records(tmp_path, monkeypatch) -> None:
    import circuitkit.labels as labels_mod

    feats = tmp_path / "f.json"
    feats.write_text(json.dumps([_feature().model_dump()]))
    monkeypatch.setattr(labels_mod, "make_llm", lambda *a, **k: ScriptedLLM(oracle))
    r = runner.invoke(app, ["label", "--records", str(feats), "--json", str(tmp_path / "l.json")])
    assert r.exit_code == 0, r.output
    assert "the word Paris" in r.output
