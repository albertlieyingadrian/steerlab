"""`steerlab serve` setup: what it checks before starting, the LAN warning, and the system prompt file."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
import uvicorn
from typer.testing import CliRunner

from steerlab import cli_web, loading
from steerlab.cli import app
from steerlab.web import app as web_app

from .conftest import CATALOG

runner = CliRunner()


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Stub the catalog and uvicorn: record what `serve` would have run."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(loading, "catalog", lambda root: list(CATALOG))
    monkeypatch.setattr(cli_web, "port_free", lambda host, port: True)

    def fake_run(application: Any, **kwargs: Any) -> None:
        seen["app"], seen["kwargs"] = application, kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    create = web_app.create_app

    def capture(worker: Any, **kwargs: Any) -> Any:
        seen["worker"], seen["hosts"], seen["admin_token"] = worker, kwargs["hosts"], kwargs.get("admin_token")
        return create(worker, **kwargs)

    monkeypatch.setattr(web_app, "create_app", capture)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Reply in English only.\n\n  ")
    seen["prompt_file"] = prompt
    return seen


def test_serve_reads_the_prompt_file_without_trailing_whitespace(started: dict[str, Any]) -> None:
    result = runner.invoke(app, ["serve", "--model", "tiny", "--system-file", str(started["prompt_file"])])
    assert result.exit_code == 0, result.output
    assert started["worker"].state()["prompt"] == "Reply in English only."
    assert started["hosts"] == ["localhost", "127.0.0.1", "[::1]"]
    assert started["kwargs"]["host"] == "127.0.0.1" and started["kwargs"]["port"] == 8765
    assert "admin: http://localhost:8765/admin" in result.output
    assert "?token=" not in result.output and started["admin_token"] is None  # loopback: no token by default


def test_serve_on_the_network_warns_that_admin_is_open(started: dict[str, Any]) -> None:
    args = ["serve", "--model", "tiny", "--system-file", str(started["prompt_file"]), "--host", "0.0.0.0"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "/admin needs the token" in result.output and "?token=" in result.output
    assert len(started["hosts"]) > 3 and "0.0.0.0" not in started["hosts"] and "*" not in started["hosts"]
    assert started["admin_token"] and started["admin_token"] in result.output


def test_serve_refuses_a_model_that_is_not_downloaded(started: dict[str, Any]) -> None:
    result = runner.invoke(app, ["serve", "--model", "llama8b", "--system-file", str(started["prompt_file"])])
    assert result.exit_code == 2
    assert "not downloaded" in result.output and "tiny" in result.output and "app" not in started


def test_a_port_in_use_is_detected() -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert not cli_web.port_free("127.0.0.1", port)


def test_serve_help_needs_no_server_imports() -> None:
    """A fresh interpreter: `serve --help` works and leaves FastAPI and uvicorn unimported."""
    code = (
        "import sys, json\n"
        "from typer.testing import CliRunner\n"
        "from steerlab.cli import app\n"
        "result = CliRunner().invoke(app, ['serve', '--help'])\n"
        "print(json.dumps([result.exit_code, '--system-file' in result.output, "
        "[m for m in ('fastapi', 'uvicorn') if m in sys.modules]]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [0, True, []]


def test_serve_without_a_prompt_file_uses_the_shipped_english_only_prompt(started: dict[str, Any]) -> None:
    from steerlab import demo

    result = runner.invoke(app, ["serve", "--model", "tiny"])
    assert result.exit_code == 0, result.output
    assert started["worker"].state()["prompt"] == demo.prompts()["S4"]


def test_the_demo_prompt_file_is_the_shipped_s4() -> None:
    """`docs/demos/english-only-system-prompt.txt` (used by `chat --system`) must not drift from steerlab's S4."""
    from steerlab import demo

    path = Path(__file__).resolve().parents[4] / "docs" / "demos" / "english-only-system-prompt.txt"
    if not path.exists():
        pytest.skip("the lab's demo prompt file is not part of this checkout")
    text = path.read_text(encoding="utf-8").rstrip()
    assert text == demo.prompts()["S4"]
