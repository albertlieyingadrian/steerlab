"""Shared test setup for the whole steerlab suite.

The CLI tests read Typer's error messages. On CI the terminal is narrower and coloured, so Typer draws its error box
with colour codes and wraps the text, which splits the words a test looks for. Every test therefore runs with plain,
wide output, the same as a local terminal, so a result never depends on where the suite runs.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _plain_wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")
    # The inspect Workbench room fits its bundled probes in a background thread on startup; tests that build the
    # inspect app with the default cache must not write probes into ~/.cache. Workbench tests turn it back on.
    monkeypatch.setenv("STEERLAB_WORKBENCH_AUTOFIT", "0")
