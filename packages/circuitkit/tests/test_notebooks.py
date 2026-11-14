"""The example notebooks run top to bottom, offline (toy model, bundled graphs, cached model replies)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

EX = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = sorted(EX.glob("0*.ipynb"))


def test_three_notebooks() -> None:
    assert [p.name for p in NOTEBOOKS] == ["01_validate_toy.ipynb", "02_graph_diff.ipynb", "03_labels.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs(path: Path, tmp_path: Path) -> None:
    (tmp_path / "data").symlink_to(EX / "data")
    (tmp_path / "reports").mkdir()
    shutil.copy(path, tmp_path / path.name)
    nb = nbformat.read(tmp_path / path.name, as_version=4)
    nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    ).execute()
    errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors
