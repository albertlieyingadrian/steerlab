"""Every code block in docs-site/ is tested.

- ```python blocks run in a fresh interpreter, in a scratch folder where `examples/` points at this package's
  examples (offline: toy model, bundled graphs, cached model replies).
- ```bash blocks: each `circuitkit ...` command runs the same way.
- Blocks marked `gpu` need a GPU and circuit-tracer: they are compiled, and every name imported from circuitkit must
  exist. `examples/modal/live_checks.py` runs the same calls on real models.
- Blocks marked `signature` must match the real function's parameters.
- Blocks marked `install` or `network` are not run (package installs and live API calls).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
SITE = PKG.parents[1] / "docs-site"
FENCE = re.compile(r"^([ \t]*)```(\w+)([^\n]*)\n(.*?)^\1```[ \t]*$", re.DOTALL | re.MULTILINE)


def blocks() -> list[tuple[str, str, str, str]]:
    out = []
    for page in sorted(SITE.rglob("*.mdx")):
        text = page.read_text(encoding="utf-8")
        for i, m in enumerate(FENCE.finditer(text)):
            indent, lang, meta, body = m.groups()
            body = "\n".join(line[len(indent) :] if line.startswith(indent) else line for line in body.splitlines())
            out.append((f"{page.relative_to(SITE)}#{i}", lang, meta.strip(), body))
    return out


ALL = blocks()


def test_docs_have_snippets() -> None:
    assert len(ALL) > 20


def _scratch(tmp_path: Path) -> Path:
    (tmp_path / "examples").symlink_to(PKG / "examples")
    return tmp_path


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)


def _check_imports(body: str) -> None:
    for node in ast.walk(ast.parse(body)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("circuitkit"):
            mod = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(mod, alias.name), f"{node.module} has no {alias.name}"


@pytest.mark.parametrize(("where", "lang", "meta", "body"), ALL, ids=[b[0] for b in ALL])
def test_snippet(where: str, lang: str, meta: str, body: str, tmp_path: Path) -> None:
    words = set(meta.lower().split())
    if words & {"install", "network"}:
        pytest.skip(f"{where}: {meta} block, not run")
    if lang == "python" and "signature" in words:
        _check_signature(body)
        return
    if lang == "python" and "gpu" in words:
        compile(body, where, "exec")
        _check_imports(body)
        return
    if lang == "python":
        r = _run([sys.executable, "-c", body], _scratch(tmp_path))
        assert r.returncode == 0, f"{where} failed:\n{r.stderr[-2000:]}"
        return
    if lang == "bash":
        cmds = [c for c in body.replace("\\\n", " ").splitlines() if c.strip().startswith("circuitkit")]
        if "gpu" in words:
            for c in cmds:
                sub = shlex.split(c)[1]
                r = _run([sys.executable, "-m", "circuitkit.cli", sub, "--help"], PKG)
                assert r.returncode == 0, f"{where}: `circuitkit {sub}` is not a command"
            return
        cwd = _scratch(tmp_path)
        exe = Path(sys.executable).with_name("circuitkit")
        for c in cmds:
            argv = shlex.split(c.split("#")[0])
            r = _run([str(exe), *argv[1:]], cwd)
            assert r.returncode == 0, f"{where}: `{c}` failed:\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}"
        return
    pytest.skip(f"{where}: {lang} block")


def _check_signature(body: str) -> None:
    text = " ".join(body.split())
    for m in re.finditer(r"(\w+)\((.*?)\)\s*->", text):
        name, params = m.group(1), m.group(2)
        fn = None
        for mod in ("circuitkit.diff", "circuitkit.validate", "circuitkit.labels"):
            fn = getattr(importlib.import_module(mod), name, None) or fn
        assert fn is not None, f"{name} not found"
        documented = [p.split("=")[0].strip().split(":")[0].strip("* ") for p in params.split(",")]
        documented = [p for p in documented if p and p != "kw"]
        real = [p for p in inspect.signature(fn).parameters if p not in ("kw",)]
        assert documented == real, f"{name}: docs {documented} != code {real}"
