"""Package-level checks: the version comes from the installed metadata, and the base install stays light."""

import importlib.metadata
import importlib.util
import json
import subprocess
import sys

import sandbagkit

HEAVY_MODULES = ["torch", "transformers", "sklearn", "peft"]
SUBPACKAGES = ["sandbagkit.core", "sandbagkit.probing", "sandbagkit.backends"]


def test_version_comes_from_metadata() -> None:
    assert sandbagkit.__version__ == importlib.metadata.version("sandbagkit")


def test_base_import_does_not_load_heavy_dependencies() -> None:
    # A fresh interpreter, because this test session may already have imported torch elsewhere.
    present = [name for name in SUBPACKAGES if importlib.util.find_spec(name) is not None]
    code = (
        "import sys, json, importlib\n"
        f"for name in ['sandbagkit', *{present!r}]: importlib.import_module(name)\n"
        f"print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
