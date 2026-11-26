"""FIXED. Runs one step of a recipe inside the sandbox and writes its output as JSON.

    python boot.py 3 run_recipe.py build  <recipe_dir> <pools_dir> <out_dir> <seed>
    python boot.py 3 run_recipe.py cutoff <recipe_dir> <cal.json> <out_dir>

build: recipe.build(ctx) -> {"rows": [...], "train": {"lr", "epochs"}, "note": str}
cutoff: recipe.fit_cutoff(cal) -> float
"""

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path


def load(recipe_dir: str):
    spec = importlib.util.spec_from_file_location("recipe", str(Path(recipe_dir) / "recipe.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, recipe_dir)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    step, recipe_dir = sys.argv[1], sys.argv[2]
    out = Path(sys.argv[4])
    t0 = time.time()
    try:
        mod = load(recipe_dir)
        if step == "build":
            res = mod.build({"data": sys.argv[3], "seed": int(sys.argv[5])})
            if not isinstance(res, dict):
                raise TypeError(f"build() returned {type(res).__name__}, not dict")
            payload = {"ok": True, "result": res}
        else:
            cal = json.loads(Path(sys.argv[3]).read_text())
            payload = {"ok": True, "result": float(mod.fit_cutoff(cal))}
    except Exception as e:  # noqa: BLE001 - every recipe error is reported back, not raised
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
    payload["seconds"] = round(time.time() - t0, 2)
    (out / f"{step}.json").write_text(json.dumps(payload))


main()
