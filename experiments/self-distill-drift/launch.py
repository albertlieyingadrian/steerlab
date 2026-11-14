"""Spawn, watch and collect the self-distillation chains. COSTS MONEY when it spawns (Modal, profile albertlieyingadrian).

MODAL_PROFILE=albertlieyingadrian python launch.py smoke     # 1 chain, 64 prompts, 1 generation
MODAL_PROFILE=albertlieyingadrian python launch.py run       # control and seeded, seeds 0-2
MODAL_PROFILE=albertlieyingadrian python launch.py collect   # volume runs/main -> results/runs/
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CHAINS = [(c, s) for c in ("control", "seeded") for s in (0, 1, 2)]


def main():
    cmd = sys.argv[1]
    f = modal.Function.from_name("self-distill-drift", "chain")
    if cmd == "smoke":
        print(f.remote("seeded", 0, n_train=64, gens=1, tag="smoke"))
    elif cmd == "run":
        calls = {f"{c}-s{s}": f.spawn(c, s).object_id for c, s in CHAINS}
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "calls.json").write_text(json.dumps(calls, indent=1))
        print(calls)
    elif cmd == "collect":
        tmp = HERE / "results" / "_dl"
        subprocess.run(["modal", "volume", "get", "--force", "self-distill-drift", "runs/main", str(tmp)], check=True)
        out = HERE / "results" / "runs"
        out.mkdir(parents=True, exist_ok=True)
        for d in (tmp / "main").iterdir():
            d.rename(out / d.name)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
