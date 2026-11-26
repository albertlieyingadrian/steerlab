"""torchrun entry: install the pushback environment and the GRAFT trainer subclass, then run Halo's stock
scripts/training/grpo/environmental.py unchanged.

    torchrun --nproc_per_node=1 run_halo.py <config.yaml>
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graft_trainer  # noqa: E402

graft_trainer.install()
script = "/workspace/scripts/training/grpo/environmental.py"
sys.argv = [script, *sys.argv[1:]]
runpy.run_path(script, run_name="__main__")
