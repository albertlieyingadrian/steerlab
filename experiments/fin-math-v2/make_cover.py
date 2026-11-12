"""Draw the note's cover: FinanceMath test accuracy for every system, as horizontal bars from zero.

    python experiments/fin-math-v2/make_cover.py

The drawing lives in ../shared/make_benchmark_cover.py, shared by both FinanceMath notes; this runs it for v2.
Writes report/figures/hero.jpg (1536 x 1024) and ../shared/thumbs/fin-math-v2.jpg (720 x 480).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "shared" / "make_benchmark_cover.py"

if __name__ == "__main__":
    sys.argv = [str(SHARED), "fin-math-v2"]
    runpy.run_path(str(SHARED), run_name="__main__")
