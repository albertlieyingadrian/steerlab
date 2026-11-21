"""
Shared pieces for The Taste Stack Episode 3 (slop probes vs judges, for design): paths, the label sets, the badge
mask box, and a loader that joins Ep 2's index (read-only) with this episode's own features.

Plan (fixed before any held-out score): the plan
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EP2 = HERE.parent / "design-index"
RESULTS = HERE / "results"
CACHE = Path.home() / ".cache" / "steerlab" / "slop-probes"  # this episode's raw outputs, never committed
EP2_CACHE = Path.home() / ".cache" / "steerlab" / "design-index"  # Ep 2's renders and features, read-only

# The badge box: builder badges ("Edit with ...", "Made in ...") float in the bottom-right corner of the first screen.
# Every page, of every class, gets this box painted mid-grey and loses the elements inside it (plan, Arms).
VIEW_W, VIEW_H = 1280, 800
MASK_W, MASK_H = 320, 96
MASK_BOX = (VIEW_W - MASK_W, VIEW_H - MASK_H, VIEW_W, VIEW_H)  # x0, y0, x1, y1
MASK_GREY = (128, 128, 128)

SOURCES = ("ai_host", "human", "curated", "template", "generated")


def ep2():
    """Import Ep 2's modules (common, dataset) without copying them."""
    if str(EP2) not in sys.path:
        sys.path.append(str(EP2))  # appended, so this episode's budget.py wins over Ep 2's
    import common as ep2_common
    import dataset as ep2_dataset

    return ep2_common, ep2_dataset


def builder_of(domain: str) -> int:
    """Index of the AI-builder host (0-4, order of Ep 2's lab-only host list); -1 if none. Public text says 'builder N'."""
    ep2_common, _ = ep2()
    for i, h in enumerate(ep2_common.AI_HOSTS):
        if domain == h or domain.endswith("." + h):
            return i
    return -1


def load_pages() -> list[dict]:
    """Ep 2's kept, live-rendered pages of the five sources this episode uses, plus a builder id and labels."""
    _, ep2_dataset = ep2()
    out = []
    for r in ep2_dataset.load():
        if r["source"] not in SOURCES:
            continue
        r = dict(r)
        r["builder"] = builder_of(r["domain"]) if r["source"] == "ai_host" else -1
        r["y_main"] = {"ai_host": 1, "human": 0}.get(r["source"])  # main task: AI-builder vs human
        r["y_known"] = {"template": 1, "generated": 1, "curated": 0}.get(r["source"])  # known-answer set
        out.append(r)
    return out


def parse_constitution() -> list[dict]:
    """The eight principles of constitution.md: id, name, and the slop / purpose-made sentences (whitespace folded)."""
    import re

    text = (HERE / "constitution.md").read_text()
    pat = r"## (S\d) (.+?)\n\n- Slop: (.+?)\n- Purpose-made: (.+?)(?=\n\n## |\Z)"
    out = [
        {"id": m[1], "name": m[2], "slop": " ".join(m[3].split()), "good": " ".join(m[4].split())}
        for m in re.finditer(pat, text, re.DOTALL)
    ]
    assert len(out) == 8, f"constitution parse found {len(out)} principles"
    return out


def load_own(name: str) -> dict:
    p = CACHE / name
    return pickle.loads(p.read_bytes()) if p.exists() else {}
