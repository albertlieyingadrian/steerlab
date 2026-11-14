"""Follow the 32B GRPO (PREREG 2026-11-13, amendment 1): wait for the hard-pool screen, build the 1-6-of-8 pool, run
40 GRPO steps on one H200 (grpo1), then validate every saved checkpoint and grade it.

    MODAL_PROFILE=albertlieyingadrian python experiments/fin-math/run_grpo32.py --screen <call id>
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modal

import fm
import modal_think as mt

THINK = {"n": 4, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0, "max_tokens": 16384}
SAVES = (10, 20, 30, 40)


def wait(call_id: str):
    c = modal.FunctionCall.from_id(call_id)
    while True:
        try:
            return c.get(timeout=600)
        except TimeoutError:
            continue
        except Exception as error:
            if any(k in repr(error) for k in ("Connection", "Deadline")):
                time.sleep(60)
                continue
            print("ERROR", call_id, repr(error)[:400], flush=True)
            return None


def kept(rows, screen):
    return [r for r, g in zip(rows, screen, strict=True) if 1 <= sum(x == 1.0 for x in g) <= 6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="")
    a = ap.parse_args()
    data, res = HERE / "data", fm.RESULTS
    hard = json.loads((data / "q32_grpo_candidates_hard.json").read_text())
    scr = wait(a.screen) if a.screen else None
    for attempt in range(3):  # the screen saves every chunk and resumes from the volume, so a timeout loses little
        if scr is not None:
            break
        call = modal.Function.from_name("fin-math-32b", "screen").spawn(hard, tag="screen-hard", chunk=10)
        print("screen-hard attempt", attempt, call.object_id, flush=True)
        scr = wait(call.object_id)
    if scr is None:
        return
    (res / "q32_screen_hard.json").write_text(json.dumps(scr))
    easy = json.loads((data / "q32_grpo_candidates.json").read_text())
    pool = kept(easy, json.loads((res / "q32_screen.json").read_text())) + kept(hard, scr)
    random.Random(7).shuffle(pool)
    counts = {k: sum(sum(x == 1.0 for x in g) == k for g in scr) for k in range(9)}
    print("hard screen right-of-8:", counts, "pool", len(pool), flush=True)
    (res / "q32_grpo_pool.json").write_text(json.dumps({"ids": [p["id"] for p in pool], "hard_counts": counts}))
    g = modal.Function.from_name("fin-math-32b", "grpo1").spawn(pool, steps=40, save_at=SAVES, tag="q32-grpo")
    print("grpo", g.object_id, flush=True)
    r = wait(g.object_id)
    if r is None:
        return
    (res / "q32_grpo_history.json").write_text(json.dumps(r, indent=1))
    print("GRPO done", r["minutes"], "min", flush=True)
    items = fm.load("validation")
    prompts = [fm.prompt(i) for i in items]
    gen = modal.Function.from_name("fin-math-32b", "generate")
    calls = {s: gen.spawn(prompts, f"q32-grpo-s{s}-think", f"/vol/adapters/q32-grpo-s{s}", True, THINK) for s in SAVES}
    print("validation spawned", {s: c.object_id for s, c in calls.items()}, flush=True)
    for s, c in calls.items():
        out = wait(c.object_id)
        if out:
            mt.grade(items, f"q32-grpo-s{s}-think", out)


if __name__ == "__main__":
    main()
