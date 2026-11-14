"""Follow the 32B jobs (PREREG 2026-11-12): wait for SFT, launch both epochs' validation arms, grade every arm.

MODAL_PROFILE=albertlieyingadrian python experiments/fin-math/run_32b.py --train <call id> --base <call id>
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modal

import fm
import modal_think as mt

THINK = {"n": 4, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0, "max_tokens": 16384}


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
            print("ERROR", call_id, repr(error)[:300], flush=True)
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    items = fm.load("validation")
    prompts = [fm.prompt(i) for i in items]
    gen = modal.Function.from_name("fin-math-32b", "generate")
    r = wait(a.train)
    if r:
        (fm.RESULTS / "q32_train.json").write_text(json.dumps(r, indent=1))
        print(
            "SFT",
            {k: v for k, v in r.items() if k != "history"},
            [h for h in r["history"] if "val_loss" in h],
            flush=True,
        )
        calls = {f"q32-sft-e{e}-think": gen.spawn(prompts, f"q32-sft-e{e}-think", f"/vol/adapters/q32-sft-e{e}", True, THINK)
                 for e in (1, 2)}  # fmt: skip
        print("spawned", {k: c.object_id for k, c in calls.items()}, flush=True)
    else:
        calls = {}
    res = wait(a.base)
    if res:
        mt.grade(items, "q32-base-think", res)
    for tag, c in calls.items():
        res = wait(c.object_id)
        if res:
            mt.grade(items, tag, res)


if __name__ == "__main__":
    main()
