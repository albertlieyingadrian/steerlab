"""Part A through the OpenAI API: gpt-4.1-mini answers every MMLU-Redux item (the auditor, A2, and the ceiling, A8),
and does TS-Guessing on the fixed 600 (A3).

    python a_api.py [--limit N]

The auditor (gpt-4.1-mini) reasons briefly in its reply (up to 1,500 tokens) and ends with "Answer: X".
Writes results/a_api.json:
ids, the auditor's letters as indices into the original options (-1 if none found), the TS-Guessing fills and the
question-only control. Every call is cached under ~/.cache/steerlab/benchmark-101 and the spend is capped
(bench.API_CAP_USD).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import bench

AUDIT_SUFFIX = "\n\nThink it through, then end your reply with 'Answer: X', where X is the letter."


def audit(r: dict) -> int:
    text = bench.chat(
        bench.AUDITOR, bench.mc_prompt(r["question"], r["choices"]) + AUDIT_SUFFIX, max_tokens=1500, reasoning=True
    )
    L = bench.letter(text)
    return bench.LETTERS.index(L) if L and bench.LETTERS.index(L) < len(r["choices"]) else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    items = bench.mmlu_redux()[: a.limit]
    ts = bench.ts_sample(bench.mmlu_redux())[: a.limit]
    with ThreadPoolExecutor(16) as ex:
        preds = list(ex.map(audit, items))
        fills = list(
            ex.map(
                lambda r: bench.chat(
                    bench.AUDITOR, bench.ts_prompt(r["question"], r["choices"], r["mask"]), max_tokens=48
                ),
                ts,
            )
        )
        ctrl = list(
            ex.map(lambda r: bench.chat(bench.AUDITOR, bench.ts_control_prompt(r["question"]), max_tokens=48), ts)
        )
    out = {
        "model": bench.AUDITOR,
        "ids": [r["id"] for r in items],
        "pred": preds,
        "ts_ids": [r["id"] for r in ts],
        "ts": fills,
        "ts_control": ctrl,
    }
    if a.limit is None:
        (bench.RESULTS / "a_api.json").write_text(json.dumps(out) + "\n")
    acc = sum(p == r["answer"] for p, r in zip(preds, items, strict=True)) / len(items)
    print(f"{len(items)} items, acc {acc:.3f}, unparsed {preds.count(-1)}, spent ${bench.spent():.2f}")


if __name__ == "__main__":
    main()
