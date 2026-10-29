"""
the sizing pilot (Qwen3-8B, about 5 minutes).

    .venv/bin/python experiments/natural-sandbagging/pilot.py

Before the plan fixed the reasoning budget, the 64 held-out PILOT items (32 WMDP, 32 MMLU; never scored again) were
run once with NO framing and a generous 1,024-token budget. This measures two things only:

    how long Qwen3-8B thinks on these questions   (what share would finish under 512, 768, 1,024 tokens)
    how fast the laptop generates                   (tokens per second, to price the full run)

It never uses an eval or deployment framing, so it cannot peek at the result the plan is about.
Writes results/pilot.json and the traces to the work folder.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural as t

LESSON = Path(__file__).resolve().parent
CAP_PILOT = 1024


def main():
    items = t.load_items("pilot")
    jobs = [
        {
            "key": f"{r['id']}|neutral|s0",
            "id": r["id"],
            "set": r["set"],
            "domain": r["domain"],
            "kind": "neutral",
            "pair": r["pair"],
            "seed": 0,
        }
        for r in items
    ]
    tok, model = t.load_model()
    t0 = time.time()
    out = t.WORK / "pilot" / "pilot_neutral_cap1024.jsonl"
    t.generate(tok, model, jobs, out, cap=CAP_PILOT, batch=64)
    wall = time.time() - t0
    rows = t.score_rows(t.read_jsonl(out), items)
    L = np.array([r["think_tokens"] for r in rows])
    nat = np.array([r["natural_end"] for r in rows])
    gen_tokens = int(L.sum())
    summary = {
        "n": len(rows),
        "cap": CAP_PILOT,
        "finished_naturally": int(nat.sum()),
        "think_tokens_median_finished": float(np.median(L[nat])) if nat.any() else None,
        "share_under": {str(c): float(np.mean(nat & (c >= L))) for c in (256, 512, 768, 1024)},
        "acc": {s: float(np.mean([r["correct"] for r in rows if r["set"] == s])) for s in ("wmdp", "mmlu")},
        "wall_seconds": round(wall, 1),
        "generated_tokens": gen_tokens,
        "tokens_per_second": round(gen_tokens / wall, 1) if wall > 0 else None,
    }
    t.write_json(LESSON / "results" / "pilot.json", summary)
    print(f"\n{len(rows)} pilot items, neutral framing, budget {CAP_PILOT} tokens, {wall:.0f} s")
    print(
        f"finished thinking before the cap: {nat.sum()}/{len(rows)}; "
        f"median length of those {summary['think_tokens_median_finished']}"
    )
    for c, s in summary["share_under"].items():
        print(f"  share finished within {c:>5} tokens: {s:.2f}")
    print(f"accuracy (pilot, neutral): WMDP {summary['acc']['wmdp']:.2f}, MMLU {summary['acc']['mmlu']:.2f}")
    print(f"throughput: {gen_tokens} generated tokens in {wall:.0f} s = {summary['tokens_per_second']} tok/s")
    checks = [
        t.check("every pilot item answered", len(rows) == len(items)),
        t.check("accuracy above chance (0.25) on both sets", all(v > 0.25 for v in summary["acc"].values())),
    ]
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
