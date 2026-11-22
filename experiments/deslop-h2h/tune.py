"""Choose each steering arm's setting on Episode 1's 24 dev prompts (never the held-out 48).

    python tune.py          # sweep -> results/gen/tune/*.jsonl, results/tune_sweep.json, results/tune_choice.json

Grids (plan, amended before any generation): register {-1, -2, -4, -6} class gaps at layers {12, 18, 24};
probe and sae-dir {4, 8, 16, 32}% of the layer-18 residual norm; sae-abl c in {1, 2, 4}. One reply per prompt.
Rule: a setting is usable when >= 22 of 24 replies pass Episode 1's `fluent` rule, task fit >= the original's - 2,
no reply is empty, and the loop share is <= the original's + 1/24. Among usable settings the lowest mean habit score
wins; if none is usable, the setting with the most fluent replies, flagged `usable: false`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from deslop.checks import task_fit
from deslop.score import score
from tune_steer import fluent  # Episode 1's fluent rule

import common  # noqa: F401  (puts Episode 1's folder on sys.path first)
from common import GEN, dev, looped, save_json, write_jsonl
from gen import generate, intervention

GRID = {
    "register": [{"layer": L, "strength": s} for L in (12, 18, 24) for s in (-1.0, -2.0, -4.0, -6.0)],
    "probe": [{"c": c} for c in (4.0, 8.0, 16.0, 32.0)],
    "sae-dir": [{"c": c} for c in (4.0, 8.0, 16.0, 32.0)],
    "sae-abl": [{"c": c} for c in (1.0, 2.0, 4.0)],
}


def key(kind: str, s: dict) -> str:
    return f"{kind}|" + "|".join(f"{k}={v}" for k, v in s.items()) if s else "original"


def stats(rows: list[dict]) -> dict:
    P = {p["id"]: p for p in dev()}
    texts = [r["text"] for r in rows]
    return {
        "habit": float(np.mean([score(t).score for t in texts])),
        "fluent": int(sum(fluent(r["text"], P[r["id"]]["words"]) for r in rows)),
        "task_fit": int(sum(all(task_fit(r["text"], P[r["id"]]["genre"], P[r["id"]]["words"]).values()) for r in rows)),
        "empty": int(sum(not t.strip() for t in texts)),
        "loops": float(np.mean([looped(t) for t in texts])),
        "words": float(np.mean([len(t.split()) for t in texts])),
    }


def main() -> None:
    rows = [{"id": p["id"], "prompt": p["prompt"]} for p in dev()]
    jobs = [("original", {}, None)] + [(k, s, intervention(k, s)) for k, ss in GRID.items() for s in ss]

    def one(job):
        kind, s, iv = job
        out = generate(f"tune-{key(kind, s)}", rows, n=1, iv=iv, chunks=1, projected=0.3)
        out.sort(key=lambda r: r["id"])
        write_jsonl(GEN / "tune" / f"{key(kind, s).replace('|', '_')}.jsonl", out)
        return key(kind, s), {"kind": kind, "setting": s, **stats(out)}

    with ThreadPoolExecutor(max_workers=10) as pool:
        sweep = dict(pool.map(one, jobs))
    base = sweep["original"]
    choice = {}
    for kind in GRID:
        cands = [v for v in sweep.values() if v["kind"] == kind]
        for v in cands:
            v["usable"] = (v["fluent"] >= 22 and v["task_fit"] >= base["task_fit"] - 2 and v["empty"] == 0
                           and v["loops"] <= base["loops"] + 1 / 24 + 1e-9)  # fmt: skip
        ok = [v for v in cands if v["usable"]]
        best = min(ok, key=lambda v: v["habit"]) if ok else max(cands, key=lambda v: v["fluent"])
        choice[kind] = {"setting": best["setting"], "dev_habit": best["habit"], "usable": best["usable"],
                        "dev_fluent": best["fluent"], "dev_task_fit": best["task_fit"],
                        "dev_loops": best["loops"]}  # fmt: skip
    for k, v in sweep.items():
        print(f"{k:32} habit {v['habit']:6.1f} fluent {v['fluent']:2d} fit {v['task_fit']:2d} "
              f"loops {v['loops']:.2f} empty {v['empty']} words {v['words']:.0f} {v.get('usable', '')}")  # fmt: skip
    print(choice)
    save_json("tune_sweep.json", sweep)
    save_json("tune_choice.json", {"rule": __doc__.split("Rule: ")[1].strip(), "original_dev": base,
                                   "choice": choice})  # fmt: skip


if __name__ == "__main__":
    main()
