"""Turn the RL-hack-circuit study's full graph dumps into the demo file `examples/data/rl_hack_graphs.json.gz`.

The study (experiments/rl-hack-circuit) built 64 attribution graphs of Qwen3-1.7B with circuit-tracer and the
`mwhanna/qwen3-1.7b-transcoders-lowl0` transcoders: 8 prompts x before/after RL x 4 training runs. Its GPU job saved
each graph's selected features, node influence and pruned edges to `graphs_full_<contrast>.json` on a Modal volume
(about 500 MB each). This script keeps what a diff needs: every unit's influence (rounded to 4 significant figures)
and the 3,000 heaviest unit-to-unit edges per graph, plus the per-layer reconstruction error the study measured.

    modal volume get rl-hack-circuit work/graphs_full_installed.json .   # and _control, _replication
    python examples/scripts/compact_rl_hack_graphs.py <dir with the three files> experiments/rl-hack-circuit/results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from circuitkit.diff.graphs import GraphSummary, save_summaries, unit

RUNS = {
    "installed": "loophole2_seed2 (hacked)",
    "replication": "loophole2_seed3 (hacked, another seed)",
    "control": "loophole2_seed1 (did not hack)",
}
MAX_EDGES = 3000


def sig(x: float, n: int = 4) -> float:
    return float(f"{x:.{n}g}")


def main(src: Path, results: Path) -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "rl_hack_graphs.json.gz"
    groups: dict[str, list[GraphSummary]] = {}
    for contrast, run in RUNS.items():
        full = json.loads((src / f"graphs_full_{contrast}.json").read_text())
        for side in ("before", "after"):
            gs = []
            for k in range(8):
                g = full[f"{contrast}_{side}_{k}"]
                feats = g["features"]
                edges: dict[tuple[str, str], float] = {}
                for t, s, w in g["edges"]:
                    su, tu = unit(feats[s][0], feats[s][2]), unit(feats[t][0], feats[t][2])
                    if su != tu:
                        edges[(su, tu)] = edges.get((su, tu), 0.0) + w
                top = sorted(edges.items(), key=lambda kv: -abs(kv[1]))[:MAX_EDGES]
                gs.append(
                    GraphSummary(
                        name=f"{contrast}_{side}_{k}",
                        model="Qwen/Qwen3-1.7B" + (" + Pigment Gym RL" if side == "after" else " (step 20)"),
                        prompt=f"Pigment Gym sketch prefix {k} (comment slot)",
                        features={u: sig(v) for u, v in g["feature_influence"].items()},
                        edges={e: sig(w) for e, w in top},
                        replacement_score=sig(g["replacement_score"]),
                        completeness_score=sig(g["completeness_score"]),
                        error_influence=sig(g["error_influence"]),
                        meta={"run": run, "n_tokens": g["n_tokens"], "target": "mean keyword logit - alternative"},
                    )
                )
            groups[f"{contrast}/{side}"] = gs
        del full
    recon = json.loads((results / "verdicts.json").read_text())["recon"]
    for key in ("installed/before", "installed/after"):
        for g in groups[key]:
            g.meta["fvu_by_layer"] = [sig(x) for x in recon[key]["fvu_by_layer"]]
    save_summaries(out, groups)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
