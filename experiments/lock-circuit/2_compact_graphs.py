"""Step 2: shrink the GPU's graphs.json (every pruned edge, ~0.9 GB) to results/graphs_summary.json (local).

    .venv/bin/python experiments/lock-circuit/2_compact_graphs.py

Keeps, per graph: the scores, the target, influence through features, errors and tokens, the error influence per
layer, and the 200 most influential features (summed over positions). The full file moves to the work folder.
"""

from __future__ import annotations

import json
import shutil
import sys

import common as c

TOP = 200


def main() -> int:
    src = c.RESULTS / "graphs.json"
    full = c.work_dir() / "graphs.json"
    if src.exists():
        shutil.move(src, full)
    graphs = json.loads(full.read_text())
    out = {}
    for key, g in graphs.items():
        agg: dict = {}
        for (layer, _pos, f), v in zip(g["features"], g["influence"], strict=True):
            agg[(layer, f)] = agg.get((layer, f), 0.0) + v
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:TOP]
        out[key] = {
            k: g[k]
            for k in (
                "key",
                "side",
                "cond",
                "qid",
                "correct",
                "n_tokens",
                "target_value_bf16",
                "seconds",
                "replacement_score",
                "completeness_score",
                "n_features",
                "n_active",
                "error_influence_by_layer",
                "token_influence",
                "feature_influence",
                "error_influence",
            )
        }
        out[key]["n_pruned_edges"] = len(g["edges"])
        out[key]["top_features"] = [[layer, f, v] for (layer, f), v in top]
    c.write_json(c.RESULTS / "graphs_summary.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
