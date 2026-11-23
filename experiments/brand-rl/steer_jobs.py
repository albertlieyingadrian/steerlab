"""Write the steering use test's jobs (configs/steer.json) from the fitted monitor (plan, Monitors 6).

    .venv/bin/python experiments/brand-rl/steer_jobs.py

The hack direction at the monitor's layer and 5 random directions with its norm (seed 66). The added vector is
scale x (median activation norm of honest monitor pages) x unit(direction), so every direction adds the same norm.
One job picks the scale on 24 monitor briefs (the largest whose random directions keep invalid + empty + loop <= 0.2);
three jobs run each scale on the 48 dev briefs (every scale is reported; the chosen one leads).
"""

from __future__ import annotations

import json

import numpy as np

import ep6

SCALES = (0.1, 0.25, 0.5)


def main() -> None:
    m = dict(np.load(ep6.RESULTS / "monitor.npz"))
    d = m["direction"]
    rng = np.random.default_rng(66)
    dirs = {"hack": d.tolist()}
    for i in range(5):
        v = rng.standard_normal(d.shape[0])
        dirs[f"rand{i}"] = (v / np.linalg.norm(v) * np.linalg.norm(d)).tolist()
    common = {
        "layer": int(m["layer"]),
        "directions": dirs,
        "base_norm": float(m["median_norm"]),
        "seed": 0,
        "batch": 16,
    }
    jobs = [dict(common, tag="steer_select_monitor", split="monitor", limit=24, scales=list(SCALES))]
    jobs += [dict(common, tag=f"steer_dev_a{sc}", split="dev", limit=48, scales=[sc]) for sc in SCALES]
    (ep6.HERE / "configs" / "steer.json").write_text(json.dumps(jobs))
    print(len(jobs), "jobs; layer", common["layer"], "base_norm", round(common["base_norm"], 2))


if __name__ == "__main__":
    main()
