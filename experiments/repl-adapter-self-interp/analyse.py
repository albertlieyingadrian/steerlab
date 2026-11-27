"""Analysis for the adapter self-interpretation replication (two-phase generation scoring, paper Appendix A.5).

    .venv/bin/python experiments/repl-adapter-self-interp/analyse.py main     # results/summary_main.json + tables

Per latent and arm: 6 candidate labels x 20 conversations. Half A (conversations 0-9) selects the candidate with the
highest hit rate (first on ties); half B (10-19) reports that candidate's hit rate. "naive" is the max over candidates
of the half-A rate (the biased best-of-6). "coverage" is 1 when any half-A conversation of any candidate hit.
"decoy" is the selected candidate's half-B hit rate on a random other latent; "specific" = hit rate - decoy.
Intervals: 95% bootstrap over latents (10,000 resamples), paired for differences.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ARMS = ["released", "ours", "shuffled", "untrained", "auto6", "para", "wrong6", "bias_ours", "bias_released"]
METRICS = ("hit", "naive", "coverage", "decoy", "specific")


def per_latent(tag: str) -> tuple[dict, dict]:
    sc = json.loads((RESULTS / f"scores_{tag}.json").read_text())
    lat = json.loads((RESULTS / f"latents_{tag}.json").read_text())
    hits = sc["hits"]
    allt = sorted(set(lat["eval"]) | {int(v) for v in lat["decoy"].values()})
    col = {t: j for j, t in enumerate(allt)}
    rows = {a: {} for a in ARMS}
    for i in lat["eval"]:
        d = int(lat["decoy"][str(i)])
        for arm in ARMS:
            cands = []
            for c in range(6):
                if arm.startswith("bias_"):
                    h = np.array(hits[f"{arm}||{c}"])  # [20, len(allt)]
                    tgt, dec = h[:, col[i]], h[:, col[d]]
                else:
                    h = np.array(hits[f"{arm}|{i}|{c}"])  # [20, 2]
                    tgt, dec = h[:, 0], h[:, 1]
                cands.append((tgt, dec))
            a_rates = [t[:10].mean() for t, _ in cands]
            sel = int(np.argmax(a_rates))
            t, dd = cands[sel]
            rows[arm][i] = {
                "hit": float(t[10:].mean()),
                "naive": float(max(a_rates)),
                "coverage": float(any(t_[:10].any() for t_, _ in cands)),
                "decoy": float(dd[10:].mean()),
                "specific": float(t[10:].mean() - dd[10:].mean()),
                "sel": sel,
                "n_scales_hit": int(sum(t_[:10].any() for t_, _ in cands)),
            }
    floor = np.array(hits["floor"])[0]
    extra = {
        "template_floor": float(np.mean([floor[col[i]] for i in lat["eval"]])),
        "parse_fallback": sc["parse_fallback"],
        "n_convs": sc["n_convs"],
        "n_latents": len(lat["eval"]),
    }
    return rows, extra


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]]


PAIRS = [
    ("released", "auto6"),
    ("released", "para"),
    ("released", "untrained"),
    ("ours", "auto6"),
    ("ours", "shuffled"),
    ("ours", "released"),
    ("auto6", "wrong6"),
    ("released", "bias_released"),
    ("ours", "bias_ours"),
    ("para", "auto6"),
    ("shuffled", "bias_ours"),
]


def main(tag: str):
    rows, extra = per_latent(tag)
    lats = sorted(rows["ours"])
    out = {"extra": extra, "arms": {}, "diffs": {}}
    for arm in ARMS:
        out["arms"][arm] = {m: boot([rows[arm][i][m] for i in lats]) for m in METRICS}
        out["arms"][arm]["mean_n_scales_hit"] = float(np.mean([rows[arm][i]["n_scales_hit"] for i in lats]))
    for a, b in PAIRS:
        out["diffs"][f"{a}-{b}"] = {
            m: boot([rows[a][i][m] - rows[b][i][m] for i in lats]) for m in ("hit", "specific", "naive")
        }
    # known-answer check (plan): the auto label beats the wrong-latent label by >= 10 points on the first 50 latents
    first = lats[:50]
    ka = float(np.mean([rows["auto6"][i]["hit"] - rows["wrong6"][i]["hit"] for i in first]))
    out["known_answer_first50"] = {"auto_minus_wrong": ka, "pass": ka >= 0.10}
    out["per_latent"] = {arm: {str(i): rows[arm][i] for i in lats} for arm in ARMS}
    (RESULTS / f"summary_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(extra))
    print(f"{'arm':15s} {'hit (2-phase)':22s} {'naive best-of-6':22s} coverage  decoy  specific")
    for arm in ARMS:
        r = out["arms"][arm]

        def f(m, r=r):
            return f"{100 * r[m][0]:.1f} [{100 * r[m][1][0]:.1f}, {100 * r[m][1][1]:.1f}]"

        print(
            f"{arm:15s} {f('hit'):22s} {f('naive'):22s} {100 * r['coverage'][0]:7.1f} {100 * r['decoy'][0]:6.1f} "
            f"{100 * r['specific'][0]:8.1f}"
        )
    for k, v in out["diffs"].items():
        h, s = v["hit"], v["specific"]
        print(
            f"{k:26s} hit {100 * h[0]:+.1f} [{100 * h[1][0]:+.1f}, {100 * h[1][1]:+.1f}]   "
            f"specific {100 * s[0]:+.1f} [{100 * s[1][0]:+.1f}, {100 * s[1][1]:+.1f}]"
        )
    print("known answer", out["known_answer_first50"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
