"""Boring-explanation checks for the steering result (added after reading the Part C drafts, 2026-11-20).

Reading the held-out steered drafts showed some of them loop ("I think that maybe they need some drywall labor staffing
for peak months." eight times), and steered drafts are about 50% longer. Both could move LUAR without carrying voice.
This script measures them and re-scores the voice position without them. It is labelled as added after results.

  loop       a draft in which some 5-word sequence occurs 3 or more times
  loop-free  the strict voice position recomputed on each episode with its looping drafts removed
             (an author is kept for an arm only if at least 2 of its 4 drafts are loop-free)
  length-matched  every draft cut to its first 80 words (after the strict normalisation), then re-scored

Floor and ceiling per author are the ones the scorer stored with each episode (score.py), so only the episode's
embedding changes. Also reports the loop rate of every Part B contestant, for comparison.

    python robustness.py    -> results/robustness.json
"""

import json
from collections import Counter, defaultdict

import numpy as np

from common import DATA, HERE, authors, cos, luar_embed, strip_greeting_signoff, texts

RNG = np.random.default_rng(20261120)
CUT_WORDS = 80
RANDS = ("rand0", "rand1", "rand2")


def norm(t):
    return " ".join(strip_greeting_signoff(t).split())


def loops(t, n=5, k=3):
    w = t.lower().split()
    grams = Counter(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
    return bool(grams) and max(grams.values()) >= k


def boot(v, n=2000):
    v = np.asarray(v, float)
    m = [float(np.mean(RNG.choice(v, len(v)))) for _ in range(n)]
    return [round(float(v.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    loop_rate = {}
    for p in sorted((DATA / "scored").glob("*__heldout.json")):
        per = defaultdict(list)
        for r in json.loads(p.read_text())["drafts"]:
            if r["text"].strip():
                per[(r["model"], r["arm"])].append(loops(r["text"]))
        for (m, arm), v in per.items():
            loop_rate[f"{m}|{arm}"] = round(float(np.mean(v)), 3)

    d = json.loads((DATA / "scored" / "partc__heldout.json").read_text())
    A = {a["author"]: a for a in authors()["authors"]}
    eps = {(e["author"], e["arm"]): e for e in d["episodes"]}
    drafts = defaultdict(list)
    for r in d["drafts"]:
        drafts[(r["author"], r["arm"])].append(r["text"])
    names = sorted({a for a, _ in eps})
    profile = {a: luar_embed([[norm(t) for t in texts(A[a], "profile")]])[0] for a in names}
    arms = sorted({arm for _, arm in eps})

    def position(a, arm, ep):
        e = eps[(a, arm)]
        return (cos(luar_embed([ep])[0], profile[a]) - e["floor"]) / max(e["ceiling"] - e["floor"], 1e-6)

    out = dict(
        rule=dict(
            loop="some 5-word sequence occurs 3+ times",
            loop_free="drop looping drafts; keep an author for an arm only with 2+ loop-free drafts",
            length_matched=f"first {CUT_WORDS} words",
        ),
        loop_rate=loop_rate,
        partc={},
    )
    for variant in ("loop_free", "length_matched"):
        pos = defaultdict(dict)
        for a in names:
            for arm in arms:
                ts = [norm(t) for t in drafts[(a, arm)] if t.strip()]
                if variant == "loop_free":
                    ts = [t for t in ts if not loops(t)]
                    if len(ts) < 2:
                        continue
                else:
                    ts = [" ".join(t.split()[:CUT_WORDS]) for t in ts]
                pos[arm][a] = position(a, arm, ts)
        rows = {arm: dict(mean=round(float(np.mean(list(v.values()))), 4), n_authors=len(v)) for arm, v in pos.items()}
        both = [a for a in pos["own"] if all(a in pos[r] for r in RANDS)]
        diff = [pos["own"][a] - np.mean([pos[r][a] for r in RANDS]) for a in both]
        wboth = [a for a in pos["own"] if a in pos["wrong"]]
        wdiff = [pos["own"][a] - pos["wrong"][a] for a in wboth]
        out["partc"][variant] = dict(
            arms=rows,
            own_minus_random=dict(value_ci=boot(diff), n_authors=len(both)),
            own_minus_wrong=dict(value_ci=boot(wdiff), n_authors=len(wboth)),
        )
    (HERE / "results" / "robustness.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["partc"], indent=1))
    print({k: v for k, v in loop_rate.items() if k.startswith("partc") or v > 0.05})


if __name__ == "__main__":
    main()
