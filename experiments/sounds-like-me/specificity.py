"""Specificity control (amendment A4, 2026-11-20): does a contestant sound like THIS author, or just more human?

The pre-registered voice position compares a 4-draft episode with the target author's profile, scaled between other
humans (floor) and the same human (ceiling). A paraphraser (DIPPER) raised Gemini's position from -0.80 to -0.18
without seeing the author at all, so part of "position" is generic human-likeness (losing the model fingerprint).

specificity = sim(episode, target profile) - mean sim(episode, the other held-out authors' profiles)
Positive = closer to the target than to other people. Reported beside position; the headline metric is unchanged.
Reference: the author's own held-out texts (ceiling episodes) give the human specificity.

    python specificity.py      -> results/specificity.json
"""

import json
from collections import defaultdict

import numpy as np

from common import DATA, HERE, authors, luar_embed, strip_greeting_signoff, texts


def f(t):
    return " ".join(strip_greeting_signoff(t).split())


def main():
    A = [a for a in authors()["authors"] if a["split"] == "heldout"]
    names = [a["author"] for a in A]
    prof = luar_embed([[f(t) for t in texts(a, "profile")] for a in A])
    idx = {n: i for i, n in enumerate(names)}

    def spec(vec, au):
        s = prof @ vec
        i = idx[au]
        return float(s[i] - np.delete(s, i).mean())

    human = [spec(luar_embed([[f(t) for t in texts(a, "ceiling")[:4]]])[0], a["author"]) for a in A]
    out = {"human_ceiling": dict(mean=round(float(np.mean(human)), 4), n=len(human))}
    for p in sorted((DATA / "scored").glob("*__heldout.json")):
        groups = defaultdict(list)
        for r in json.loads(p.read_text())["drafts"]:
            if r["text"].strip() and r["author"] in idx:
                groups[(r["model"], r["arm"], r["author"])].append(f(r["text"]))
        keys = list(groups)
        if not keys:
            continue
        vecs = luar_embed([groups[k] for k in keys])
        per = defaultdict(list)
        for k, v in zip(keys, vecs, strict=False):
            per[(k[0], k[1])].append(spec(v, k[2]))
        for (m, arm), vals in per.items():
            out[f"{m}|{arm}"] = dict(
                mean=round(float(np.mean(vals)), 4),
                n=len(vals),
                share_positive=round(float(np.mean([x > 0 for x in vals])), 3),
            )
    (HERE / "results" / "specificity.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
