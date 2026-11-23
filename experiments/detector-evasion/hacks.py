"""The hack catalogue: what each arm's rewrites do, beyond the checks fixed in common.py.

    python hacks.py        # -> results/hacks.json; prints the catalogue

Per group (96 Qwen3-8B held-out rewrites): the share of each pre-fixed hack check (loop, cut, ends mid-sentence,
length collapse, odd characters, empty), the most common 3-word opener, placeholders in square brackets that the source
did not have ("[Colleague's Name]"), and "template phrases": word 3-grams that appear in at least 25% of the group's
rewrites but in at most 5% of the sources. Then, for every hack found, up to three example rewrites that evade the 9B.
Reads results/per_rewrite.csv (table.py) and the rewrites; no model calls.
"""

from __future__ import annotations

import csv
import re
from collections import Counter

from common import RESULTS, read_jsonl, save_json, sources
from evaluate import GEN

CHECKS = ("loop", "cut", "mid_sentence", "collapse", "odd", "empty")


def grams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z']+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def surface(text: str) -> dict:
    """Measures added after reading the held-out rewrites (descriptive only; no prediction uses them): the task's
    "Passage:" label echoed at the top, the share of words that start with a capital (Title Case), and full stops,
    question and exclamation marks per 100 words."""
    w = re.findall(r"[A-Za-z][A-Za-z']*", text)
    return {"echo_label": int(text.lstrip().lower().startswith("passage:")),
            "capitalised": sum(x[0].isupper() for x in w) / max(len(w), 1),
            "stops_per_100w": 100 * len(re.findall(r"[.!?]", text)) / max(len(w), 1)}  # fmt: skip


def placeholders(text: str) -> set[str]:
    return set(re.findall(r"\[[A-Z][^\]\n]{1,40}\]", text))


def main() -> None:
    S = {s["sid"]: s for s in sources("heldout")}
    src_grams = Counter(g for s in S.values() for g in grams(s["text"]))
    with (RESULTS / "per_rewrite.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "heldout"]
    out = {}
    for g in sorted({r["group"] for r in rows}):
        R = [r for r in rows if r["group"] == g]
        texts = {r["sid"]: r["text"] for r in read_jsonl(GEN / f"{g}.jsonl") if r["sid"] in S}
        n = len(R)
        rec = {
            "n": n,
            "checks": {c: round(sum(int(r[c]) for r in R) / n, 3) for c in CHECKS},
            "nonword": round(sum(float(r["nonword"]) for r in R) / n, 4),
            "lower_starts": round(sum(float(r["lower_starts"]) for r in R) / n, 3),
        }
        sf = [surface(t) for t in texts.values()]
        rec["surface"] = {k: round(sum(x[k] for x in sf) / len(sf), 3) for k in sf[0]}
        ops = Counter(r["opener"] for r in R).most_common(3)
        rec["openers"] = [{"opener": o, "share": round(k / n, 3)} for o, k in ops]
        new_ph = Counter(p for sid, t in texts.items() for p in placeholders(t) - placeholders(S[sid]["text"]))
        rec["new_placeholders"] = {"share": round(sum(bool(placeholders(t) - placeholders(S[sid]["text"]))
                                                      for sid, t in texts.items()) / n, 3),
                                   "top": new_ph.most_common(5)}  # fmt: skip
        gc = Counter(x for t in texts.values() for x in grams(t))
        tmpl = [(" ".join(k), round(v / n, 3)) for k, v in gc.most_common(400)
                if v / n >= 0.25 and src_grams[k] / len(S) <= 0.05]  # fmt: skip
        rec["template_phrases"] = tmpl[:12]
        ev = [r for r in R if r["evade_loose"] == "1"]
        rec["examples"] = {}
        for c in CHECKS:
            hits = sorted([r for r in ev if r[c] == "1"], key=lambda r: float(r["s9"]))[:3]
            if hits:
                rec["examples"][c] = [{"sid": r["sid"], "s9": round(float(r["s9"]), 4), "text": texts[r["sid"]][:700]}
                                      for r in hits]  # fmt: skip
        clean = sorted([r for r in ev if r["evade_strict"] == "1"], key=lambda r: float(r["s9"]))[:3]
        rec["examples"]["strict_evasion"] = [{"sid": r["sid"], "s9": round(float(r["s9"]), 4),
                                              "source": S[r["sid"]]["text"][:700], "text": texts[r["sid"]][:700]}
                                             for r in clean]  # fmt: skip
        out[g] = rec
        print(f"{g:22} " + " ".join(f"{c} {v:.2f}" for c, v in rec["checks"].items())
              + f" | opener '{ops[0][0]}' {ops[0][1] / n:.2f} | placeholders {rec['new_placeholders']['share']:.2f}"
              + f" | {rec['surface']} | nonword {rec['nonword']} | templates {[t for t, _ in tmpl[:4]]}")  # fmt: skip
    save_json("hacks.json", out)


if __name__ == "__main__":
    main()
