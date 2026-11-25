"""Which categories changed, how often, and under which conditions.

Per prompt and category the score is +1 if the category points toward the new model there, -1 toward the reference,
0 if absent (several descriptions in one prompt are summed, then signed). A category has changed if its +1s and -1s
are unbalanced beyond chance: an exact two-sided sign test, Benjamini-Hochberg across categories. Under which
conditions: for each factor, the mean score per level and a permutation test that shuffles the level labels.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def sign_test(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    tail: float = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def bh(ps: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), in input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, ps[i] * m / rank)
        q[i] = running
    return q


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def scores(labels: list[dict[str, Any]], ids: list[str]) -> dict[str, dict[str, int]]:
    """{category: {prompt id: -1 | 0 | +1}} over every judged prompt id."""
    raw: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(ids, 0))
    for lab in labels:
        if lab["category"] is None or lab["id"] not in raw[lab["category"]]:
            continue
        sign = (1 if lab["toward"] == "new" else -1) * lab.get("polarity", 1)
        raw[lab["category"]][lab["id"]] += sign
    return {c: {i: (v > 0) - (v < 0) for i, v in s.items()} for c, s in raw.items()}


def factor_test(score: dict[str, int], levels: dict[str, str], n_perm: int = 2000, seed: int = 0) -> float:
    """Permutation p-value for 'the mean score differs between levels' (between-level sum of squares)."""
    ids = [i for i in score if i in levels]
    xs = [score[i] for i in ids]
    labs = [levels[i] for i in ids]

    def ss(lb: list[str]) -> float:
        tot: dict[str, list[int]] = defaultdict(list)
        for x, lv in zip(xs, lb, strict=True):
            tot[lv].append(x)
        mu = sum(xs) / len(xs) if xs else 0.0
        return sum(len(v) * (sum(v) / len(v) - mu) ** 2 for v in tot.values())

    obs = ss(labs)
    if obs == 0:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(labs)
        hits += ss(labs) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def breakdown(
    s: dict[str, int], factors: dict[str, dict[str, str]], ids: list[str], n_perm: int
) -> dict[str, dict[str, Any]]:
    """Per factor: counts and net rate per level over `ids`, and the permutation p-value."""
    out: dict[str, dict[str, Any]] = {}
    moved = sum(s[i] != 0 for i in ids)
    for f, lv in factors.items():
        levels: dict[str, dict[str, Any]] = {}
        for name in sorted({lv[i] for i in ids if i in lv}):
            members = [i for i in ids if lv.get(i) == name]
            kn, kr = sum(s[i] > 0 for i in members), sum(s[i] < 0 for i in members)
            levels[name] = {"n": len(members), "n_new": kn, "n_ref": kr, "net": (kn - kr) / max(1, len(members))}
        sub = {i: lv[i] for i in ids if i in lv}
        p_f = factor_test({i: s[i] for i in ids}, sub, n_perm) if moved >= 4 else 1.0
        out[f] = {"p": p_f, "levels": levels}
    return out


def summarize(
    categories: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    ids: list[str],
    alpha: float = 0.05,
    n_perm: int = 2000,
) -> list[dict[str, Any]]:
    """One row per category, most significant first, with a per-factor breakdown."""
    by_id = {p["id"]: p for p in prompts}
    factors: dict[str, dict[str, str]] = defaultdict(dict)
    for i in ids:
        p = by_id[i]
        factors["kind"][i] = p.get("kind", "item")
        for f, lv in p.get("conditions", {}).items():
            factors[f][i] = lv
    sc = scores(labels, ids)
    rows = []
    for c in categories:
        s = sc.get(c["id"], dict.fromkeys(ids, 0))
        k_new, k_ref = sum(v > 0 for v in s.values()), sum(v < 0 for v in s.values())
        row: dict[str, Any] = {
            **c,
            "n": len(ids),
            "n_new": k_new,
            "n_ref": k_ref,
            "toward": "new" if k_new > k_ref else "ref" if k_ref > k_new else "none",
            "p": sign_test(k_new, k_new + k_ref),
            "examples": [lab["text"] for lab in labels if lab["category"] == c["id"]][:5],
            "factors": breakdown(s, factors, ids, n_perm),
        }
        # A condition can matter inside one kind of prompt and be diluted by the others (a trigger that only acts on
        # multiple-choice items): the same breakdown within the kind where the category's net change is largest.
        kinds = row["factors"]["kind"]["levels"]
        top = max(kinds, key=lambda k: abs(kinds[k]["n_new"] - kinds[k]["n_ref"]))
        inside = [i for i in ids if factors["kind"][i] == top]
        sub = {f: lv for f, lv in factors.items() if f != "kind"}
        row["within"] = {"kind": top, "factors": breakdown({i: s[i] for i in inside}, sub, inside, n_perm)}
        rows.append(row)
    for row, q in zip(rows, bh([r["p"] for r in rows]), strict=True):
        row["q"] = q
        row["changed"] = q < alpha
    return sorted(rows, key=lambda r: (r["q"], r["p"]))


def flip_labels(labels: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """The label-shuffle control: swap which model is which, independently per prompt."""
    rng = random.Random(seed)
    flip: dict[str, bool] = {}
    out = []
    for lab in labels:
        f = flip.setdefault(lab["id"], rng.random() < 0.5)
        out.append({**lab, "toward": ({"new": "ref", "ref": "new"}[lab["toward"]] if f else lab["toward"])})
    return out
