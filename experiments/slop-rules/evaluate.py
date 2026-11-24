"""Score every v1 rule: human vs open-model windows (all, train, test), per model, per source, and on the transfer sets.

Writes results/rules_v1_{all,train,test,closed,raid,classic}.csv, rules_v1_by_model.csv, rules_v1_by_source.csv and
random_lists.json (the ban list against 200 frequency-matched random word lists).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from common import CACHE, RESULTS, SEED, WORD, load_corpus, write_json
from ruleset import Rule, base_words, fire, v1_rules
from stats import rule_table

FIRES = CACHE / "fires_v1.npz"


def firing(rows: list[dict], rules: list[Rule], path=FIRES) -> np.ndarray:
    key = f"{len(rows)}:{sum(r['words'] for r in rows)}:{len(rules)}"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["counts"]
    counts = fire(rules, [r["text"] for r in rows])
    np.savez_compressed(path, counts=counts, key=np.array(key))
    return counts


def table(rows, counts, rids, human_mask, ai_mask, **kw) -> pd.DataFrame:
    m = human_mask | ai_mask
    sub = [r for r, k in zip(rows, m, strict=True) if k]
    return rule_table(
        counts[m], np.array([r["words"] for r in sub]), ai_mask[m], np.array([r["group"] for r in sub]), rids, **kw
    )


def random_lists(rows: list[dict], n_lists: int = 200) -> dict:
    """BP of 'any word in the list fires' for the real ban list and for lists matched word by word on frequency."""
    from deslop.habits import inflections

    mt = [r for r in rows if r["set"] == "matched"]
    freq: Counter[str] = Counter()
    for r in mt:
        freq.update(w.lower() for w in WORD.findall(r["text"]))
    vocab = sorted(w for w in freq if w.isalpha())
    edges = np.quantile([np.log10(freq[w]) for w in vocab], np.linspace(0, 1, 11))

    def decile(w: str) -> int:
        return int(np.clip(np.searchsorted(edges, np.log10(max(freq[w], 1)), side="right") - 1, 0, 9))

    bins: dict[int, list[str]] = {}
    for w in vocab:
        bins.setdefault(decile(w), []).append(w)
    real = base_words("v1")
    texts = [{w.lower() for w in WORD.findall(r["text"])} for r in mt]
    is_ai = np.array([r["side"] == "ai" for r in mt])

    def bp(words: list[str]) -> float:
        forms = set().union(*(inflections(w) for w in words))
        f = np.array([bool(t & forms) for t in texts])
        h, a = f[~is_ai].mean(), f[is_ai].mean()
        return float(a / (a + h)) if a + h else float("nan")

    rng = np.random.default_rng(SEED)
    real_bins = [decile(w) for w in real]
    rand = [bp([bins[b][rng.integers(len(bins[b]))] for b in real_bins]) for _ in range(n_lists)]
    return {
        "real_bp": bp(real),
        "random_bp": rand,
        "n_words": len(real),
        "note": "BP of 'any list word fires in the window', matched windows, all splits; random words drawn from "
        "the same frequency decile (pooled human+AI counts) as each ban word",
    }


def masks(rows: list[dict]) -> dict[str, np.ndarray]:
    def get(k):
        return np.array([r[k] for r in rows])

    side, st, split = get("side"), get("set"), get("split")
    return {
        "hum": side == "human",
        "ai": side == "ai",
        "set": st,
        "split": split,
        "model": get("model"),
        "source": get("source"),
        "mh": (side == "human") & (st == "matched"),
        "ma": (side == "ai") & (st == "matched"),
    }


def all_tables(rows, counts, rids, meta: pd.DataFrame, tag: str) -> pd.DataFrame:
    k = masks(rows)
    mh, ma, split, st, hum, ai = k["mh"], k["ma"], k["split"], k["set"], k["hum"], k["ai"]

    def save(name: str, df: pd.DataFrame) -> pd.DataFrame:
        df = meta.merge(df, on="rule")
        df.to_csv(RESULTS / f"rules_{tag}_{name}.csv", index=False, float_format="%.4f")
        return df

    allt = save("all", table(rows, counts, rids, mh, ma))
    save("train", table(rows, counts, rids, mh & (split == "train"), ma & (split == "train")))
    save("test", table(rows, counts, rids, mh & (split == "test"), ma & (split == "test")))
    # transfer sets: closed models against matched human test windows; RAID; the classic setting
    save("closed", table(rows, counts, rids, mh & (split == "test"), ai & (st == "closed")))
    save("raid", table(rows, counts, rids, hum & (st == "raid"), ai & (st == "raid")))
    save("classic", table(rows, counts, rids, hum & (st == "classic"), ai & (st == "classic")))
    cols = ["rule", "group", "h", "a", "bp", "bp_lo", "bp_hi"]
    per = []
    for m in sorted(set(k["model"][ma])):
        t = table(rows, counts, rids, mh, ma & (k["model"] == m), n_boot=200)
        per.append(t.assign(group=m)[cols])
    for m in sorted(set(k["model"][ai & (st == "closed")])):
        t = table(rows, counts, rids, mh & (split == "test"), ai & (k["model"] == m), n_boot=200)
        per.append(t.assign(group=f"closed:{m}")[cols])
    pd.concat(per).to_csv(RESULTS / f"rules_{tag}_by_model.csv", index=False, float_format="%.4f")
    per = []
    for s in sorted(set(k["source"][mh])):
        t = table(rows, counts, rids, mh & (k["source"] == s), ma & (k["source"] == s), n_boot=200)
        per.append(t.assign(group=s)[cols])
    pd.concat(per).to_csv(RESULTS / f"rules_{tag}_by_source.csv", index=False, float_format="%.4f")
    return allt


def main() -> None:
    rows = load_corpus()
    rules = v1_rules()
    rids = [r.rid for r in rules]
    counts = firing(rows, rules)
    meta = pd.DataFrame({"rule": rids, "kind": [r.kind for r in rules], "label": [r.label for r in rules]})
    allt = all_tables(rows, counts, rids, meta, "v1")
    write_json(RESULTS / "random_lists.json", random_lists(rows))
    print(allt[allt.kind == "word"].verdict.value_counts().to_dict())
    print(allt[allt.kind != "word"][["rule", "h", "a", "bp", "bp_lo", "verdict"]].to_string())
    rl = json.loads((RESULTS / "random_lists.json").read_text())
    print(
        "ban list BP",
        round(rl["real_bp"], 3),
        "random lists p95",
        round(float(np.nanpercentile(rl["random_bp"], 95)), 3),
    )


if __name__ == "__main__":
    main()
