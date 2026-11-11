"""Shared local helpers: prompt sets, GenEval scoring with intervals, GPU prices, Modal lookups."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TAG_ORDER = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
# Modal list prices per GPU-second (2026-11), used for $ per image; billed totals are in the execution log.
PRICE = {"H100": 0.001097, "A10G": 0.000306}


def load_set(name):
    path = {"official": HERE / "data/official/evaluation_metadata.jsonl", "val": HERE / "data/val.jsonl"}.get(
        name, HERE / f"data/{name}.jsonl"
    )
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def geneval_summary(rows, key="correct", n_boot=2000, seed=0):
    """GenEval overall = mean over tags of the image pass rate. Interval: bootstrap over prompts within each tag.

    rows: dicts with "tag", "idx" (prompt id) and `key` (bool). Returns {tag: (rate, lo, hi, n)} plus "overall".
    """
    rng = np.random.default_rng(seed)
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], {}).setdefault(r["idx"], []).append(float(r[key]))
    tags = [t for t in TAG_ORDER if t in by_tag]
    out = {}
    boots = np.zeros((n_boot, len(tags)))
    for j, t in enumerate(tags):
        per_prompt = [np.array(v) for v in by_tag[t].values()]
        sums = np.array([v.sum() for v in per_prompt])
        cnts = np.array([len(v) for v in per_prompt])
        rate = sums.sum() / cnts.sum()
        pick = rng.integers(0, len(per_prompt), size=(n_boot, len(per_prompt)))
        boots[:, j] = sums[pick].sum(1) / cnts[pick].sum(1)
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        out[t] = (rate, lo, hi, int(cnts.sum()))
    overall = np.mean([out[t][0] for t in tags])
    lo, hi = np.percentile(boots.mean(1), [2.5, 97.5])
    out["overall"] = (overall, lo, hi, sum(out[t][3] for t in tags))
    return out


def fmt_summary(s):
    lines = []
    for t in [*TAG_ORDER, "overall"]:
        if t in s:
            r, lo, hi, n = s[t]
            lines.append(f"{t:<14} {r:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
    return "\n".join(lines)


def modal_cls(name, **params):
    import modal

    cls = modal.Cls.from_name("t2i-geneval", name)
    return cls(**params) if params else cls()


def chunks(xs, n):
    return [xs[i : i + n] for i in range(0, len(xs), n)]
