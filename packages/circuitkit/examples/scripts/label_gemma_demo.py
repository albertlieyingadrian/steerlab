"""Label 50 Gemma-2-2B transcoder features with Claude Haiku 5.5 and compare with Neuronpedia's labels.

Features: the 21 from circuit-tracer's "capital of the state containing Dallas" graph that Neuronpedia hosts, plus
29 drawn at random (seed 0) from layers 4, 10, 16, 20 and 24. For each, Neuronpedia's activation records are the
examples. Negatives: the few silent records Neuronpedia keeps, plus 8 top records of other features in the set,
assumed silent for this one (transcoder features are sparse; the assumption is stated in the docs).

Every label is scored by the same detector (Haiku 5.5) on the same held-out records: ours, and the two labels
Neuronpedia shows (a long one from an LLM explainer and a short one). Outputs, all in examples/data/:

- labels_gemma_features.json.gz  the records used (so the notebook runs offline)
- labels_llm_cache.jsonl         every model reply (so reruns cost nothing)
- labels_gemma_results.json      labels, scores, calibration, comparisons, spend

    python examples/scripts/label_gemma_demo.py              # Claude Haiku 5.5
    python examples/scripts/label_gemma_demo.py --openai gpt-5-mini   # also an OpenAI explainer, same detector
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from pathlib import Path

from circuitkit.labels import FeatureData, Record, calibration, label_features, make_llm, score_labels
from circuitkit.labels.neuronpedia import compare, fetch_feature, neuronpedia_labels, records_from_neuronpedia

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL, SOURCE = "gemma-2-2b", "gemmascope-transcoder-16k"
DALLAS = ["20:15589", "21:5943", "23:12237", "16:25", "14:2268", "18:8959", "4:13154", "7:6861", "19:1445",
          "0:13727", "6:4012", "17:7178", "15:4494", "6:4662", "4:7671", "3:13984", "1:1000", "19:7477",
          "18:6101", "16:4298", "7:691"]  # fmt: skip


def load_env() -> None:
    """Read API keys from the nearest .env without printing them (only if not already set)."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        env = d / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def collect(cache: Path) -> tuple[list[FeatureData], dict[str, list[dict[str, str]]]]:
    rng = random.Random(0)
    units = list(DALLAS)
    while len(units) < 50:
        u = f"{rng.choice([4, 10, 16, 20, 24])}:{rng.randrange(16384)}"
        if u not in units:
            units.append(u)
    feats, theirs = [], {}
    for u in units:
        layer, idx = u.split(":")
        try:
            raw = fetch_feature(MODEL, f"{layer}-{SOURCE}", int(idx), cache)
        except Exception as e:  # a feature Neuronpedia does not host is skipped
            print(f"skip {u}: {e}")
            continue
        recs = records_from_neuronpedia(raw)
        if sum(r.peak > 0 for r in recs) < 16:
            print(f"skip {u}: only {sum(r.peak > 0 for r in recs)} activating records")
            continue
        feats.append(FeatureData(unit=u, records=recs))
        theirs[u] = neuronpedia_labels(raw)
    # negatives borrowed from other features (assumed silent here)
    pool = [(f.unit, r) for f in feats for r in sorted(f.records, key=lambda r: -r.peak)[:3]]
    for f in feats:
        others = [r for u, r in pool if u != f.unit]
        for r in random.Random(f.unit).sample(others, 8):
            f.records.append(Record(tokens=r.tokens, acts=[0.0] * len(r.tokens), source="other-feature"))
    return feats, theirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai", default=None, help="also label with this OpenAI model")
    args = ap.parse_args()
    load_env()
    feats_path = DATA / "labels_gemma_features.json.gz"
    if feats_path.exists():
        payload = json.loads(gzip.decompress(feats_path.read_bytes()))
        feats = [FeatureData.model_validate(x) for x in payload["features"]]
        theirs = payload["neuronpedia_labels"]
    else:
        feats, theirs = collect(Path.home() / ".cache" / "circuitkit" / "neuronpedia")
        payload = {"features": [f.model_dump() for f in feats], "neuronpedia_labels": theirs}
        feats_path.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    print(f"{len(feats)} features")
    cache = DATA / "labels_llm_cache.jsonl"
    haiku = make_llm("anthropic", "claude-haiku-5-5", cache)
    ours = label_features(haiku, feats, score="both")
    long_lab = {u: v[0]["description"] for u, v in theirs.items() if v}
    short_lab = {u: v[1]["description"] for u, v in theirs.items() if len(v) > 1}
    np_long = score_labels(haiku, feats, long_lab, score="both")
    np_short = score_labels(haiku, feats, short_lab, score="both")
    res = {
        "explainer": haiku.model,
        "detector": haiku.model,
        "n_features": len(feats),
        "ours": [x.model_dump() for x in ours],
        "neuronpedia_long": [x.model_dump() for x in np_long],
        "neuronpedia_short": [x.model_dump() for x in np_short],
        "neuronpedia_models": {u: [e["model"] for e in v] for u, v in theirs.items()},
        "calibration": calibration(ours).model_dump(),
        "compare_long": [c.model_dump() | {"winner": c.winner} for c in compare(ours, np_long)],
        "compare_short": [c.model_dump() | {"winner": c.winner} for c in compare(ours, np_short)],
    }
    if args.openai:
        oa = make_llm("openai", args.openai, cache)
        theirs_oa = label_features(oa, feats, score="none")
        scored = score_labels(haiku, feats, {x.unit: x.label for x in theirs_oa if x.label}, score="both")
        res["openai"] = {"model": args.openai, "labels": [x.model_dump() for x in scored],
                         "usage": oa.usage.__dict__}  # fmt: skip
    res["usage"] = haiku.usage.__dict__
    (DATA / "labels_gemma_results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")

    def mean(xs: list) -> float:
        v = [x.score for x in xs if x.score is not None]
        return sum(v) / max(len(v), 1)

    print(
        f"mean detection AUROC: ours {mean(ours):.3f}, neuronpedia long {mean(np_long):.3f}, short {mean(np_short):.3f}"
    )
    print(f"spend: {haiku.usage}")


if __name__ == "__main__":
    main()
