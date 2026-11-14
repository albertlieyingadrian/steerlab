"""Neuronpedia in and out: fetch a feature's activation records and labels, export ours in its format, compare.

Neuronpedia hosts dashboards for many SAE and transcoder sets (for example Gemma-2-2B's GemmaScope transcoders:
model `gemma-2-2b`, source `20-gemmascope-transcoder-16k` for layer 20). A feature's page carries its top activating
records, some records from lower activation bins, and labels written by other explainers. That is enough to label
features with circuitkit and compare the result with Neuronpedia's label on equal terms: both labels are scored by
the same detector on the same held-out records.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from circuitkit.labels.autointerp import FeatureData, FeatureLabel, Record

API = "https://www.neuronpedia.org/api/feature"


def fetch_feature(model_id: str, source: str, index: int, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """GET one feature's JSON from Neuronpedia's public API (no key needed); cached on disk when `cache_dir` is set."""
    cache = Path(cache_dir) / f"{model_id}__{source}__{index}.json" if cache_dir else None
    if cache is not None and cache.exists():
        out: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8"))
        return out
    req = urllib.request.Request(f"{API}/{model_id}/{source}/{index}", headers={"User-Agent": "circuitkit"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    if "error" in data:
        raise ValueError(f"Neuronpedia: {data['error']}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def records_from_neuronpedia(feature: Mapping[str, Any]) -> list[Record]:
    """Activation records (tokens and per-token values) from a Neuronpedia feature JSON, duplicates removed."""
    out: list[Record] = []
    seen: set[tuple[str, ...]] = set()
    for a in feature.get("activations", []):
        toks = [str(t) for t in a.get("tokens", [])]
        vals = [float(v) for v in a.get("values", [])]
        if not toks or len(toks) != len(vals) or tuple(toks) in seen:
            continue
        seen.add(tuple(toks))
        out.append(Record(tokens=toks, acts=[max(0.0, v) for v in vals], source="neuronpedia"))
    return out


def neuronpedia_labels(feature: Mapping[str, Any]) -> list[dict[str, str]]:
    """The labels Neuronpedia shows for a feature: description, explainer model and method."""
    return [
        {
            "description": str(e.get("description", "")).strip(),
            "model": str(e.get("explanationModelName", "")),
            "type": str(e.get("typeName", "")),
        }
        for e in feature.get("explanations", [])
    ]


def feature_data(
    model_id: str, source: str, index: int, unit: str | None = None, cache_dir: Path | str | None = None
) -> tuple[FeatureData, list[dict[str, str]]]:
    """Fetch a feature and return its records as `FeatureData` plus Neuronpedia's own labels."""
    f = fetch_feature(model_id, source, index, cache_dir)
    u = unit or f"{source.split('-')[0]}:{index}"
    return FeatureData(unit=u, records=records_from_neuronpedia(f)), neuronpedia_labels(f)


def export_explanations(
    labels: Sequence[FeatureLabel], model_id: str, source: str, path: Path | str | None = None
) -> list[dict[str, Any]]:
    """Our labels as Neuronpedia explanation rows (`modelId`, `layer`, `index`, `description`,
    `explanationModelName`, `typeName`, `scores`). `layer` is Neuronpedia's source id, e.g. `20-gemmascope-...`.

    Units are `"layer:feature"`; the layer in `source` must match. Written as JSON if `path` is given.
    """
    rows = []
    for lab in labels:
        if not lab.label:
            continue
        layer, feat = lab.unit.split(":")
        src = source if source.split("-")[0] == layer else f"{layer}-{source.split('-', 1)[1]}"
        scores = []
        if lab.detection is not None and lab.detection.auroc is not None:
            scores.append({"method": "circuitkit-detection-auroc", "value": lab.detection.auroc})
        if lab.simulation is not None and lab.simulation.correlation is not None:
            scores.append({"method": "circuitkit-simulation-corr", "value": lab.simulation.correlation})
        rows.append(
            {
                "modelId": model_id,
                "layer": src,
                "index": str(int(feat)),
                "description": lab.label,
                "explanationModelName": lab.model,
                "typeName": "circuitkit-token-act",
                "scores": scores,
            }
        )
    if path is not None:
        Path(path).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return rows


class Comparison(BaseModel):
    unit: str
    ours: str
    theirs: str
    theirs_model: str
    ours_score: float | None
    theirs_score: float | None

    @property
    def winner(self) -> str:
        if self.ours_score is None or self.theirs_score is None:
            return "n/a"
        if abs(self.ours_score - self.theirs_score) < 0.05:
            return "tie"
        return "ours" if self.ours_score > self.theirs_score else "theirs"


def compare(
    ours: Sequence[FeatureLabel], theirs: Sequence[FeatureLabel], theirs_model: Mapping[str, str] | None = None
) -> list[Comparison]:
    """Pair our scored labels with Neuronpedia's (scored by `score_labels` with the same detector) by unit."""
    t = {lab.unit: lab for lab in theirs}
    out = []
    for o in ours:
        if o.unit not in t:
            continue
        out.append(
            Comparison(
                unit=o.unit,
                ours=o.label,
                theirs=t[o.unit].label,
                theirs_model=(theirs_model or {}).get(o.unit, "neuronpedia"),
                ours_score=o.score,
                theirs_score=t[o.unit].score,
            )
        )
    return out
