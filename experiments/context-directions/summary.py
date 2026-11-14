"""Apply the pre-registered decision rules to the results and write every number the note quotes.

    python experiments/context-directions/summary.py

Geometry: "stable" if S >= 0.9 at layer 14 and on average over layers 8-20; "context-dependent" if S < 0.7 (on both);
else "partial". Transfer: "transfers" if T >= 0.8 and the effect's interval lies above the largest random effect;
"fails" if T < 0.5 or the interval overlaps the random band; else "partial". A transfer cell is only a test when the target context's own direction beats the random band (its
interval above the largest random effect); otherwise the ratio has no denominator and the cell reads "no test" (this
precondition was not written down before the run; see the plan's changes). Writes results/summary.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
TAG = os.environ.get("CTX_TAG", "")
LAYER = int(os.environ.get("CTX_LAYER", "14"))


def geometry_verdict(s_layer: float, s_mid: float) -> str:
    if s_layer >= 0.9 and s_mid >= 0.9:
        return "stable"
    if s_layer < 0.7 and s_mid < 0.7:
        return "context-dependent"
    return "partial"


def transfer_verdict(e: dict, random_max: float) -> str:
    above = e["ci"][0] > random_max
    if e["T"] >= 0.8 and above:
        return "transfers"
    if e["T"] < 0.5 or not above:
        return "fails"
    return "partial"


def main() -> None:
    g = json.loads((R / f"geometry{TAG}.json").read_text())
    n = g["n_layers"]
    MID = slice(round(8 / 28 * n), round(21 / 28 * n))  # layers 8-20 of 28, the same depths on a deeper model
    out: dict = {"geometry": {}, "transfer": {}, "controls": dict(g["controls"])}
    out["controls"].pop("cross_concept", None)
    out["controls"]["cross_concept_L"] = {k: v[LAYER] for k, v in g["controls"]["cross_concept"].items()}
    for concept, row in g["concepts"].items():
        out["geometry"][concept] = {}
        for key, v in row["pairs"].items():
            s_l, s_m = v["S"][LAYER], float(np.nanmean(v["S"][MID]))
            out["geometry"][concept][key] = {
                "S_layer": s_l, "S_low": v["S_low"][LAYER], "S_high": v["S_high"][LAYER], "S_mid": round(s_m, 4),
                "full_cos": v["full_cos"][LAYER], "verdict": geometry_verdict(s_l, s_m),
            }  # fmt: skip
        out["geometry"][concept]["ceiling"] = {x: v[LAYER] for x, v in row["ceiling"].items()}
        out["geometry"][concept]["shuffled_vs_real"] = row["shuffled_vs_real_plain"][LAYER]
    for extra in ("geometry_controls", "robustness"):
        path = R / f"{extra}{TAG}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        flat = {}
        for concept, block in data.items():
            if extra == "geometry_controls":
                items = list(block.items())
            else:
                items = [(f"{kind}:{k}", v) for kind, d in block.items() for k, v in d.items()]
            flat[concept] = {k: {"S_layer": v[LAYER], "S_mid": round(float(np.nanmean(v[MID])), 4)} for k, v in items}
        out[extra] = flat
    for name, label in (("steer", ""), ("steer_truth_capitals", "truth_capitals")):
        steer_path = R / f"{name}{TAG}.json"
        if not steer_path.exists():
            continue
        st = json.loads(steer_path.read_text())
        for concept, row in st["concepts"].items():
            concept = label or concept
            out["transfer"][concept] = {"strength": row["strength"], "cells": {}}
            for y, t in row["targets"].items():
                rmax = max(t["random"])
                own_ok = t["effects"][y]["ci"][0] > rmax  # the target's own direction must beat the random band
                for x, e in t["effects"].items():
                    out["transfer"][concept]["cells"][f"{x}->{y}"] = {
                        **e, "random_max": rmax, "unsteered": t["unsteered"],
                        "verdict": ("own" if own_ok else "own fails") if x == y
                        else (transfer_verdict(e, rmax) if own_ok else "no test"),
                    }  # fmt: skip
    (R / f"summary{TAG}.json").write_text(json.dumps(out, indent=1) + "\n")
    for concept, pairs in out["geometry"].items():
        print(
            concept,
            {
                k: (v["S_layer"], v["S_mid"], v["verdict"])
                for k, v in pairs.items()
                if isinstance(v, dict) and "verdict" in v
            },
        )
    for concept, row in out["transfer"].items():
        print(concept, {k: (v["E"], v["T"], v["verdict"]) for k, v in row["cells"].items()})


if __name__ == "__main__":
    main()
