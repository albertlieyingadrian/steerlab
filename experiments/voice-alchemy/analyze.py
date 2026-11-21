"""Every reported number for a Voice Alchemy round: measured profiles, polysemanticity, composition labels, geometry,
the type comparison and the pre-registered predictions. Reads results/scored_<round>.jsonl, results/selection.json and
the unit vectors; writes results/analysis_<round>.json.

    python analyze.py heldout      # also: pilot, dev4 (for the circuit-pair choice)
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import pipeline
import va

TRAITS = list(va.TRAITS)
B = 2000
TYPES_ALL = ("prompt", "direction", "neuron", "sae", "community", "circuit")


def boot_ci(vals: list[float], rng: random.Random) -> list[float]:
    if not vals:
        return [float("nan"), float("nan")]
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]


def boot_median(v: list[float], rng: random.Random) -> list[float]:
    n = len(v)
    meds = sorted(st.median([v[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return [meds[int(0.025 * B)], meds[int(0.975 * B) - 1]]


def nanmean(v: list[float]) -> float | None:
    v = [x for x in v if not math.isnan(x)]
    return st.mean(v) if v else None


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


class Round:
    def __init__(self, name: str, judge_name: str = va.JUDGE):
        self.name, self.j = name, judge_name
        self.rows = pipeline.load_scored(name)
        self.base = pipeline.baseline(self.rows, judge_name)
        self.by = defaultdict(list)
        for r in self.rows:
            self.by[(r["kind"], tuple(r["uids"]), tuple(r["s"]))].append(r)

    def delta(self, r: dict, a: str) -> float | None:
        v = r["scores"].get(self.j, {}).get(a)
        b = self.base.get((r["pi"], a))
        return None if v is None or b is None else v - b

    def profile(self, rs: list[dict], rng: random.Random, aspects=TRAITS) -> dict:
        out: dict = {"d": {}, "ci": {}, "n": len(rs)}
        for a in aspects:
            ds = [x for x in (self.delta(r, a) for r in rs) if x is not None]
            out["d"][a] = st.mean(ds) if ds else float("nan")
            out["ci"][a] = boot_ci(ds, rng)
        for k in ("coherence", "honesty"):
            v = [r["scores"][self.j].get(k) for r in rs]
            v = [x for x in v if x is not None]
            out[k] = st.mean(v) if v else None
        out["loop_share"] = sum(r["loops"] for r in rs) / max(len(rs), 1)
        out["words"] = st.mean(len(r["text"].split()) for r in rs) if rs else 0
        out["empty"] = sum(not r["text"].strip() for r in rs)
        return out


def load_vectors() -> dict:
    import torch

    units = {}
    for p in ("units.pt", "circuits.pt"):
        f = va.RESULTS / p
        if f.exists():
            for u, d in torch.load(f).items():
                units[u] = {int(L): v.float().numpy() for L, v in d["parts"].items()}
    return units


def geometry(units: dict, ua: str, ub: str, R: list[float], WU) -> dict:
    layers = sorted(set(units[ua]) | set(units[ub]))
    z = np.zeros_like(next(iter(units[ua].values())))
    va_ = np.concatenate([units[ua].get(L, z) / R[L] for L in layers])
    vb_ = np.concatenate([units[ub].get(L, z) / R[L] for L in layers])
    out = {"cos": float(va_ @ vb_ / (np.linalg.norm(va_) * np.linalg.norm(vb_) + 1e-9))}
    if WU is not None:
        sa = WU @ sum(units[ua].values())
        sb = WU @ sum(units[ub].values())
        sa, sb = sa - sa.mean(), sb - sb.mean()
        out["logit_cos"] = float(sa @ sb / (np.linalg.norm(sa) * np.linalg.norm(sb) + 1e-9))
    return out


def load_wu():
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        idx = json.loads(Path(hf_hub_download(va.MODEL, "model.safetensors.index.json")).read_text())
        shard = idx["weight_map"]["model.embed_tokens.weight"]
        with safe_open(hf_hub_download(va.MODEL, shard), "pt") as f:
            return f.get_tensor("model.embed_tokens.weight").float().numpy()
    except Exception as e:
        print(f"no unembedding ({type(e).__name__}); logit cosine skipped")
        return None


def pair_type(uids: tuple) -> str:
    u = uids[0]
    if u.startswith("prompt-"):
        return "prompt"
    if u.startswith("rand_circuit-"):
        return "rand_circuit"
    if u.startswith("circuit-"):
        return "circuit"
    return u.split("-")[1]


def analyse(name: str) -> dict:
    rng = random.Random(0)
    rd = Round(name)
    sel = json.loads((va.RESULTS / "selection.json").read_text())
    R = pipeline.meta()["R"]
    pis = sorted({r["pi"] for r in rd.rows})
    out: dict = {"round": name, "n_prompts": len(pis), "units": {}, "pairs": [], "types": {}}
    out["judge_missing"] = {
        "missing": sum(v is None for r in rd.rows for v in r["scores"][va.JUDGE].values()),
        "calls": sum(len(r["scores"][va.JUDGE]) for r in rd.rows),
    }
    out["unsteered_raw"] = {
        a: st.mean(rd.base[(pi, a)] for pi in pis if rd.base.get((pi, a)) is not None)
        for a in [*TRAITS, "coherence", "honesty"]
    }

    # --- singles: profiles at every strength, controls, polysemanticity
    for key, s in sel.items():
        uid, t = s["uid"], s["trait"]
        u: dict = {
            "type": s["type"],
            "trait": t,
            "uid": uid,
            "control": s["control"],
            "s_op": s["s"],
            "by_s": {},
            "control_by_s": {},
        }
        for (kind, uids, ss), rs in rd.by.items():
            if uids == (uid,) and kind in ("single", "single_full"):
                u["by_s"][str(ss[0])] = rd.profile(rs, rng)
            if uids == (s["control"],) and kind == "control":  # control_full repeats the op strength for random pairs
                c = rd.profile(rs, rng, [t])
                u["control_by_s"][str(ss[0])] = {
                    "d": c["d"][t],
                    "ci": c["ci"][t],
                    "coherence": c["coherence"],
                    "loop_share": c["loop_share"],
                }
        op = u["by_s"].get(str(s["s"]))
        if op:
            u["target_op"] = op["d"][t]
            u["breadth"] = sum(abs(op["d"][a]) >= 1.5 for a in TRAITS)
            u["top_trait_op"] = max(TRAITS, key=lambda a: op["d"][a])
            ctrl = u["control_by_s"].get(str(s["s"]))
            u["beats_control"] = bool(ctrl and op["d"][t] - ctrl["d"] >= 2.0)
            srt = sorted(u["by_s"], key=float)
            lo, hi = u["by_s"][srt[0]], u["by_s"][srt[-1]]
            u["strength_stability"] = spearman([lo["d"][a] for a in TRAITS], [hi["d"][a] for a in TRAITS])
            u["top_trait_by_s"] = {
                k: max(TRAITS, key=lambda a, p=v: p["d"][a])
                for k, v in u["by_s"].items()
                if all(v["d"][a] == v["d"][a] for a in TRAITS)
            }  # full profiles only
            cats = defaultdict(list)
            for r in rd.by.get(("single", (uid,), (s["s"],)), []):
                d = rd.delta(r, t)
                if d is not None:
                    cats[r["cat"]].append(d)
            u["context"] = {c: st.mean(v) for c, v in cats.items()}
        out["units"][key] = u
    for t in TRAITS:  # prompting baseline
        rs = rd.by.get(("prompt", (f"prompt-{t}",), ()), [])
        if rs:
            p = rd.profile(rs, rng)
            out["units"][f"prompt-{t}"] = {
                "type": "prompt",
                "trait": t,
                "uid": f"prompt-{t}",
                "s_op": None,
                "by_s": {"op": p},
                "target_op": p["d"][t],
                "breadth": sum(abs(p["d"][a]) >= 1.5 for a in TRAITS),
                "top_trait_op": max(TRAITS, key=lambda a: p["d"][a]),
            }

    # --- pairs
    units = load_vectors()
    WU = load_wu() if name == "heldout" and va.TAG == "gemma" else None

    def single_op(typ: str, t: str) -> dict | None:
        u = out["units"].get(f"{typ}-{t}")
        if not u:
            return None
        return u["by_s"].get("op") if typ == "prompt" else u["by_s"].get(str(u["s_op"]))

    for (kind, uids, ss), rs in rd.by.items():
        if kind not in ("pair", "control_pair"):
            continue
        ta, tb = rs[0]["traits"]
        typ = pair_type(uids)
        if kind == "pair":
            pa, pb = single_op(typ, ta), single_op(typ, tb)
        else:
            ra = rd.by.get(("control_full", (uids[0],), (ss[0],)))
            rb = rd.by.get(("control_full", (uids[1],), (ss[1],)))
            pa = rd.profile(ra, rng) if ra else None
            pb = rd.profile(rb, rng) if rb else None
        if pa is None or pb is None:
            continue
        pab = rd.profile(rs, rng)
        da, db, dab = ([p["d"][a] for a in TRAITS] for p in (pa, pb, pab))
        lo = []
        for i, a in enumerate(TRAITS):
            ds = [x - da[i] - db[i] for x in (rd.delta(r, a) for r in rs) if x is not None]
            lo.append(boot_ci(ds, rng)[0])
        E = va.additivity_error(da, db, dab)
        lab, info = va.composition_label(TRAITS, ta, tb, da, db, dab, lo, pab["loop_share"], pab["coherence"] or 0)
        loose, _ = va.composition_label(TRAITS, ta, tb, da, db, dab, lo, 0.0, 10.0, strict=False)
        eligible = da[TRAITS.index(ta)] >= va.MIN_SINGLE and db[TRAITS.index(tb)] >= va.MIN_SINGLE
        # post hoc (added after the held-out run): the sum clipped to the 0-10 scale per prompt, so two units that
        # each lower a trait cannot be credited with a rise just because the trait hit the floor
        pis_ = sorted({r["pi"] for r in rs})
        clip_pred = [
            st.mean(
                min(10, max(0, rd.base[(pi, a)] + da[i] + db[i])) - rd.base[(pi, a)]
                for pi in pis_
                if rd.base.get((pi, a)) is not None
            )
            for i, a in enumerate(TRAITS)
        ]
        lo_c = []
        for i, a in enumerate(TRAITS):
            ds = [x - clip_pred[i] for x in (rd.delta(r, a) for r in rs) if x is not None]
            lo_c.append(boot_ci(ds, rng)[0])
        E_clip = va.norm([y - p for y, p in zip(dab, clip_pred, strict=True)]) / max(va.norm(da) + va.norm(db), 1e-9)
        rise_c = {a: dab[i] - clip_pred[i] for i, a in enumerate(TRAITS) if a not in (ta, tb)}
        best_c = max(rise_c, key=rise_c.get)
        emergent_clip = best_c if rise_c[best_c] >= va.EMERGE and lo_c[TRAITS.index(best_c)] > 0 else None
        pair = {
            "type": typ,
            "kind": kind,
            "uids": list(uids),
            "traits": [ta, tb],
            "s": list(ss),
            "E": E,
            "label": lab,
            "label_loose": loose,
            "eligible": eligible,
            "d_a": da,
            "d_b": db,
            "d_ab": dab,
            "lo_rise": lo,
            "coherence": pab["coherence"],
            "honesty": pab["honesty"],
            "loop_share": pab["loop_share"],
            "words": pab["words"],
            **info,
            "pair_coh_singles": [pa["coherence"], pb["coherence"]],
            "posthoc_E_clipped": E_clip,
            "posthoc_emergent_clipped": emergent_clip,
        }
        if typ != "prompt" and all(u in units for u in uids):
            pair.update(geometry(units, uids[0], uids[1], R, WU))
        out["pairs"].append(pair)

    # --- per type summaries
    for typ in [*TYPES_ALL, "rand_direction", "rand_neuron", "rand_sae", "rand_community", "rand_circuit"]:
        ps = [p for p in out["pairs"] if p["type"] == typ]
        us = [u for u in out["units"].values() if u["type"] == typ and "target_op" in u]
        if not ps and not us:
            continue
        el = [p for p in ps if p["eligible"]]
        Es = [p["E"] for p in el]
        xs = [x for p in el for x in (np.array(p["d_a"]) + np.array(p["d_b"])).tolist()]
        ys = [y for p in el for y in p["d_ab"]]
        r2 = (
            (1 - sum((y - x) ** 2 for x, y in zip(xs, ys, strict=True)) / sum((y - st.mean(ys)) ** 2 for y in ys))
            if len(ys) > 2
            else None
        )
        labels, labels_all = defaultdict(int), defaultdict(int)
        for p in el:
            labels[p["label"]] += 1
        for p in ps:
            labels_all[p["label"]] += 1
        out["types"][typ] = {
            "n_units": len(us),
            "n_pairs": len(ps),
            "n_eligible": len(el),
            "units_moving": sum(u["target_op"] >= va.MIN_SINGLE for u in us),
            "mean_target_op": st.mean(u["target_op"] for u in us) if us else None,
            "median_E": st.median(Es) if Es else None,
            "median_E_ci": boot_median(Es, rng) if len(Es) > 1 else None,
            "median_E_all_pairs": st.median([p["E"] for p in ps]) if ps else None,
            "r2_sum": r2,
            "labels_eligible": dict(labels),
            "labels_all": dict(labels_all),
            "additive_share": (labels["additive"] / len(el)) if el else None,
            "mean_breadth": st.mean(u["breadth"] for u in us) if us else None,
            "mean_strength_stability": nanmean([u.get("strength_stability", float("nan")) for u in us]),
            "beats_control": sum(bool(u.get("beats_control")) for u in us),
            "pair_coherence": st.mean(p["coherence"] for p in ps if p["coherence"] is not None) if ps else None,
            "pair_loop_share": st.mean(p["loop_share"] for p in ps) if ps else None,
            "posthoc_median_E_clipped": st.median([p["posthoc_E_clipped"] for p in el]) if el else None,
            "posthoc_emergent_strict": sum(p["label"] == "emergent" for p in el),
            "posthoc_emergent_after_clipping": sum(p["posthoc_emergent_clipped"] is not None for p in el),
        }
    geo = [p for p in out["pairs"] if "cos" in p and p["kind"] == "pair" and p["eligible"]]
    geo_all = [p for p in out["pairs"] if "cos" in p and p["kind"] == "pair"]
    out["geometry"] = {
        "n": len(geo),
        "spearman_abs_cos_E": spearman([abs(p["cos"]) for p in geo], [p["E"] for p in geo]),
        "spearman_logit_cos_E": spearman([abs(p["logit_cos"]) for p in geo], [p["E"] for p in geo])
        if geo and all("logit_cos" in p for p in geo)
        else None,
        "n_all": len(geo_all),
        "spearman_abs_cos_E_all": spearman([abs(p["cos"]) for p in geo_all], [p["E"] for p in geo_all]),
    }
    out["predictions"] = predictions(out)
    (va.RESULTS / f"analysis_{name}{va.SUFFIX}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def predictions(out: dict) -> list[dict]:
    T = out["types"]

    def g(t, k):
        return T.get(t, {}).get(k)

    def diff(x, y):
        return None if x is None or y is None else x - y

    v1 = diff(g("neuron", "median_E"), g("community", "median_E"))
    v2 = diff(g("sae", "median_E"), g("community", "median_E"))
    v3 = diff(g("neuron", "mean_breadth"), g("community", "mean_breadth"))
    v4 = out["geometry"]["spearman_abs_cos_E_all"]  # the plan: "over all steered pairs"
    v5 = g("neuron", "beats_control")
    return [
        {
            "id": "P1",
            "metric": "median E(neuron) - median E(community)",
            "threshold": "> 0.10",
            "value": v1,
            "hit": None if v1 is None else v1 > 0.10,
            "n": [g("neuron", "n_eligible"), g("community", "n_eligible")],
        },
        {
            "id": "P2",
            "metric": "median E(SAE) - median E(community)",
            "threshold": "<= 0.05",
            "value": v2,
            "hit": None if v2 is None else v2 <= 0.05,
            "n": [g("sae", "n_eligible"), g("community", "n_eligible")],
        },
        {
            "id": "P3",
            "metric": "mean breadth(neuron) - mean breadth(community)",
            "threshold": ">= 0.5",
            "value": v3,
            "hit": None if v3 is None else v3 >= 0.5,
        },
        {
            "id": "P4",
            "metric": "Spearman(|cos|, E), all steered pairs",
            "threshold": ">= 0.30",
            "value": v4,
            "hit": None if v4 is None or math.isnan(v4) else v4 >= 0.30,
            "n": out["geometry"]["n_all"],
        },
        {
            "id": "P5",
            "metric": "neurons beating their control by >= 2 points",
            "threshold": "< 4 of 8",
            "value": v5,
            "hit": None if v5 is None else v5 < 4,
        },
    ]


if __name__ == "__main__":
    a = analyse(sys.argv[1])
    for typ, v in a["types"].items():
        print(
            f"{typ:15s} units {v['n_units']} moving {v['units_moving']} pairs {v['n_pairs']} eligible "
            f"{v['n_eligible']} medE {v['median_E']} all {v['median_E_all_pairs']} labels {v['labels_eligible']} "
            f"breadth {v['mean_breadth']} beats {v['beats_control']}"
        )
    print("geometry", a["geometry"])
    for p in a["predictions"]:
        print(p)
