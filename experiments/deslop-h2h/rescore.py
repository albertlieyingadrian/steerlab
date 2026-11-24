"""Addendum (2026-11-24): re-score every arm by its distance to human writing, not by "lower is better".

    python rescore.py    # -> results/rescore.json, results/rescore_table.csv

The habit score ranks arms by how few habits they keep, and the "human band" counts a reply as human-like when it
scores at or below the 90th percentile of human passages, so a reply with no habits at all is always "in band". This
script asks the other question: where does each arm sit inside the human distribution, metric by metric, and does any
arm land *below* it (fewer em dashes, fewer lists of three, shorter sentences than people write)?

Reference groups:
  classic  the 318 human passages the habit band was built on (Episode 1's `data/human/passages.jsonl`); every metric.
  modern   slop-rules' 4,443 modern human windows (the v2 band: `../slop-rules/results/window_scores.csv`, human side,
           train split); only their total habit scores (v1 and v2 rules) are saved, so only the totals are compared.

Per reply and metric, the score is the reply's mid-rank percentile in the reference (share of human values below it,
plus half the share equal to it), so a human passage sits at 50 on average. Per arm: the mean over prompts (Track A's
two samples are averaged within a prompt first, as table.py does), and its signed shift from 50. Negative means the
arm sits below the human centre on that metric. The arm's distance to human writing is the mean absolute shift over
the 14 metrics (11 habits, MATTR, mean sentence length, sentence-length spread). Intervals: 95% bootstrap over
prompts (10,000 draws, seed 0), with the whole metric vector resampled together. A robustness version replaces
percentiles with the arm mean's distance from the human mean in human standard deviations (|z|).

A metric is "below human" for an arm when the upper end of its shift interval is under 0; "overcorrected" when, in
addition, the track's untouched model was not below human on it (so the method pushed it there). "Above human" is the
mirror image (lower end over 0).

Amendment, written after the first run and reported as such: zero-inflated habits made the rule above flag shifts of
half a percentile point (a habit 99% of human passages never use sits at 49.5 when absent), so `overcorrected_5` also
requires the shift's upper end to be at least 5 points below 0. Two sensitivity versions of the distance are also
added after the first run: over the 11 habits alone (the three shape metrics put every arm, the untouched models
included, 40-49 points below the classic passages, whose sentences average 27 words), and `distance_z` above.
"""

from __future__ import annotations

import csv
import json
import re

import numpy as np
from deslop.checks import mattr
from deslop.habits import DETECTORS as HABITS
from deslop.score import score
from scipy.stats import spearmanr

from common import EP1, GEN, HERE, RESULTS, SEED, TRACK_B, B, held_out, human_passages, read_jsonl, save_json
from table import TRACK_A_ORDER

SHAPE = ("mattr", "sent_mean", "sent_sd")
MARGIN = 5.0  # amendment: percentile points below the human centre needed to call a metric overcorrected
METRICS = tuple(HABITS) + SHAPE  # the 14 metrics of the distance
TOTALS = ("habit_v1", "habit_v2")
LABELS = {"em_dash": "em dashes", "not_x_but_y": "not X but Y", "triad": "lists of three",
          "rhetorical_qa": "self-answered questions", "signpost": "signposts", "stock_phrase": "stock phrases",
          "fragment": "emphasis fragments", "which_chain": "', which' chains", "slop_word": "slop words",
          "formatting": "formatting", "invented_claim": "invented claims", "mattr": "word variety (MATTR)",
          "sent_mean": "sentence length", "sent_sd": "sentence-length spread",
          "habit_v1": "habit score v1", "habit_v2": "habit score v2"}  # fmt: skip

# Written before any number below was computed (2026-11-24). Arms are Track A's 12 methods unless named.
PREDICTIONS = [
    {"id": "R1", "text": "prompt + sampler and SFT + sampler sit below the human centre on the v1 habit total "
     "(classic reference)", "threshold": "upper end of the shift interval < 0 for both"},
    {"id": "R2", "text": "Both overcorrect on at least 2 of the 14 metrics, em dashes and lists of three among them",
     "threshold": ">= 2 overcorrected metrics each, including em_dash and triad"},
    {"id": "R3", "text": "Ranking by distance to human writing changes the winner: SFT + sampler (lowest habit score) "
     "is not the closest arm to human", "threshold": "SFT + sampler not rank 1 by distance"},
    {"id": "R4", "text": "Arms further from human writing lose more often to the original in the blind judge",
     "threshold": "Spearman(distance, judge loss share) > 0.3 over the 12 methods"},
]  # fmt: skip

SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")


def sentences(text: str) -> list[int]:
    """Words per sentence; a line break also ends a sentence (headings, list items)."""
    lens = [len(WORD.findall(s)) for s in SENT.split(text)]
    return [n for n in lens if n > 0]


def features(text: str) -> dict[str, float]:
    s1, s2 = score(text, band=None, rules="v1"), score(text, band=None, rules="v2")
    lens = sentences(text) or [0]
    out = {h: s1.per_1k[h] for h in HABITS}
    out |= {"mattr": mattr(text), "sent_mean": float(np.mean(lens)), "sent_sd": float(np.std(lens)),
            "habit_v1": s1.score, "habit_v2": s2.score, "words": s1.words}  # fmt: skip
    return out


def midrank(ref: np.ndarray, x: float) -> float:
    """Percentile (0-100) of x in the sorted reference: share below plus half the share equal."""
    lo, hi = np.searchsorted(ref, x, "left"), np.searchsorted(ref, x, "right")
    return 100.0 * (lo + hi) / (2 * len(ref))


def modern_reference() -> dict[str, np.ndarray]:
    path = HERE.parent / "slop-rules" / "results" / "window_scores.csv"
    rows = [r for r in csv.DictReader(path.open())
            if r["side"] == "human" and r["split"] == "train" and r["set"] in ("matched", "raid")]  # fmt: skip
    assert len(rows) == 4443, len(rows)  # the v2 band's 'modern' group
    return {"habit_v1": np.sort([float(r["score_v1"]) for r in rows]),
            "habit_v2": np.sort([float(r["score_v2"]) for r in rows])}  # fmt: skip


def groups() -> dict[str, list[dict]]:
    out = {f"A/{a}": read_jsonl(GEN / f"{a}.jsonl") for a in TRACK_A_ORDER}
    out |= {f"B/{r}": read_jsonl(EP1 / "results" / "gen" / f"{r}.jsonl") for r in TRACK_B}
    return out


def cell(value: float, draws: np.ndarray) -> dict:
    return {"value": round(value, 2), "lo": round(float(np.percentile(draws, 2.5)), 2),
            "hi": round(float(np.percentile(draws, 97.5)), 2)}  # fmt: skip


def main() -> None:
    pids = sorted(p["id"] for p in held_out())
    human = [features(h["text"]) for h in human_passages()]
    classic = {m: np.sort([h[m] for h in human]) for m in METRICS + TOTALS}
    modern = modern_reference()
    hmean = np.array([np.mean(classic[m]) for m in METRICS])
    hsd = np.array([np.std(classic[m]) or 1.0 for m in METRICS])
    idx = np.random.default_rng(SEED).integers(0, len(pids), size=(B, len(pids)))
    vpath = RESULTS / "judge" / "verdicts.json"
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else []

    arms = {}
    for g, rs in groups().items():
        feats = [(r["id"], features(r["text"])) for r in rs]

        def per_prompt(fn, feats=feats) -> np.ndarray:
            return np.array([np.mean([fn(f) for i, f in feats if i == p]) for p in pids])

        pct = np.stack([per_prompt(lambda f, m=m: midrank(classic[m], f[m])) for m in METRICS], 1)  # (48, 14)
        raw = np.stack([per_prompt(lambda f, m=m: f[m]) for m in METRICS], 1)
        shift, draws = pct.mean(0) - 50, pct[idx].mean(1) - 50  # (14,), (B, 14)
        z, zdraws = np.abs((raw.mean(0) - hmean) / hsd), np.abs((raw[idx].mean(1) - hmean) / hsd)
        nh = len(HABITS)
        out = {"replies": len(rs), "distance": cell(float(np.abs(shift).mean()), np.abs(draws).mean(1)),
               "distance_habits": cell(float(np.abs(shift[:nh]).mean()), np.abs(draws[:, :nh]).mean(1)),
               "distance_z": cell(float(z.mean()), zdraws.mean(1)), "metrics": {}}  # fmt: skip
        for j, m in enumerate(METRICS):
            out["metrics"][m] = {"shift": cell(float(shift[j]), draws[:, j]),
                                 "arm_mean": round(float(raw[:, j].mean()), 3)}  # fmt: skip
        for m in TOTALS:
            out[m] = round(float(per_prompt(lambda f, m=m: f[m]).mean()), 3)
            out[f"{m}_zero_share"] = round(float(np.mean([f[m] == 0 for _, f in feats])), 3)
            for ref_name, ref in (("classic", classic[m]), ("modern", modern[m])):
                v = per_prompt(lambda f, m=m, ref=ref: midrank(ref, f[m])) - 50
                out[f"{m}_{ref_name}"] = cell(float(v.mean()), v[idx].mean(1))
        out["words"] = round(float(np.mean([f["words"] for _, f in feats])), 1)
        track, arm = g.split("/", 1)
        vs = [v["result"] for v in verdicts if v["track"] == track and v["arm"] == arm]
        if vs:
            out["judge_loss_share"] = round(vs.count("loss") / len(vs), 3)
        out["_draws"] = np.abs(draws).mean(1)
        arms[g] = out

    for track, base in (("A", "A/original"), ("B", "B/base")):
        named = [g for g in arms if g.startswith(track + "/")]
        for g in named:
            for m, c in arms[g]["metrics"].items():
                c["below"], c["above"] = c["shift"]["hi"] < 0, c["shift"]["lo"] > 0
                c["over"] = c["below"] and g != base and not arms[base]["metrics"][m]["shift"]["hi"] < 0
                c["over_5"] = c["over"] and c["shift"]["hi"] < -MARGIN
            for key, test in (("overcorrected", "over"), ("overcorrected_5", "over_5"), ("below_human", "below"),
                              ("above_human", "above")):  # fmt: skip
                arms[g][key] = [m for m, c in arms[g]["metrics"].items() if c[test]]
        for i, g in enumerate(sorted(named, key=lambda g: arms[g]["habit_v1"]), 1):
            arms[g]["rank_habit"] = i
        for i, g in enumerate(sorted(named, key=lambda g: arms[g]["distance"]["value"]), 1):
            arms[g]["rank_distance"] = i

    summary = json.loads((RESULTS / "summary.json").read_text())["arms"]
    for g, a in arms.items():
        a["loop"] = summary[g]["loop"]["value"]
    paired = {}
    for x, y in (("A/sft+sampler", "A/dpo"), ("A/plain+sampler", "A/sft"), ("A/plain+sampler", "A/original"),
                 ("A/plain+sampler", "A/plain"), ("A/sft+sampler", "A/sft"), ("B/kannike", "B/lora-r2")):  # fmt: skip
        d = arms[x]["_draws"] - arms[y]["_draws"]
        paired[f"{x} - {y}"] = cell(arms[x]["distance"]["value"] - arms[y]["distance"]["value"], d)
    for a in arms.values():
        del a["_draws"]
    methods = [f"A/{a}" for a in TRACK_A_ORDER if a != "original"]
    loss = [arms[g]["judge_loss_share"] for g in methods]
    rho = spearmanr([arms[g]["distance"]["value"] for g in methods], loss)
    rho_z = spearmanr([arms[g]["distance_z"]["value"] for g in methods], loss)
    rho_h = spearmanr([arms[g]["habit_v1"] for g in methods], loss)
    checks = {"paired_distance": paired,
              "spearman_distance_judge_loss": round(float(rho.statistic), 3),
              "spearman_distance_judge_loss_p": round(float(rho.pvalue), 4),
              "spearman_distance_z_judge_loss": round(float(rho_z.statistic), 3),
              "spearman_habit_judge_loss": round(float(rho_h.statistic), 3),
              "human_zero_share": {m: round(float(np.mean(classic[m] == 0)), 3) for m in METRICS + TOTALS},
              "human_mean": {m: round(float(v), 3) for m, v in zip(METRICS, hmean, strict=True)},
              "human_median_total": {m: round(float(np.median(classic[m])), 3) for m in TOTALS},
              "modern_median_total": {m: round(float(np.median(modern[m])), 3) for m in TOTALS},
              "modern_zero_share": {m: round(float(np.mean(modern[m] == 0)), 3) for m in TOTALS}}  # fmt: skip
    preds = score_predictions(arms, checks)
    save_json("rescore.json", {"reference": {"classic": 318, "modern": 4443}, "metrics": list(METRICS),
                               "labels": LABELS, "arms": arms, "checks": checks, "predictions": preds})  # fmt: skip
    cols = ["group", "habit_v1", "rank_habit", "distance", "distance_lo", "distance_hi", "rank_distance",
            "distance_habits", "distance_z", "habit_v1_classic", "habit_v1_modern", "habit_v2_modern",
            "overcorrected_5", "above_human", "loop", "judge_loss_share"]  # fmt: skip
    with (RESULTS / "rescore_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for g, a in arms.items():
            d = a["distance"]
            w.writerow([g, a["habit_v1"], a["rank_habit"], d["value"], d["lo"], d["hi"], a["rank_distance"],
                        a["distance_habits"]["value"], a["distance_z"]["value"], a["habit_v1_classic"]["value"],
                        a["habit_v1_modern"]["value"], a["habit_v2_modern"]["value"], " ".join(a["overcorrected_5"]),
                        " ".join(a["above_human"]), a["loop"], a.get("judge_loss_share", "")])  # fmt: skip
    for g, a in arms.items():
        d = a["distance"]
        print(f"{g:<20} habit {a['habit_v1']:6.1f} (#{a['rank_habit']:>2})  dist {d['value']:5.1f} "
              f"[{d['lo']:4.1f},{d['hi']:4.1f}] (#{a['rank_distance']:>2}) hab {a['distance_habits']['value']:4.1f} "
              f"z {a['distance_z']['value']:.2f} loop {a['loop']:.2f} "
              f"v1-cl {a['habit_v1_classic']['value']:+5.1f} v1-mod {a['habit_v1_modern']['value']:+5.1f} "
              f"v2-mod {a['habit_v2_modern']['value']:+5.1f}  over5: {','.join(a['overcorrected_5']) or '-'}  "
              f"above: {','.join(a['above_human']) or '-'}")  # fmt: skip
    print(json.dumps(checks, indent=1))
    print(json.dumps(preds, indent=1))


def score_predictions(arms: dict, checks: dict) -> list[dict]:
    ps, sfts = arms["A/plain+sampler"], arms["A/sft+sampler"]
    top = min((g for g in arms if g.startswith("A/")), key=lambda g: arms[g]["distance"]["value"])

    def ci(c: dict) -> str:
        return f"{c['value']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"

    res = {
        "R1": (ps["habit_v1_classic"]["hi"] < 0 and sfts["habit_v1_classic"]["hi"] < 0,
               f"prompt + sampler {ci(ps['habit_v1_classic'])}; SFT + sampler {ci(sfts['habit_v1_classic'])}"),
        "R2": (all(len(a["overcorrected"]) >= 2 and {"em_dash", "triad"} <= set(a["overcorrected"])
                   for a in (ps, sfts)),
               f"prompt + sampler: {', '.join(ps['overcorrected']) or 'none'}; "
               f"SFT + sampler: {', '.join(sfts['overcorrected']) or 'none'}"),
        "R3": (top != "A/sft+sampler",
               f"closest: {top} ({arms[top]['distance']['value']}); SFT + sampler rank {sfts['rank_distance']}"),
        "R4": (checks["spearman_distance_judge_loss"] > 0.3,
               f"Spearman {checks['spearman_distance_judge_loss']:+.2f} "
               f"(p = {checks['spearman_distance_judge_loss_p']})"),
    }  # fmt: skip
    return [{**p, "result": res[p["id"]][1], "verdict": "hit" if res[p["id"]][0] else "miss"} for p in PREDICTIONS]


if __name__ == "__main__":
    main()
