"""The shared scorer: every reply of both tracks, one set of rulers, one table.

    python table.py      # -> results/summary.json, results/table.csv, results/per_reply.csv, results/predictions.json

Per arm, on the 48 held-out prompts (Track A: 2 samples per prompt, averaged within a prompt first; Track B: 1):
habit score (deslop toolkit, per 1,000 words) and in-band count; EQ-Bench's slop score (slop-score's own code, run
through Episode 1's `eqbench_slop.mjs`; the leaderboard-style corpus score and the per-reply mean); task fit; invented
claims; MATTR; words; loop and empty shares; the three detectors' flag rates at a threshold that flags 1% of the 318
human passages (and their mean scores); KL drift and quiz (drift.py for Track A, Episode 1's for Track B); and the
judge's wins, ties and losses against the track's untouched model. Intervals: 95% bootstrap over prompts (10,000
draws, seed 0); paired differences resample prompts with each prompt's replies kept together.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from deslop.checks import invented_claims, mattr, task_fit
from deslop.score import score

from common import (
    EP1,
    GEN,
    RESULTS,
    SEED,
    TRACK_B,
    B,
    boot,
    cell,
    held_out,
    human_passages,
    load_json,
    looped,
    read_jsonl,  # fmt: skip
    save_json,
)

SLOP_SCORE = Path(os.environ.get("SLOP_SCORE_DIR", Path.home() / ".cache" / "steerlab" / "slop-score"))
DETECTORS = ("nine_b", "voi380", "desklib")  # all: higher = more AI
STEERING = ("register", "probe", "sae-dir", "sae-abl")
TRACK_A_ORDER = ["original", "plain", "sampler", "plain+sampler", "register", "probe", "sae-dir", "sae-abl",
                 "plain+register", "plain+sae", "sft", "dpo", "sft+sampler"]  # fmt: skip


def track_a() -> list[str]:
    have = {p.stem for p in GEN.glob("*.jsonl")}
    rand = sorted(a for a in have if a.startswith("random-"))
    return [a for a in TRACK_A_ORDER if a in have] + rand


def eqbench(rows: list[dict]) -> dict:
    tmp_in, tmp_out = RESULTS / "eqbench_in.jsonl", RESULTS / "eqbench.json"
    with tmp_in.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    subprocess.run(["node", str(EP1 / "eqbench_slop.mjs"), str(SLOP_SCORE), str(tmp_in), str(tmp_out)], check=True)
    tmp_in.unlink()
    return json.loads(tmp_out.read_text())


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    rng = np.random.default_rng(SEED)
    draws = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return cell((float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))))


def kl_cell(rows: list[dict] | None) -> dict | None:
    return cell(boot(np.array([x["kl"] for x in rows])), 4) if rows else None


def main() -> None:
    P = {p["id"]: p for p in held_out()}
    pids = sorted(P)
    rows: dict[str, list[dict]] = {}
    for arm in track_a():
        rows[f"A/{arm}"] = [{**r, "key": f"A/{arm}/{r['id']}/{r['sample']}"} for r in read_jsonl(GEN / f"{arm}.jsonl")]
    for run in TRACK_B:
        rows[f"B/{run}"] = [{**r, "sample": 0, "key": f"B/{run}/{r['id']}"}
                            for r in read_jsonl(EP1 / "results" / "gen" / f"{run}.jsonl")]  # fmt: skip
    human = human_passages()
    eq_in = [{"key": r["key"], "group": g, "text": r["text"]} for g, rs in rows.items() for r in rs]
    eq_in += [{"key": f"human/{h['source']}/{h['id']}", "group": f"human-{h['source']}", "text": h["text"]}
              for h in human]  # fmt: skip
    eq = eqbench(eq_in)
    det = json.loads((RESULTS / "rulers" / "scores.json").read_text())
    hum_keys = [f"human/{h['source']}/{h['id']}" for h in human]
    thresholds = {d: float(np.quantile([det[k][d] for k in hum_keys], 0.99)) for d in DETECTORS}
    human_flag = {d: float(np.mean([det[k][d] >= thresholds[d] for k in hum_keys])) for d in DETECTORS}
    drift = json.loads((RESULTS / "drift.json").read_text()) if (RESULTS / "drift.json").exists() else {}
    ep1_kl, ep1_quiz = (json.loads((EP1 / "results" / f).read_text()) for f in ("kl.json", "quiz.json"))
    verdicts = load_json("judge/verdicts.json") if (RESULTS / "judge" / "verdicts.json").exists() else []

    per_reply, summary, prompt_level = [], {}, {}
    for g, rs in rows.items():
        track, arm = g.split("/", 1)
        recs = []
        for r in rs:
            t, p = r["text"], P[r["id"]]
            s = score(t)
            d = det.get(r["key"], {})
            e = eq["rows"].get(r["key"]) or {"slop_score": 0.0}
            rec = {"group": g, "id": r["id"], "sample": r["sample"], "habit": round(s.score, 3),
                   "in_band": int(bool(s.in_human_band)), "eqbench": round(e["slop_score"], 3),
                   "task_fit": int(all(task_fit(t, p["genre"], p["words"]).values())),
                   "invented": int(invented_claims(t) > 0), "mattr": round(mattr(t), 4), "words": s.words,
                   "loop": int(looped(t)), "empty": int(not t.strip()),
                   "finished": int(bool(r.get("finished", True)))}  # fmt: skip
            for k in DETECTORS:
                rec[k] = d.get(k)
                rec[f"{k}_flag"] = None if d.get(k) is None else int(d[k] >= thresholds[k])
            recs.append(rec)
        per_reply += recs

        def by_prompt(field: str, recs=recs) -> np.ndarray:
            return np.array([np.mean([x[field] for x in recs if x["id"] == i]) for i in pids])

        out = {"track": track, "arm": arm, "replies": len(recs)}
        for f in ("habit", "in_band", "eqbench", "task_fit", "invented", "mattr", "words", "loop", "empty"):
            out[f] = cell(boot(by_prompt(f)), 3)
        out["in_band_count"] = int(sum(x["in_band"] for x in recs))
        out["eqbench_corpus"] = round(eq["groups"][g]["slop_score"], 2)
        have_det = [k for k in DETECTORS if all(x[k] is not None for x in recs)]
        for k in have_det:
            out[f"{k}_flag"] = cell(boot(by_prompt(f"{k}_flag")), 3)
            out[f"{k}_mean"] = round(float(np.mean([x[k] for x in recs])), 4)
        if track == "A":
            # the sampler changes no next-token prediction: a sampler arm's KL is that of the same arm without it
            src = arm.replace("+sampler", "") if arm.endswith("+sampler") else ("original" if arm == "sampler" else arm)
            if arm == "original" or src == "original":
                out["kl"] = {"value": 0.0, "lo": 0.0, "hi": 0.0}
            elif src in drift:
                out["kl"] = kl_cell(drift[src]["kl"])
            if arm in drift:
                out["quiz"] = int(sum(x["correct"] for x in drift[arm]["quiz"]))
        else:
            if arm in ep1_kl:
                out["kl"] = kl_cell(ep1_kl[arm])
            elif arm == "base" or arm.endswith("sampler"):
                out["kl"] = {"value": 0.0, "lo": 0.0, "hi": 0.0}
            if arm in ep1_quiz:
                out["quiz"] = int(sum(x["correct"] for x in ep1_quiz[arm]))
        vs = [v["result"] for v in verdicts if v["track"] == track and v["arm"] == arm]
        if vs:
            pts = np.array([1.0 if r == "win" else 0.5 if r == "tie" else 0.0 for r in vs])
            out["judge"] = {"wins": vs.count("win"), "ties": vs.count("tie"), "losses": vs.count("loss"),
                            "points": cell(boot(pts), 3)}  # fmt: skip
        prompt_level[g] = {f: by_prompt(f) for f in ["habit", "eqbench", "loop", "words"] +
                           [f"{k}_flag" for k in have_det]}  # fmt: skip
        summary[g] = out

    for track, base in (("A", "A/original"), ("B", "B/base")):
        for g in list(summary):
            if g.startswith(track + "/") and g != base and base in prompt_level:
                summary[g]["habit_vs_original"] = paired(prompt_level[g]["habit"], prompt_level[base]["habit"])
                if "nine_b_flag" in prompt_level[g] and "nine_b_flag" in prompt_level[base]:
                    summary[g]["nine_b_vs_original"] = paired(prompt_level[g]["nine_b_flag"],
                                                              prompt_level[base]["nine_b_flag"])  # fmt: skip
    randoms = {}
    for kind in ("register", "probe", "sae-dir"):
        rs = [g for g in summary if g.startswith(f"A/random-{kind}-")]
        if rs:
            vals = [summary[g]["habit"]["value"] for g in rs]
            randoms[kind] = {"n": len(rs), "habit_mean": round(float(np.mean(vals)), 3),
                             "habit_range": [min(vals), max(vals)]}  # fmt: skip

    preds = predictions(summary, prompt_level)
    checks = extra_checks(summary, per_reply, prompt_level, randoms)
    save_json("summary.json", {"thresholds_1pct": thresholds, "human_flag_rate": human_flag, "arms": summary,
                               "random_bands": randoms, "checks": checks, "predictions": preds})  # fmt: skip
    print(json.dumps(checks, indent=1))
    save_json("predictions.json", preds)
    cols = [
        "group",
        "habit",
        "habit_lo",
        "habit_hi",
        "in_band_count",
        "eqbench_corpus",
        "task_fit",
        "invented",
        "loop",
        "words",
        "nine_b_flag",
        "voi380_flag",
        "desklib_flag",
        "kl",
        "quiz",
        "judge_w",
        "judge_t",
        "judge_l",
    ]
    with (RESULTS / "table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for g, o in summary.items():
            j = o.get("judge", {})
            w.writerow([g, o["habit"]["value"], o["habit"]["lo"], o["habit"]["hi"], o["in_band_count"],
                        o["eqbench_corpus"], o["task_fit"]["value"], o["invented"]["value"], o["loop"]["value"],
                        o["words"]["value"], o.get("nine_b_flag", {}).get("value"),
                        o.get("voi380_flag", {}).get("value"), o.get("desklib_flag", {}).get("value"),
                        (o.get("kl") or {}).get("value"), o.get("quiz"),
                        j.get("wins"), j.get("ties"), j.get("losses")])  # fmt: skip
    with (RESULTS / "per_reply.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_reply[0]))
        w.writeheader()
        w.writerows(per_reply)
    print(f"thresholds {thresholds}; human flag {human_flag}")
    nan = float("nan")
    for g, o in summary.items():
        j = o.get("judge", {})
        print(f"{g:24} habit {o['habit']['value']:6.1f} band {o['in_band_count']:3d} eq {o['eqbench_corpus']:5.1f} "
              f"9B {o.get('nine_b_flag', {}).get('value', nan):.2f} "
              f"voi {o.get('voi380_flag', {}).get('value', nan):.2f} "
              f"desk {o.get('desklib_flag', {}).get('value', nan):.2f} loop {o['loop']['value']:.2f} "
              f"kl {(o.get('kl') or {}).get('value', nan):.3f} quiz {o.get('quiz')} "
              f"judge {j.get('wins')}/{j.get('ties')}/{j.get('losses')}")  # fmt: skip
    print(json.dumps(randoms))
    print(json.dumps(preds, indent=1))


def extra_checks(S: dict, per_reply: list[dict], PL: dict, randoms: dict) -> dict:
    """The plan's re-checks: outside metric agreement, the seed noise floor, steering vs its random band, medians."""
    from scipy.stats import spearmanr

    named = [g for g in S if "/random-" not in g]
    rho = spearmanr([S[g]["habit"]["value"] for g in named], [S[g]["eqbench_corpus"] for g in named])
    orig = [r for r in per_reply if r["group"] == "A/original"]
    s0 = np.array(
        [
            np.mean([r["habit"] for r in orig if r["id"] == i and r["sample"] == 0])
            for i in sorted({r["id"] for r in orig})
        ]
    )
    s1 = np.array(
        [
            np.mean([r["habit"] for r in orig if r["id"] == i and r["sample"] == 1])
            for i in sorted({r["id"] for r in orig})
        ]
    )
    vs_random = {}
    for kind in ("register", "probe", "sae-dir"):
        rs = [g for g in S if g.startswith(f"A/random-{kind}-")]
        if rs and f"A/{kind}" in PL:
            pooled = np.mean([PL[g]["habit"] for g in rs], axis=0)
            below = S[f"A/{kind}"]["habit"]["value"] < randoms[kind]["habit_range"][0]
            vs_random[kind] = {"diff_vs_random_mean": paired(PL[f"A/{kind}"]["habit"], pooled),
                               "below_every_random": below}  # fmt: skip
    medians = {g: round(float(np.median([r["habit"] for r in per_reply if r["group"] == g])), 2) for g in named}
    return {"spearman_habit_eqbench": {"rho": round(float(rho.statistic), 3), "p": float(rho.pvalue),
                                       "n_arms": len(named)},
            "original_seed1_minus_seed0": paired(s1, s0), "steering_vs_random": vs_random,
            "habit_medians": medians}  # fmt: skip


def predictions(S: dict, PL: dict) -> dict:
    need = ["A/original", "A/plain", "A/sampler", "A/plain+sampler", "A/sft", "A/dpo"] + [f"A/{a}" for a in STEERING]
    if any(g not in S for g in need):
        return {"status": "incomplete", "missing": [g for g in need if g not in S]}

    def h(a):
        return S[f"A/{a}"]["habit"]["value"]

    out = {}
    nowc = min(("plain", "sampler", "plain+sampler"), key=h)
    best_steer = min(STEERING, key=h)
    d1 = paired(PL["A/plain+sampler"]["habit"], PL[f"A/{best_steer}"]["habit"])
    out["P1"] = {"best_no_weight_change": nowc, "best_steering": best_steer, "diff": d1,
                 "hit": nowc == "plain+sampler" and d1["hi"] < 0}  # fmt: skip
    best_sae = min(("sae-dir", "sae-abl"), key=h)
    d2 = paired(PL[f"A/{best_sae}"]["habit"], PL["A/probe"]["habit"])
    out["P2"] = {"best_sae": best_sae, "diff": d2, "hit": d2["hi"] < 0}
    named = [g for g in S if "nine_b_flag" in S[g] and "/random-" not in g]
    lowest = min(named, key=lambda g: S[g]["nine_b_flag"]["value"])
    out["P3"] = {"lowest_arm": lowest, "lowest_rate": S[lowest]["nine_b_flag"]["value"],
                 "hit": S[lowest]["nine_b_flag"]["value"] >= 0.5}  # fmt: skip
    kl_sft = (S["A/sft"].get("kl") or {}).get("value")
    kls = {a: (S[f"A/{a}"].get("kl") or {}).get("value") for a in STEERING}
    out["P4"] = {"sft_kl": kl_sft, "steering_kl": kls,
                 "hit": kl_sft is not None and all(v is not None and v > kl_sft for v in kls.values())}  # fmt: skip
    judged = [g for g in S if g.startswith("A/") and "judge" in S[g] and "/random-" not in g]
    lost = [g for g in judged if S[g]["judge"]["losses"] > S[g]["judge"]["wins"]]
    out["P5"] = {"arms_judged": len(judged), "arms_lost": len(lost), "lost": lost,
                 "hit": bool(judged) and len(lost) >= 2 / 3 * len(judged)}  # fmt: skip
    d6 = paired(PL["A/dpo"]["habit"], PL["A/sft"]["habit"])
    kl_dpo = (S["A/dpo"].get("kl") or {}).get("value")
    out["P6"] = {"diff": d6, "kl_dpo": kl_dpo, "kl_sft": kl_sft,
                 "hit": d6["hi"] < 0 and None not in (kl_dpo, kl_sft) and kl_dpo <= 1.2 * kl_sft}  # fmt: skip
    return out


if __name__ == "__main__":
    main()
