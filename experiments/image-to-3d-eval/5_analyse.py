"""Every number in the note: metric-human agreement, the GSO leaderboard, the known-answer checks, speed and memory.

    python 5_analyse.py

Reads results/human_pairs.csv, human_video_scores.csv, judge_human.jsonl, human_all_models.csv, gso_scores.csv,
gso_timing.csv and judge_gso.jsonl. Writes results/agreement_human.csv, rank_human.csv, gso_leaderboard.csv,
gso_checks.csv, gso_metric_agreement.csv, arena_overlap.csv and summary.json, and prints each table.

Agreement with people is pairwise: for each human-voted pair, a scorer "agrees" when it prefers the side most voters
chose (a tie counts as half). The human ceiling is how often one voter agrees with the majority of the other four.
Intervals are 95% bootstrap intervals over pairs (over objects for the GSO tables).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, boot_mean, bradley_terry

CRITERIA = ["alignment", "plausibility", "coherency", "texture", "geometry", "final"]
JUDGES = ["gpt-5.4-mini", "gemini-3.6-flash"]
SYSTEM_NAMES = {
    "trellis_mesh": "TRELLIS mesh",
    "trellis_splats": "TRELLIS splats",
    "triposr": "TripoSR",
    "sf3d": "Stable Fast 3D",
    "idol": "Idol",
}
DAMAGED = {"gt_bumpy": "true shape, bumpy", "gt_squashed": "true shape, squashed", "gt_blurred": "true shape, blurred"}
ARENA_NAMES = {"trellis_mesh": "TRELLIS", "trellis_splats": "TRELLIS-3DGS", "triposr": "TripoSR", "sf3d": "SF3D"}
BENCH_TO_ARENA = {"TRELLIS.2": "TRELLIS.2-4B", "TRELLIS": "TRELLIS", "Hunyuan3D 2.1": "Hunyuan3D-2.1",
                  "SAM 3D Objects": "SAM-3D-Objects-3DGS", "Pixal3D": "Pixal3D"}  # fmt: skip
METRICS = ["chamfer", "fscore@0.05", "fscore@0.02", "mask_iou", "psnr", "ssim", "lpips", "normal_lpips", "clip"]
summary: dict = {}


def judge_verdicts(path: Path) -> pd.DataFrame:
    """One row per judge and pair: each criterion's score for the pair's first-listed side (1, 0, or 0.5 when the
    two orders disagree), and whether the two orders agreed."""
    raw = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("verdict"):
            raw[(r["judge"], r["pair"], r["order"])] = r["verdict"]
    out = []
    for judge, pair in sorted({(j, p) for j, p, _ in raw}):
        if (judge, pair, 0) not in raw or (judge, pair, 1) not in raw:
            continue
        v0, v1 = raw[(judge, pair, 0)], raw[(judge, pair, 1)]
        row = {"judge": judge, "pair": pair}
        for c in CRITERIA:
            a0 = str(v0.get(c, "")).strip() == "1"  # order 0 shows side A first
            a1 = str(v1.get(c, "")).strip() == "2"  # order 1 shows side A second
            row[c] = 1.0 if a0 and a1 else 0.0 if not a0 and not a1 else 0.5
            row[f"{c}_consistent"] = a0 == a1
        out.append(row)
    return pd.DataFrame(out)


def ceiling(a: int, b: int) -> float:
    """Expected agreement of one voter with the majority of the others (a 2-2 split counts as half)."""

    def agree(mine_a: bool, oa: int, ob: int) -> float:
        return 0.5 if oa == ob else float((oa > ob) == mine_a)

    return (a * agree(True, a - 1, b) + b * agree(False, a, b - 1)) / (a + b)


def loo(a: int, b: int, v: float) -> float:
    """Expected agreement of a verdict v (1 = side A, 0 = side B, in between = split) with the majority of four
    voters, averaged over which of the five is left out: the same target the one-voter ceiling is scored on."""

    def agree(oa: int, ob: int) -> float:
        return 0.5 if oa == ob else (v if oa > ob else 1 - v)

    return (a * agree(a - 1, b) + b * agree(a, b - 1)) / (a + b)


def spearman_ci(x, y, boot: int = 1000) -> tuple[float, float, float]:
    """Spearman's rho and a 95% interval from resampling the models."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(boot):
        pick = rng.integers(0, len(x), len(x))
        if len(set(pick)) > 2:
            draws.append(spearmanr(x[pick], y[pick]).statistic)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(spearmanr(x, y).statistic), float(lo), float(hi)


def human_agreement() -> None:
    pairs = pd.read_csv(RESULTS / "human_pairs.csv")
    pairs["human_a"] = (pairs.votes_a > pairs.votes_b).astype(float)
    pairs["margin"] = pairs[["votes_a", "votes_b"]].max(axis=1)
    pairs["ceiling"] = [ceiling(a, b) for a, b in zip(pairs.votes_a, pairs.votes_b, strict=True)]
    vs = pd.read_csv(RESULTS / "human_video_scores.csv").set_index("video")
    scorers = {}
    for m in ("clip", "dino"):
        diff = pairs.video_a.map(vs[m]) - pairs.video_b.map(vs[m])
        scorers[m] = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    jv = judge_verdicts(RESULTS / "judge_human.jsonl")
    for judge in JUDGES:
        sub = jv[jv.judge == judge].set_index("pair")
        for c in CRITERIA:
            scorers[f"{judge}:{c}"] = pairs.pair.map(sub[c]).to_numpy(float)
    both = pd.concat([jv[jv.judge == j].set_index("pair")["final"] for j in JUDGES], axis=1).mean(axis=1)
    scorers["both judges:final"] = pairs.pair.map(both).to_numpy(float)
    rows = []
    for q in ("geometry", "appearance"):
        for subset, mask in (("all", pairs.question == q), ("clear", (pairs.question == q) & (pairs.margin >= 4))):
            idx = np.flatnonzero(mask)
            human = pairs.human_a.to_numpy()[idx]
            for name, s in [("one human (ceiling)", None), *scorers.items()]:
                if s is None:
                    agree = agree4 = pairs.ceiling.to_numpy()[idx]
                else:
                    v = s[idx]
                    ok = ~np.isnan(v)
                    # A split verdict (0.5, or 0.25/0.75 when averaging two judges) earns partial credit.
                    agree = np.where(human[ok] == 1, v[ok], 1 - v[ok])
                    va, vb = pairs.votes_a.to_numpy()[idx][ok], pairs.votes_b.to_numpy()[idx][ok]
                    agree4 = np.array([loo(a, b, x) for a, b, x in zip(va, vb, v[ok], strict=True)])
                mean, lo, hi = boot_mean(agree)
                mean4, lo4, hi4 = boot_mean(agree4)
                rows.append({"question": q, "subset": subset, "scorer": name, "n": len(agree), "agree": mean,
                             "lo": lo, "hi": hi, "agree4": mean4, "lo4": lo4, "hi4": hi4})  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "agreement_human.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format="%.3f"))

    # Paired differences on the same pairs: does a scorer beat CLIP (or the one-human ceiling) item by item?
    paired = []
    for q in ("geometry", "appearance"):
        idx = np.flatnonzero(pairs.question == q)

        def per_pair(name: str, idx=idx) -> np.ndarray:
            """Per-pair agreement with the leave-one-out four-voter majority, the ceiling's own target."""
            if name == "one human (ceiling)":
                return pairs.ceiling.to_numpy()[idx]
            va, vb = pairs.votes_a.to_numpy()[idx], pairs.votes_b.to_numpy()[idx]
            return np.array([loo(a, b, x) for a, b, x in zip(va, vb, scorers[name][idx], strict=True)])

        gem, gpt, voter = "gemini-3.6-flash:final", "gpt-5.4-mini:final", "one human (ceiling)"
        for a, b in [(gem, "clip"), (gpt, "clip"), ("dino", "clip"), (gem, gpt), (gem, voter), ("clip", voter)]:
            mean, lo, hi = boot_mean(per_pair(a) - per_pair(b))
            paired.append({"question": q, "scorer": a, "minus": b, "diff": mean, "lo": lo, "hi": hi})
    paired = pd.DataFrame(paired)
    paired.to_csv(RESULTS / "agreement_paired.csv", index=False, float_format="%.4f")
    print(paired.to_string(index=False, float_format="%.3f"))

    # Halo: how often a judge gives the same side on all six questions in one reply.
    rows = [json.loads(line) for line in (RESULTS / "judge_human.jsonl").read_text().splitlines()]
    halo = {}
    for judge in JUDGES:
        same = [len({str(r["verdict"].get(c)) for c in CRITERIA}) == 1 for r in rows if r["judge"] == judge]
        halo[judge] = float(np.mean(same))
    summary["human_halo"] = halo
    first = {
        j: float(np.mean([str(r["verdict"].get("final")) == "1" for r in rows if r["judge"] == j])) for j in JUDGES
    }
    summary["human_first_shown_share"] = first
    print("all six answers the same side:", halo, "; final answer is the first object shown:", first)
    summary["human_swap_consistency"] = jv.groupby("judge")["final_consistent"].mean().to_dict()
    summary["human_pairs"] = {q: int((pairs.question == q).sum()) for q in ("geometry", "appearance")}
    summary["human_clear_pairs"] = {
        q: int(((pairs.question == q) & (pairs.margin >= 4)).sum()) for q in ("geometry", "appearance")
    }
    summary["human_models"] = sorted(set(pairs.model1) | set(pairs.model2))
    print("judge swap consistency (final):", summary["human_swap_consistency"])

    # Model level: does each scorer, over the sample, rank the 13 models as all of benchmark.ai's votes do?
    full = pd.read_csv(RESULTS / "human_all_models.csv")
    ranks = []
    for q in ("geometry", "appearance"):
        sub = pairs[pairs.question == q]
        truth = full[full.question == q].set_index("model").bt
        candidates = {}
        battles = [(r.model1, r.model2, r.votes_a, r.votes_b) for r in sub.itertuples()]
        candidates["humans, these 400 pairs only"], _ = bradley_terry(battles, boot=0)
        long = pd.concat([sub[["model1", "video_a"]].set_axis(["model", "video"], axis=1),
                          sub[["model2", "video_b"]].set_axis(["model", "video"], axis=1)])  # fmt: skip
        for m in ("clip", "dino"):
            candidates[m] = long.assign(s=long.video.map(vs[m])).groupby("model").s.mean().to_dict()
        for name in [f"{j}:{c}" for j in JUDGES for c in ("final", "alignment", "geometry")] + ["both judges:final"]:
            v = pd.Series(scorers[name], index=pairs.index)[sub.index]
            rows_v = zip(sub.itertuples(), v, strict=True)
            battles = [(r.model1, r.model2, x, 1 - x) for r, x in rows_v if not np.isnan(x)]
            candidates[name], _ = bradley_terry(battles, boot=0)
        for name, scores in candidates.items():
            models = sorted(set(scores) & set(truth.index))
            x, y = [scores[m] for m in models], [truth[m] for m in models]
            rho, lo, hi = spearman_ci(x, y)
            ranks.append({"question": q, "scorer": name, "models": len(models), "spearman": rho, "lo": lo, "hi": hi,
                          "kendall": kendalltau(x, y).statistic})  # fmt: skip
    ranks = pd.DataFrame(ranks)
    ranks.to_csv(RESULTS / "rank_human.csv", index=False, float_format="%.3f")
    print(ranks.to_string(index=False, float_format="%.3f"))


def gso() -> None:
    scores = pd.read_csv(RESULTS / "gso_scores.csv")
    timing = pd.read_csv(RESULTS / "gso_timing.csv")
    objects = sorted(scores.object.unique())
    names = {**SYSTEM_NAMES, **DAMAGED}
    ok = scores[scores.system.isin(names) & scores.chamfer.notna()]
    path = RESULTS / "judge_gso.jsonl"
    jv = judge_verdicts(path) if path.exists() else pd.DataFrame()
    if len(jv):
        jv[["object", "a", "b"]] = jv.pair.str.split("|", expand=True)
    rows = []
    for system, name in names.items():
        sub = ok[ok.system == system]
        if sub.empty:
            continue
        row = {"system": system, "name": name, "objects": len(sub), "failed": len(objects) - len(sub)}
        for m in METRICS:
            mean, lo, hi = boot_mean(sub[m])
            row.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
        runs = timing[timing.model == system.split("_")[0]] if system in SYSTEM_NAMES else timing.iloc[:0]
        t = runs[runs.seconds.notna()]
        if len(t):
            row.update({"seconds_median": t.seconds.median(), "seconds_p90": t.seconds.quantile(0.9),
                        "peak_reserved_gb": t.peak_reserved_gb.max(), "peak_allocated_gb": t.peak_allocated_gb.max(),
                        "gen_failed": int(runs.seconds.isna().sum())})  # fmt: skip
        rows.append(row)
    board = pd.DataFrame(rows)
    gt_clip = scores[scores.system == "gt"]["clip"]
    summary["gso_gt_clip"] = boot_mean(gt_clip)
    model_rows = jv[jv.a.isin(SYSTEM_NAMES) & jv.b.isin(SYSTEM_NAMES)] if len(jv) else jv
    if len(model_rows):
        for judge in [*JUDGES, "both"]:
            sub = model_rows if judge == "both" else model_rows[model_rows.judge == judge]
            battles = [(r.a, r.b, r.final, 1 - r.final) for r in sub.itertuples()]
            if not battles:
                continue
            elo, ci = bradley_terry(battles, boot=1000)
            key = "elo" if judge == "both" else f"elo_{judge}"
            board[key] = board.system.map(elo)
            board[f"{key}_lo"] = board.system.map(lambda s, ci=ci: ci.get(s, (np.nan, np.nan))[0])
            board[f"{key}_hi"] = board.system.map(lambda s, ci=ci: ci.get(s, (np.nan, np.nan))[1])
        summary["gso_swap_consistency"] = model_rows.groupby("judge")["final_consistent"].mean().to_dict()
    board.to_csv(RESULTS / "gso_leaderboard.csv", index=False, float_format="%.4f")
    print(board.to_string(index=False, float_format="%.3f"))
    summary["gso_objects"] = len(objects)

    # Known answers for the reference-free scorers: each should prefer the true object to a damaged copy of it.
    checks = []
    gt_by_obj = scores[scores.system == "gt"].set_index("object")["clip"]
    for kind in DAMAGED:
        sub = ok[ok.system == kind]
        wins = (gt_by_obj[sub.object].to_numpy() > sub["clip"].to_numpy()).astype(float)
        mean, lo, hi = boot_mean(wins)
        checks.append({"damage": kind, "scorer": "clip", "n": len(wins), "prefers_true": mean, "lo": lo, "hi": hi})
        if len(jv):
            for judge in JUDGES:
                sub_j = jv[(jv.judge == judge) & (jv.b == kind)]
                for c in CRITERIA:
                    mean, lo, hi = boot_mean(sub_j[c])
                    checks.append({"damage": kind, "scorer": f"{judge}:{c}", "n": len(sub_j), "prefers_true": mean,
                                   "lo": lo, "hi": hi})  # fmt: skip
                summary.setdefault("gso_check_consistency", {})[f"{judge}:{kind}"] = sub_j.final_consistent.mean()
    checks = pd.DataFrame(checks)
    checks.to_csv(RESULTS / "gso_checks.csv", index=False, float_format="%.3f")
    print(checks.to_string(index=False, float_format="%.3f"))

    # On GSO the true object is known: how often does each reference-free scorer pick the model with the lower
    # Chamfer distance (the geometry truth) or the lower LPIPS (the appearance truth)?
    agree = []
    models_ok = ok[ok.system.isin(SYSTEM_NAMES)]
    piv = models_ok.pivot_table(index="object", columns="system", values=["chamfer", "lpips", "clip"])
    systems = [s for s in SYSTEM_NAMES if s in models_ok.system.unique()]
    pairs = [(a, b) for k, a in enumerate(systems) for b in systems[k + 1 :]]
    for truth in ("chamfer", "lpips"):
        records = {"clip": [], **{f"{j}:{c}": [] for j in JUDGES for c in ("final", "geometry", "texture")}}
        groups = []
        for obj in piv.index:
            for a, b in pairs:
                ta, tb = piv.loc[obj, (truth, a)], piv.loc[obj, (truth, b)]
                if np.isnan(ta) or np.isnan(tb):
                    continue
                a_better = ta < tb
                groups.append(obj)
                records["clip"].append(float((piv.loc[obj, ("clip", a)] > piv.loc[obj, ("clip", b)]) == a_better))
                for j in JUDGES:
                    # The judge file lists each pair in its own order; flip the verdict when it is b, a.
                    hit, flip = (jv[(jv.judge == j) & (jv.object == obj) & (jv.a == a) & (jv.b == b)], False)
                    if not len(hit):
                        hit, flip = (jv[(jv.judge == j) & (jv.object == obj) & (jv.a == b) & (jv.b == a)], True)
                    for c in ("final", "geometry", "texture"):
                        v = hit.iloc[0][c] if len(hit) else np.nan
                        v = 1 - v if flip else v
                        records[f"{j}:{c}"].append(v if a_better else 1 - v)
        for name, vals in records.items():
            vals = np.array(vals, float)
            keep = ~np.isnan(vals)
            if keep.sum():
                mean, lo, hi = boot_mean(vals[keep], groups=np.array(groups)[keep])
                agree.append({"truth": truth, "scorer": name, "pairs": int(keep.sum()), "agree": mean, "lo": lo,
                              "hi": hi})  # fmt: skip
    agree = pd.DataFrame(agree)
    agree.to_csv(RESULTS / "gso_metric_agreement.csv", index=False, float_format="%.3f")
    print(agree.to_string(index=False, float_format="%.3f"))

    # Public human ratings for the systems that are also on 3D Arena (a snapshot is saved for reruns).
    snap = RESULTS / "arena_leaderboard.json"
    if not snap.exists():
        url = "https://3d-arena-3d-arena.hf.space/api/leaderboard?vote_type=render"
        snap.write_text(json.dumps(json.load(urllib.request.urlopen(url, timeout=60)), indent=1))
    arena = {r["name"]: r["score"] for r in json.loads(snap.read_text())}
    overlap = board[board.system.isin(ARENA_NAMES)].copy()
    overlap["arena_elo"] = overlap.system.map(lambda s: arena.get(ARENA_NAMES[s]))
    overlap.to_csv(RESULTS / "arena_overlap.csv", index=False, float_format="%.3f")
    corr = {}
    for col in ["elo", "clip", "chamfer", "lpips", "psnr", "fscore@0.05"]:
        if col in overlap and overlap[col].notna().sum() >= 3:
            sign = -1 if col in ("chamfer", "lpips") else 1
            corr[col] = float(kendalltau(sign * overlap[col], overlap.arena_elo).statistic)
    summary["arena_overlap_kendall"] = corr
    summary["arena_overlap_n"] = len(overlap)
    print("3D Arena overlap (Kendall tau, n =", len(overlap), "):", corr)
    # Two human leaderboards against each other, on the models both list.
    bench = pd.read_csv(RESULTS / "human_all_models.csv")
    pair_up = {}
    for q in ("geometry", "appearance"):
        b = bench[bench.question == q].set_index("model").bt
        common = [m for m in BENCH_TO_ARENA if m in b.index and BENCH_TO_ARENA[m] in arena]
        rho = spearmanr([b[m] for m in common], [arena[BENCH_TO_ARENA[m]] for m in common]).statistic
        pair_up[q] = {"models": common, "spearman": float(rho)}
    summary["benchmark_vs_arena"] = pair_up
    print("benchmark.ai vs 3D Arena:", pair_up)


def main() -> None:
    human_agreement()
    if (RESULTS / "gso_scores.csv").exists():
        gso()
    tokens = {}
    for src in ("human", "gso"):
        path = RESULTS / f"judge_{src}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                t = tokens.setdefault(f"{r['judge']}:{src}", {"calls": 0, "in": 0, "out": 0})
                t["calls"] += 1
                t["in"] += r.get("in_tokens") or 0
                t["out"] += r.get("out_tokens") or 0
    summary["judge_tokens"] = tokens
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(tokens))


if __name__ == "__main__":
    main()
