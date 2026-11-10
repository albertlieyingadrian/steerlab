"""Every number in the note: scorer-human agreement, the traps, the camera check, and our models' leaderboard.

    python 6_analyse.py

Reads results/human_pairs.csv, scores_human.csv, judge_human.jsonl, human_all_models.csv, traps.csv, scores_traps.csv,
judge_traps.jsonl, gen_clips.csv, gen_timing.csv, scores_ours.csv and judge_ours.jsonl. Writes agreement_human.csv,
agreement_paired.csv, real_footage.csv, rank_human.csv, traps_result.csv, camera_moves.csv, ours_leaderboard.csv and
summary.json in results/, and prints each table.

Agreement with people is pairwise and scored as in the image-to-3D note: for each human-voted pair a scorer earns 1
when it prefers the side the voters preferred, 0 when not, and half for a tie or a split judge; the target is the
majority of the voters left after removing one, averaged over which voter is removed, which is exactly what "one more
voter" (the ceiling) is scored against. Intervals are 95% bootstrap intervals over pairs (over stills for our models).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, boot_mean, bradley_terry, ceiling, loo

JUDGES = ["gpt-5.4-mini", "gemini-3.6-flash"]
VBENCH = ["subject_consistency", "background_consistency", "motion_smoothness", "temporal_flickering",
          "dynamic_degree", "flow", "aesthetic_quality", "imaging_quality",
          "i2v_subject", "i2v_background"]  # fmt: skip
EMBED = ["clip", "dinov2"]
CAMERA = ["camera_hit", "camera_shift"]
QUESTIONS = ["photorealism", "integrity", "camera"]
CLEAR = 0.75  # a pair is "clear" when at least this share of its voters chose one side
summary: dict = {}


def judge_verdicts(path: Path, flip: set[str] | None = None) -> pd.DataFrame:
    """One row per judge and pair: the score of the pair's first-listed side (1, 0, or 0.5 when the two orders
    disagree) and whether the two orders agreed. For questions in `flip` the judge picks the WORSE video (integrity
    asks which has more glitches), so its pick is turned around."""
    raw = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("verdict"):
            raw[(r["judge"], r["pair"], r["order"])] = (str(r["verdict"].get("answer", "")).strip(), r.get("question"))
    out = []
    for judge, pair in sorted({(j, p) for j, p, _ in raw}):
        if (judge, pair, 0) not in raw or (judge, pair, 1) not in raw:
            continue
        (v0, q), (v1, _) = raw[(judge, pair, 0)], raw[(judge, pair, 1)]
        a0, a1 = v0 == "1", v1 == "2"  # side A is shown first in order 0, second in order 1
        if flip and q in flip:
            a0, a1 = not a0, not a1
        score = 1.0 if a0 and a1 else 0.0 if not a0 and not a1 else 0.5
        out.append({"judge": judge, "pair": pair, "question": q, "a": score, "consistent": a0 == a1,
                    "first_shown": [v0, v1].count("1") / 2})  # fmt: skip
    return pd.DataFrame(out)


def prefer(a, b) -> np.ndarray:
    d = np.asarray(a, float) - np.asarray(b, float)
    return np.where(np.isnan(d), np.nan, np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5)))


def human() -> None:
    pairs = pd.read_csv(RESULTS / "human_pairs.csv", keep_default_na=False)
    # Camera: the cheap scorers were run on the even-numbered 400 of the 800 sampled pairs, the judges on every
    # fourth (200); both cuts were fixed for the budget before any score was read.
    number = pairs.pair.str[1:].astype(int)
    pairs = pairs[(pairs.question != "camera") | (number % 2 == 0)].reset_index(drop=True)
    pairs["judged"] = (pairs.question != "camera") | (pairs.pair.str[1:].astype(int) % 4 == 0)
    pairs["share"] = pairs[["votes_a", "votes_b"]].max(axis=1) / (pairs.votes_a + pairs.votes_b)
    pairs["voters"] = pairs.votes_a + pairs.votes_b
    sc = pd.read_csv(RESULTS / "scores_human.csv").set_index("video")
    scorers: dict[str, np.ndarray] = {}
    for m in EMBED + VBENCH + CAMERA:
        if m in sc:
            scorers[m] = prefer(pairs.video_a.map(sc[m]), pairs.video_b.map(sc[m]))
    jv = judge_verdicts(RESULTS / "judge_human.jsonl", flip={"integrity"})
    for judge in JUDGES:
        scorers[judge] = pairs.pair.map(jv[jv.judge == judge].set_index("pair").a).to_numpy(float)
    scorers["both judges"] = pairs.pair.map(jv.groupby("pair").a.mean()).to_numpy(float)
    summary["judge_swap_consistency"] = jv.groupby(["question", "judge"]).consistent.mean().unstack().to_dict()
    summary["judge_first_shown_share"] = jv.groupby(["question", "judge"]).first_shown.mean().unstack().to_dict()
    summary["pairs"] = pairs.groupby("question").size().to_dict()
    summary["clear_pairs"] = pairs[pairs.share >= CLEAR].groupby("question").size().to_dict()
    summary["voters_per_pair"] = pairs.groupby("question").voters.describe()[["min", "50%", "mean", "max"]].to_dict()

    rows = []
    for q in QUESTIONS:
        for subset in ("all", "clear", "judged") if q == "camera" else ("all", "clear"):
            mask = (pairs.question == q) & {"all": True, "clear": pairs.share >= CLEAR, "judged": pairs.judged}[subset]
            idx = np.flatnonzero(mask)
            va, vb = pairs.votes_a.to_numpy()[idx], pairs.votes_b.to_numpy()[idx]
            for name, s in [("one more voter", None), *scorers.items()]:
                if s is None:
                    per = np.array([ceiling(a, b) for a, b in zip(va, vb, strict=True)])
                else:
                    v = s[idx]
                    ok = ~np.isnan(v)
                    if ok.sum() < 20:
                        continue
                    per = np.array([loo(a, b, x) for a, b, x in zip(va[ok], vb[ok], v[ok], strict=True)])
                mean, lo, hi = boot_mean(per)
                rows.append({"question": q, "subset": subset, "scorer": name, "n": len(per), "agree": mean,
                             "lo": lo, "hi": hi})  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "agreement_human.csv", index=False, float_format="%.4f")
    print(table[table.subset == "all"].to_string(index=False, float_format="%.3f"))

    # Paired differences on the same pairs, against the simplest baseline (CLIP) and against one more voter.
    paired = []
    for q in QUESTIONS:
        idx = np.flatnonzero(pairs.question == q)
        va, vb = pairs.votes_a.to_numpy()[idx], pairs.votes_b.to_numpy()[idx]
        voter = np.array([ceiling(a, b) for a, b in zip(va, vb, strict=True)])

        def per_pair(name: str, idx=idx, va=va, vb=vb) -> np.ndarray:
            return np.array([loo(a, b, x) for a, b, x in zip(va, vb, scorers[name][idx], strict=True)])

        clip = per_pair("clip")
        for name in scorers:
            p = per_pair(name)
            if np.isnan(p).mean() > 0.5:
                continue
            for base, ref in (("clip", clip), ("one more voter", voter)):
                if name == base:
                    continue
                d = p - ref
                d = d[~np.isnan(d)]
                mean, lo, hi = boot_mean(d)
                paired.append({"question": q, "scorer": name, "minus": base, "n": len(d), "diff": mean, "lo": lo,
                               "hi": hi})  # fmt: skip
    paired = pd.DataFrame(paired)
    paired.to_csv(RESULTS / "agreement_paired.csv", index=False, float_format="%.4f")
    print(paired.to_string(index=False, float_format="%.3f"))

    # Known answer inside the human data: real camera footage against a generated clip.
    is_gt = (pairs.model1 == "ground_truth_baseline") | (pairs.model2 == "ground_truth_baseline")
    gt = pairs[(pairs.question != "camera") & is_gt]
    gt_a = (gt.model1 == "ground_truth_baseline").to_numpy()
    real = []
    for q in ("photorealism", "integrity"):
        m = (gt.question == q).to_numpy()
        va, vb = gt.votes_a.to_numpy()[m], gt.votes_b.to_numpy()[m]
        hum = np.where(gt_a[m], va > vb, vb > va).astype(float)
        mean, lo, hi = boot_mean(hum)
        real.append({"question": q, "scorer": "voter majority", "n": int(m.sum()), "picks_real": mean, "lo": lo,
                     "hi": hi})  # fmt: skip
        for name, s in scorers.items():
            v = s[gt.index.to_numpy()][m]
            v = np.where(gt_a[m], v, 1 - v)
            v = v[~np.isnan(v)]
            if len(v):
                mean, lo, hi = boot_mean(v)
                real.append({"question": q, "scorer": name, "n": len(v), "picks_real": mean, "lo": lo, "hi": hi})
    real = pd.DataFrame(real)
    real.to_csv(RESULTS / "real_footage.csv", index=False, float_format="%.4f")
    print(real.to_string(index=False, float_format="%.3f"))

    # Model level: does each scorer, over the sample, rank the models as all of benchmark.ai's votes do?
    full = pd.read_csv(RESULTS / "human_all_models.csv")
    ranks = []
    for q in QUESTIONS:
        sub = pairs[pairs.question == q]
        truth = full[full.question == q].set_index("model").bt
        cands = {}
        battles = [(r.model1, r.model2, r.votes_a, r.votes_b) for r in sub.itertuples()]
        cands["voters, sampled pairs only"], _ = bradley_terry(battles, boot=0)
        for name, s in scorers.items():
            v = s[sub.index.to_numpy()]
            battles = [(r.model1, r.model2, x, 1 - x) for r, x in zip(sub.itertuples(), v, strict=True)
                       if not np.isnan(x)]  # fmt: skip
            if len(battles) > 50:
                cands[name], _ = bradley_terry(battles, boot=0)
        for name, scores in cands.items():
            models = sorted(set(scores) & set(truth.index))
            x, y = [scores[m] for m in models], [truth[m] for m in models]
            ranks.append({"question": q, "scorer": name, "models": len(models),
                          "spearman": spearmanr(x, y).statistic, "kendall": kendalltau(x, y).statistic})  # fmt: skip
    ranks = pd.DataFrame(ranks)
    ranks.to_csv(RESULTS / "rank_human.csv", index=False, float_format="%.3f")
    print(ranks.to_string(index=False, float_format="%.3f"))

    # Camera: how often the heuristic reads the asked move, per move.
    cam = pairs[pairs.question == "camera"]
    cols = ["video", "move", "movement"]
    long = pd.concat([cam[["video_a", "move", "movement"]].set_axis(cols, axis=1),
                      cam[["video_b", "move", "movement"]].set_axis(cols, axis=1)])  # fmt: skip
    long = long.drop_duplicates("video")
    long["hit"] = long.video.map(sc["camera_hit"])
    moves = long.groupby("move").agg(clips=("video", "size"), hit=("hit", "mean"), movement=("movement", "first"))
    moves.to_csv(RESULTS / "camera_moves.csv", float_format="%.3f")
    print(moves.to_string(float_format="%.3f"))
    summary["camera_hit_overall"] = boot_mean(long.hit.dropna())


def traps() -> None:
    t = pd.read_csv(RESULTS / "traps.csv")
    sh = pd.read_csv(RESULTS / "scores_human.csv").set_index("video")
    st = pd.read_csv(RESULTS / "scores_traps.csv").set_index("video")
    path = RESULTS / "judge_traps.jsonl"
    jv = judge_verdicts(path) if path.exists() else pd.DataFrame(columns=["judge", "pair", "a", "consistent"])
    rows = []
    for trap in t.trap.unique():
        sub = t[t.trap == trap]
        for m in EMBED + VBENCH:
            if m not in st:
                continue
            v = prefer(sub.trap_video.map(st[m]), sub.video.map(sh[m]))
            v = v[~np.isnan(v)]
            mean, lo, hi = boot_mean(v)
            rows.append({"trap": trap, "scorer": m, "n": len(v), "prefers_trap": mean, "lo": lo, "hi": hi})
        keys = sub.video + "|" + trap
        for judge in JUDGES:
            j = jv[jv.judge == judge].set_index("pair").a
            v = 1 - keys.map(j).dropna().to_numpy(float)  # side A is the real clip
            if len(v):
                mean, lo, hi = boot_mean(v)
                rows.append({"trap": trap, "scorer": judge, "n": len(v), "prefers_trap": mean, "lo": lo, "hi": hi})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "traps_result.csv", index=False, float_format="%.4f")
    print(table.pivot(index="scorer", columns="trap", values="prefers_trap").to_string(float_format="%.2f"))
    if len(jv):
        summary["trap_judge_consistency"] = jv.groupby("judge").consistent.mean().to_dict()


def ours() -> None:
    clips = pd.read_csv(RESULTS / "gen_clips.csv", keep_default_na=False)
    clips = clips[clips.ok.astype(str) == "True"]
    timing = pd.read_csv(RESULTS / "gen_timing.csv")
    sc = pd.read_csv(RESULTS / "scores_ours.csv").set_index("video")
    path = RESULTS / "judge_ours.jsonl"
    jv = judge_verdicts(path) if path.exists() else pd.DataFrame()
    rows = []
    for model, sub in clips.groupby("model"):
        row = {"model": model, "clips": len(sub)}
        for m in EMBED + VBENCH:
            if m in sc:
                mean, lo, hi = boot_mean(sub.video.map(sc[m]).to_numpy(float))
                row.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
        cam = sub[sub.movement != ""]
        if len(cam) and "camera_hit" in sc:
            mean, lo, hi = boot_mean(cam.video.map(sc["camera_hit"]).to_numpy(float))
            row.update({"camera_hit": mean, "camera_hit_lo": lo, "camera_hit_hi": hi, "camera_clips": len(cam)})
        t = timing[(timing.model == model) & timing.seconds.notna()]
        row.update({"seconds_median": t.seconds.median(), "peak_reserved_gb": t.peak_reserved_gb.max(),
                    "frames": int(t.frames.median()), "fps": int(t.fps.median()),
                    "resolution": f"{int(t.width.median())}x{int(t.height.median())}",
                    "failed": int(timing[timing.model == model].seconds.isna().sum())})  # fmt: skip
        rows.append(row)
    board = pd.DataFrame(rows).set_index("model")
    if len(jv):
        jv[["id", "a_model", "b_model"]] = jv.pair.str.split("|", expand=True)
        for judge in [*JUDGES, "both"]:
            sub = jv if judge == "both" else jv[jv.judge == judge]
            elo, ci = bradley_terry([(r.a_model, r.b_model, r.a, 1 - r.a) for r in sub.itertuples()], boot=1000)
            key = "elo" if judge == "both" else f"elo_{judge}"
            board[key] = pd.Series(elo)
            board[f"{key}_lo"] = pd.Series({m: c[0] for m, c in ci.items()})
            board[f"{key}_hi"] = pd.Series({m: c[1] for m, c in ci.items()})
        summary["ours_judge_consistency"] = jv.groupby("judge").consistent.mean().to_dict()
    board.to_csv(RESULTS / "ours_leaderboard.csv", float_format="%.4f")
    print(board.T.to_string(float_format="%.3f"))
    # Do the scorers rank our four models the same way? Kendall's tau between each pair of rankings.
    cols = [c for c in ["elo", "clip", "dinov2", *VBENCH, "camera_hit"] if c in board]
    summary["ours_rank_agreement"] = {f"{a}~{b}": float(kendalltau(board[a], board[b]).statistic)
                                      for k, a in enumerate(cols) for b in cols[k + 1 :]}  # fmt: skip


def main() -> None:
    human()
    if (RESULTS / "scores_traps.csv").exists():
        traps()
    if (RESULTS / "scores_ours.csv").exists():
        ours()
    tokens = {}
    for src in ("human", "traps", "ours"):
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
