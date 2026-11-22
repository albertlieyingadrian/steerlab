"""
Zero-shot pairwise judges (tiers 2-4) for The Taste Stack Episode 7: job files for the open models (modal_judge.py)
and the frontier judge (frontier.py), and the collector that turns replies into P(a beats b) per item.

An item is (track, part, scene, a, b) with a < b by asset id; every designer of the track is scored against the same
item verdict. Open models see both orders ("ab": a is Image 1; "ba"), and P(a) = mean(P1 | ab, 1 - P1 | ba).

    python judges.py jobs --stage dev        # dev items x 2 prompt wordings x 2 orders + R5 known-answer pairs
    python judges.py jobs --stage confirm    # confirm items, prompt chosen on dev, 2 orders
    python judges.py dev                     # R5 + prompt choice per model (ledger rows)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd

from taste import CACHE, CRITERION, HERE, PARTS, load

JOBS = CACHE / "jobs"
PROMPTS = {
    "plain": (
        (
            "You are a professional graphic designer reviewing two candidate designs made for the same client brief.\n\n"
            "Brief:\n{brief}\n\nCriterion: {criterion}\n"
        ),
        "Which image is better on this criterion? Answer with 1 or 2 only.",
    ),
    "focused": (
        (
            "You are a professional graphic designer choosing which of two candidate designs to send to a client. Judge "
            "only the criterion named below, the way an experienced designer would, and ignore every other quality.\n\n"
            "Brief:\n{brief}\n\nCriterion: {criterion}\n"
        ),
        "On this criterion only, which image is better? Answer with 1 or 2 only.",
    ),
}
CHOICE = HERE / "results" / "judge_prompt_choice.json"


def items(pairs: pd.DataFrame, split: set) -> pd.DataFrame:
    it = pairs[pairs.split.isin(split)][["track", "dim", "scene", "a", "b"]].drop_duplicates()
    return it.reset_index(drop=True)


def job(prefix: str, prompt: str, brief: str, dim: str, imgs: list, extra: str = "") -> dict:
    pre, ask = PROMPTS[prompt]
    return {
        "id": prefix,
        "imgs": imgs,
        "pre": pre.format(brief=brief.strip(), criterion=CRITERION[dim]) + extra,
        "ask": ask,
        "mode": "score",
    }


def known_answer_items(sc: pd.DataFrame) -> list[dict]:
    """R5: per dev scene and part, the true image vs another dev scene's image (descriptions) or vs a degraded copy
    (aesthetics). The true image should win."""
    rnd = random.Random(5)
    out = []
    for track, dim in PARTS:
        dev = sc[(sc.track == track) & (sc.split == "dev")].reset_index(drop=True)
        for i, r in dev.iterrows():
            true = rnd.choice(r.assets)
            if track == "descriptions":
                other = dev.iloc[(i + 1 + rnd.randrange(len(dev) - 1)) % len(dev)]
                foil = rnd.choice(other.assets)
            else:
                foil = f"deg:{true}"
            out.append({"track": track, "dim": dim, "scene": int(r.scene), "true": int(true), "foil": foil})
    return out


def write_jobs(stage: str) -> None:
    d = load()
    pairs, sc = d["pairs"], d["scenes"]
    brief = dict(zip(sc.scene, sc.brief))
    JOBS.mkdir(parents=True, exist_ok=True)
    out = []
    if stage == "dev":
        for r in items(pairs, {"dev"}).itertuples():
            for pr in PROMPTS:
                for o, imgs in (("ab", [r.a, r.b]), ("ba", [r.b, r.a])):
                    out.append(
                        job(f"item|{pr}|{r.track}|{r.dim}|{r.scene}|{r.a}|{r.b}|{o}", pr, brief[r.scene], r.dim, imgs)
                    )
        for k in known_answer_items(sc):
            for pr in PROMPTS:
                for o, imgs in (("tf", [k["true"], k["foil"]]), ("ft", [k["foil"], k["true"]])):
                    out.append(
                        job(
                            f"ka|{pr}|{k['track']}|{k['dim']}|{k['scene']}|{k['true']}|{k['foil']}|{o}",
                            pr,
                            brief[k["scene"]],
                            k["dim"],
                            imgs,
                        )
                    )
    elif stage == "confirm":
        choice = json.loads(CHOICE.read_text())
        for model, pr in choice.items():
            rows = []
            for r in items(pairs, {"A", "B"}).itertuples():
                for o, imgs in (("ab", [r.a, r.b]), ("ba", [r.b, r.a])):
                    rows.append(
                        job(f"item|{pr}|{r.track}|{r.dim}|{r.scene}|{r.a}|{r.b}|{o}", pr, brief[r.scene], r.dim, imgs)
                    )
            (JOBS / f"confirm_{model}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            print(model, len(rows), "jobs")
        return
    elif stage == "smoke":
        for r in items(pairs, {"dev"}).head(12).itertuples():
            for pr in PROMPTS:
                out.append(
                    job(f"item|{pr}|{r.track}|{r.dim}|{r.scene}|{r.a}|{r.b}|ab", pr, brief[r.scene], r.dim, [r.a, r.b])
                )
    (JOBS / f"{stage}.jsonl").write_text("\n".join(json.dumps(x) for x in out) + "\n")
    print(stage, len(out), "jobs")


def read(model: str) -> dict:
    path = CACHE / "vlm" / f"{model}.jsonl"
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def item_probs(model: str, prompt: str) -> pd.DataFrame:
    """One row per item: p_a (both orders averaged), p_ab, p_ba, flip (the two orders disagree), mass."""
    R = read(model)
    acc = {}
    for k, r in R.items():
        f = k.split("|")
        if f[0] != "item" or f[1] != prompt:
            continue
        key = (f[2], f[3], int(f[4]), int(f[5]), int(f[6]))
        acc.setdefault(key, {})[f[7]] = r
    rows = []
    for (track, dim, scene, a, b), o in acc.items():
        if "ab" not in o or "ba" not in o:
            continue
        pab, pba = o["ab"]["p1"], 1 - o["ba"]["p1"]
        rows.append(
            {
                "track": track,
                "dim": dim,
                "scene": scene,
                "a": a,
                "b": b,
                "p": 0.5 * (pab + pba),
                "p_ab": pab,
                "p_ba": pba,
                "flip": (pab > 0.5) != (pba > 0.5),
                "mass": min(o["ab"]["mass"], o["ba"]["mass"]),
                "gpu_s": o["ab"].get("gpu_s", 0) + o["ba"].get("gpu_s", 0),
            }
        )
    return pd.DataFrame(rows)


def known_answers(model: str, prompt: str) -> dict:
    R = read(model)
    acc = {}
    for k, r in R.items():
        f = k.split("|")
        if f[0] != "ka" or f[1] != prompt:
            continue
        acc.setdefault(tuple(f[2:7]), {})[f[7]] = r
    right, masses, per = [], [], {}
    for key, o in acc.items():
        if "tf" not in o or "ft" not in o:
            continue
        p = 0.5 * (o["tf"]["p1"] + 1 - o["ft"]["p1"])
        ok = 1.0 if p > 0.5 else (0.5 if p == 0.5 else 0.0)
        right.append(ok)
        per.setdefault(key[0], []).append(ok)
        masses += [o["tf"]["mass"], o["ft"]["mass"]]
    return {
        "n": len(right),
        "agreement": float(np.mean(right)),
        "by_track": {k: float(np.mean(v)) for k, v in per.items()},
        "parse_share": float(np.mean(np.array(masses) >= 0.9)),
        "pass": float(np.mean(right)) >= 0.9 and float(np.mean(np.array(masses) >= 0.9)) >= 0.98,
    }


def agreement_vs_designers(ip: pd.DataFrame, pairs: pd.DataFrame) -> float:
    m = pairs.merge(ip, on=["track", "dim", "scene", "a", "b"])
    hit = np.where(np.abs(m.p - 0.5) < 1e-9, 0.5, (m.p > 0.5) == (m.y == 1))
    per_part = pd.Series(hit).groupby([m.track, m.dim]).mean()
    return float(per_part.mean())


def dev_choice() -> None:
    from heads import log_row

    d = load()
    pairs = d["pairs"]
    dev_pairs = pairs[pairs.split == "dev"]
    choice, report = {}, {}
    for model in ("qwen7", "qwen72", "gemma27"):
        if not (CACHE / "vlm" / f"{model}.jsonl").exists():
            continue
        best = None
        for pr in PROMPTS:
            ip = item_probs(model, pr)
            if ip.empty:
                continue
            ag = agreement_vs_designers(ip, dev_pairs)
            ka = known_answers(model, pr)
            report[f"{model}|{pr}"] = {
                "dev_agreement": ag,
                "flip_share": float(ip.flip.mean()),
                "R5": ka,
                "n_items": len(ip),
            }
            kept = best is None or ag > best[0]
            log_row(
                "dev",
                f"judge {model} prompt={pr}",
                ag,
                "kept" if kept else "discarded",
                f"R5 {ka['agreement']:.3f} parse {ka['parse_share']:.3f}; order flips {ip.flip.mean():.3f}",
            )
            if kept:
                best = (ag, pr)
        choice[model] = best[1]
    CHOICE.write_text(json.dumps(choice, indent=1))
    (HERE / "results" / "judges_dev.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), choice)


def frontier_order(track, dim, scene, a, b) -> str:
    h = int(hashlib.sha256(f"order:{track}:{dim}:{scene}:{a}:{b}".encode()).hexdigest()[:8], 16)
    return "ab" if h % 2 == 0 else "ba"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["jobs", "dev"])
    ap.add_argument("--stage", default="dev")
    a = ap.parse_args()
    if a.cmd == "jobs":
        write_jobs(a.stage)
    else:
        dev_choice()
