"""
Confirm-fold analysis for The Taste Stack Episode 7. Every number in the note comes from here.

Agreement = share of a designer's held-out pairs where the arm's choice (p > 0.5) matches the designer's; p = 0.5
counts 0.5. A part's agreement pools its pairs over designers; "mean over parts" weights the 9 parts equally.
Intervals: 95% bootstrap over scenes (resampled within track, 2,000 draws); differences are paired (same draws).

    python analyse.py        # writes results/{heads,judges,formal,constitution,router,buckets,summary}.json
"""

from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from formal import CHECK_FOR, image_scores, pair_p
from judges import CHOICE, item_probs
from router import EPS, TIERS, choose
from taste import CACHE, HERE, PART_LABEL, PARTS, load

RES = HERE / "results"
B = 2000
H100, A10G = 3.95, 1.10  # $/h, Modal list price
CPU_CORE, MEM_GIB = 0.0473, 0.008  # $/h


def hit(p, y):
    p = np.asarray(p, dtype=float)
    return np.where(np.abs(p - 0.5) < 1e-9, 0.5, (p > 0.5) == (np.asarray(y) == 1)).astype(float)


class Boot:
    """Scene bootstrap over a long frame with columns track, dim, scene and one hit column per arm."""

    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self.df = df
        self.parts = [p for p in PARTS if ((df.track == p[0]) & (df.dim == p[1])).any()]
        rng = np.random.default_rng(seed)
        self.scenes = {t: np.array(sorted(df[df.track == t].scene.unique())) for t in df.track.unique()}
        self.draws = {t: rng.integers(0, len(s), size=(B, len(s))) for t, s in self.scenes.items()}

    def _sums(self, col: str):
        out = {}
        for track, dim in self.parts:
            g = self.df[(self.df.track == track) & (self.df.dim == dim)]
            s = g.groupby("scene")[col].agg(["sum", "count"]).reindex(self.scenes[track]).fillna(0)
            out[(track, dim)] = (s["sum"].to_numpy(), s["count"].to_numpy())
        return out

    def stat(self, col: str, parts=None) -> dict:
        parts = parts or self.parts
        sums = self._sums(col)
        point = {p: sums[p][0].sum() / max(sums[p][1].sum(), 1) for p in parts}
        boots = np.zeros((B, len(parts)))
        for j, p in enumerate(parts):
            sm, ct = sums[p]
            dr = self.draws[p[0]]
            boots[:, j] = sm[dr].sum(1) / np.maximum(ct[dr].sum(1), 1)
        return {
            "point": point,
            "mean": float(np.mean(list(point.values()))),
            "boot": boots.mean(1),
            "boot_parts": boots,
            "parts": parts,
        }

    def summary(self, col: str, parts=None) -> dict:
        s = self.stat(col, parts)
        lo, hi = np.percentile(s["boot"], [2.5, 97.5])
        per = {PART_LABEL[p]: round(float(s["point"][p]), 4) for p in s["parts"]}
        ci_parts = {
            PART_LABEL[p]: [round(float(x), 4) for x in np.percentile(s["boot_parts"][:, j], [2.5, 97.5])]
            for j, p in enumerate(s["parts"])
        }
        return {
            "mean": round(s["mean"], 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "per_part": per,
            "per_part_ci": ci_parts,
        }

    def diff(self, a: str, b: str, parts=None) -> dict:
        sa, sb = self.stat(a, parts), self.stat(b, parts)
        d = sa["boot"] - sb["boot"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {
            "diff": round(sa["mean"] - sb["mean"], 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_le_0": round(float((d <= 0).mean()), 4),
        }


def loo_majority(df: pd.DataFrame) -> np.ndarray:
    g = df.groupby(["track", "dim", "scene", "a", "b"]).y
    tot, n = g.transform("sum"), g.transform("count")
    others = (tot - df.y) / (n - 1)
    return np.where(np.isclose(others, 0.5), 0.5, (others > 0.5).astype(float))


def others_split(df: pd.DataFrame) -> np.ndarray:
    g = df.groupby(["track", "dim", "scene", "a", "b"]).y
    k = g.transform("sum") - df.y  # others preferring a, of 4
    return np.where((k == 0) | (k == 4), "4-0", np.where((k == 1) | (k == 3), "3-1", "2-2"))


def formal_items(d: dict) -> pd.DataFrame:
    feat = pickle.loads((CACHE / "features.pkl").read_bytes())
    sc = d["scenes"]
    rows, proofs = [], {}
    for r in sc.itertuples():
        sc_scores = {a: image_scores(r.brief, r.track, feat["img"][a]) for a in r.assets}
        for a in r.assets:
            proofs[a] = {k: v[1] for k, v in sc_scores[a].items()}
        for i, a in enumerate(r.assets):
            for b in r.assets[i + 1 :]:
                for dim, chk in CHECK_FOR.items():
                    if chk is None:
                        continue
                    sa_, sb_ = sc_scores[a][chk][0], sc_scores[b][chk][0]
                    rows.append(
                        {
                            "track": r.track,
                            "dim": dim,
                            "scene": int(r.scene),
                            "a": a,
                            "b": b,
                            "p": pair_p(sa_, sb_),
                            "gap": abs(sa_ - sb_) if sa_ is not None and sb_ is not None else 0.0,
                            "defined": sc_scores[a][chk][0] is not None and sc_scores[b][chk][0] is not None,
                        }
                    )
    return pd.DataFrame(rows), proofs


def post_hoc(df: pd.DataFrame, bt: Boot) -> None:
    """Checks added after the confirm scores were read (plan amendment); none changes a registered number."""
    share = df.groupby(["track", "dim", "scene", "a", "b"]).y.transform("mean")
    df["h_bound5"] = hit(np.where(share > 0.5, 1.0, 0.0), df.y)  # all five designers, this one included
    df["h_pooled_c_own"] = hit(df.pooled_c_own, df.y)
    b = Boot(df)
    out = {
        "note": "post hoc (plan amendment, after the confirm scores); no registered number changes",
        "designer_blind_bound_in_sample": {
            **b.summary("h_bound5"),
            "what": "majority of all five designers, the scored designer included: no designer-blind verdict beats it",
        },
        "other_designers_majority": b.summary("h_loo"),
        "pooled_at_own_C": b.summary("h_pooled_c_own"),
        "diffs": {
            "own-pooled_at_own_C": b.diff("h_own", "h_pooled_c_own"),
            "gen_prior-own": b.diff("h_gen_prior", "h_own"),
            "gen_prior-gen_prior_pooled": b.diff("h_gen_prior", "h_gen_prior_pooled"),
            "own-resid_own": b.diff("h_own", "h_resid_own"),
            "stats-own": b.diff("h_stats", "h_own"),
            "gen_prior-loo": b.diff("h_gen_prior", "h_loo"),
        },
    }
    (RES / "post_hoc.json").write_text(json.dumps(out, indent=1))


def closed_known_answer(R: dict) -> dict | None:
    """The closed judge's known-answer dev pairs (frontier.py ka): the true image should win. Unparsed counts 0.5."""
    from frontier import answer_of

    ka = {k: r for k, r in R.items() if k.startswith("ka|")}
    if not ka:
        return None
    ans = {k: answer_of(r["text"]) for k, r in ka.items()}
    right = [0.5 if a is None else float((a == "1") == (k.split("|")[7] == "tf")) for k, a in ans.items()]
    return {
        "n": len(ka),
        "agreement": round(float(np.mean(right)), 4),
        "parsed_share": round(float(np.mean([a is not None for a in ans.values()])), 4),
        "cut_off_share": round(float(np.mean([r.get("finish") == "length" for r in ka.values()])), 4),
    }


def closed_arm(df, rt, tr_all, cost, available, KEY) -> None:
    """The exploratory closed judge (plan amendment): Claude Sonnet 5.5 on one pair per (scene, criterion)."""
    from frontier import OUT, answer_of, build, swap_subset
    from judges import frontier_order

    if not OUT.exists():
        return
    R = {r["id"]: r for r in map(json.loads, OUT.read_text().splitlines())}

    def p_of(j_id, strict):
        r = R.get(j_id)
        if r is None:
            return None
        a = answer_of(r["text"])
        if a is None or (strict and r.get("finish") == "length"):
            return 0.5
        return 1.0 if (a == "1") == (j_id.split("|")[7] == "ab") else 0.0

    rows = []
    for j in build("s1"):
        f = j["id"].split("|")
        r = R[j["id"]]
        rows.append(
            {
                "track": f[2],
                "dim": f[3],
                "scene": int(f[4]),
                "a": int(f[5]),
                "b": int(f[6]),
                "order": f[7],
                "p_closed": p_of(j["id"], True),
                "p_closed_loose": p_of(j["id"], False),
                "finish": r["finish"],
                "parsed": answer_of(r["text"]) is not None,
                "api_cost": r["cost"],
            }
        )
    S = pd.DataFrame(rows)
    choice = json.loads(CHOICE.read_text())
    for t, m_ in (("small", "qwen7"), ("mid", "gemma27"), ("large", "qwen72")):
        m = S[KEY + ["order"]].merge(item_probs(m_, choice[m_]), on=KEY, how="left")
        S[f"p_{t}_same_order"] = np.where(m.order == "ab", m.p_ab, m.p_ba)
    sub = df.merge(S, on=KEY)
    for c in ("p_closed", "p_closed_loose", "p_small_same_order", "p_mid_same_order", "p_large_same_order"):
        sub["h_" + c[2:]] = hit(sub[c], sub.y)
    bs = Boot(sub)
    arms = {
        "closed_strict": "h_closed",
        "closed_loose": "h_closed_loose",
        "qwen72_same_order": "h_large_same_order",
        "qwen7_same_order": "h_small_same_order",
        "gemma27_same_order": "h_mid_same_order",
        "qwen72_both_orders": "h_large",
        "probe_shrunk": "h_shrunk",
        "own_head": "h_own",
        "generator_prior": "h_gen_prior",
        "other_designers_majority": "h_loo",
    }
    out = {
        "note": "EXPLORATORY: added after the confirm scores were read; sample and comparisons fixed in the plan first",
        "model": "anthropic/claude-sonnet-5.5, reasoning effort low, max 300 output tokens, sync (batch refused inline images)",
        "n_items": len(S),
        "n_designer_pairs": len(sub),
        "parsed_share": round(float(S.parsed.mean()), 4),
        "cut_off_share": round(float((S.finish == "length").mean()), 4),
        "api_cost_per_call": round(float(S.api_cost.mean()), 5),
        "arms": {k: bs.summary(v) for k, v in arms.items()},
        "diffs": {
            "closed_strict-qwen72_same_order": bs.diff("h_closed", "h_large_same_order"),
            "closed_strict-probe_shrunk": bs.diff("h_closed", "h_shrunk"),
            "closed_loose-qwen72_same_order": bs.diff("h_closed_loose", "h_large_same_order"),
        },
        "known_answer": closed_known_answer(R),
    }
    par = sub[sub.parsed & (sub.finish != "length")]
    out["answered_only"] = {
        "n_designer_pairs": len(par),
        "closed": round(float(par.h_closed.mean()), 4),
        "qwen72_same_order": round(float(par.h_large_same_order.mean()), 4),
        "probe_shrunk": round(float(par.h_shrunk.mean()), 4),
    }
    ok = S[S.parsed & (S.finish != "length")]
    out["agrees_with_qwen72_same_order"] = round(
        float(((ok.p_closed > 0.5) == (ok.p_large_same_order > 0.5)).mean()), 4
    )
    a_ = load()["assets"]
    gl = {
        m: f"G{k + 1}"
        for k, m in enumerate(sorted(a_.model.unique(), key=lambda m: hashlib.sha256(m.encode()).hexdigest()))
    }
    gen = dict(zip(a_.asset_id.astype(int), a_.model.map(gl)))
    w = pd.concat(
        [
            pd.DataFrame({"g": ok.a.map(gen), "w": (ok.p_closed > 0.5).astype(float)}),
            pd.DataFrame({"g": ok.b.map(gen), "w": (ok.p_closed < 0.5).astype(float)}),
        ]
    )
    out["generator_win_rate_answered"] = w.groupby("g").w.mean().round(4).to_dict()
    sw = []
    for j in swap_subset(build("s2")):
        f = j["id"].split("|")
        first = "|".join([*f[:7], frontier_order(f[2], f[3], int(f[4]), int(f[5]), int(f[6]))])
        p1, p2 = p_of(first, True), p_of(j["id"], True)
        if p1 not in (None, 0.5) and p2 not in (None, 0.5):
            sw.append(p1 != p2)
    out["swap_check"] = {
        "items_answered_in_both_orders": len(sw),
        "verdict_flips_with_order": round(float(np.mean(sw)), 4) if sw else None,
    }
    keys = set(map(tuple, S[KEY].itertuples(index=False)))
    pick = lambda frame: (
        frame[[tuple(x) in keys for x in frame[KEY].itertuples(index=False)]]
        .merge(  # noqa: E731
            S[KEY + ["p_closed"]], on=KEY
        )
        .rename(columns={"p_closed": "closed"})
    )
    rts, trs = pick(rt), pick(tr_all)
    c2 = {**cost, "closed": float(S.api_cost.mean())}
    av2 = {k: {**v, "closed": True} for k, v in available.items()}
    res = {}
    for name, tiers in (("without_closed", TIERS), ("with_closed", [*TIERS, "closed"])):
        p, c, chosen = np.zeros(len(rts)), np.zeros(len(rts)), {}
        for tef, trf in (("A", "B"), ("B", "A")):
            ch = choose(trs[trs.fold_test == f"inner:{trf}"], c2, av2, tiers=tiers)
            chosen[f"train {trf} -> test {tef}"] = ch
            m = (rts.fold_test == tef).to_numpy()
            p[m] = [rts.loc[i, ch[pt]] for i, pt in zip(rts.index[m], rts.part[m])]
            c[m] = [c2[ch[pt]] for pt in rts.part[m]]
        rts["h_r"] = hit(p, rts.y)
        res[name] = {**Boot(rts).summary("h_r"), "cost_per_1k": round(1000 * float(c.mean()), 4), "choices": chosen}
    out["router_on_these_items"] = {
        **res,
        "costs_per_decision": {k: round(v, 6) for k, v in c2.items()},
        "cost_units": "closed: API dollars per call (one order); the others: Modal compute per decision",
    }
    (RES / "closed_judge.json").write_text(json.dumps(out, indent=1))


def main():
    d = load()
    pairs = d["pairs"]
    conf = pairs[pairs.split.isin(["A", "B"])].reset_index(drop=True)
    KEY = ["track", "dim", "scene", "a", "b"]
    heads = pd.read_pickle(CACHE / "heads_confirm.pkl")
    test = heads[heads.fold_test.isin(["A", "B"])].reset_index(drop=True)
    inner = heads[heads.fold_test.str.startswith("inner:")].reset_index(drop=True)
    assert len(test) == len(conf), (len(test), len(conf))
    choice = json.loads(CHOICE.read_text())
    J = {
        "small": item_probs("qwen7", choice["qwen7"]),
        "mid": item_probs("gemma27", choice["gemma27"]),
        "large": item_probs("qwen72", choice["qwen72"]),
    }
    F, _proofs = formal_items(d)
    df = test.copy()
    for tier, ip in J.items():
        if len(ip):
            df = df.merge(ip[KEY + ["p"]].rename(columns={"p": f"p_{tier}"}), on=KEY, how="left")
    df = df.merge(
        F[KEY + ["p", "defined"]].rename(columns={"p": "p_formal", "defined": "formal_defined"}), on=KEY, how="left"
    )
    df["p_formal"] = df.p_formal.fillna(0.5)
    df["formal_defined"] = df.formal_defined.fillna(False).astype(bool)
    df["p_loo"] = loo_majority(df)
    df["split4"] = others_split(df)
    arms = [
        "pooled",
        "pooled_fifth",
        "own",
        "shrunk",
        "other",
        "gen_prior",
        "gen_prior_pooled",
        "stats",
        "resid_pooled",
        "resid_own",
        "resid_shrunk",
        "bright",
    ]
    for a in arms:
        if a == "bright":
            df["h_bright"] = hit(df.bright, df.bright_y)
        else:
            df[f"h_{a}"] = hit(df[a], df.y)
    for t in ("formal", "small", "mid", "large", "loo"):
        if f"p_{t}" in df:
            df[f"h_{t}"] = hit(df[f"p_{t}"].fillna(0.5), df.y)
    for k in range(10):
        df[f"h_shuf_pooled_{k}"] = hit(df.shuffled_pooled.map(lambda v, k=k: v[k]), df.y)
        df[f"h_shuf_own_{k}"] = hit(df.shuffled_own.map(lambda v, k=k: v[k]), df.y)
    bt = Boot(df)
    out_heads = {a: bt.summary(f"h_{a}") for a in arms}
    shuf_p = [bt.stat(f"h_shuf_pooled_{k}")["mean"] for k in range(10)]
    shuf_o = [bt.stat(f"h_shuf_own_{k}")["mean"] for k in range(10)]
    out_heads["shuffled_pooled"] = {
        "mean": round(float(np.mean(shuf_p)), 4),
        "range": [round(min(shuf_p), 4), round(max(shuf_p), 4)],
    }
    out_heads["shuffled_own"] = {
        "mean": round(float(np.mean(shuf_o)), 4),
        "range": [round(min(shuf_o), 4), round(max(shuf_o), 4)],
    }
    out_heads["human_loo_ceiling"] = bt.summary("h_loo")
    out_heads["diffs"] = {
        "shrunk-pooled": bt.diff("h_shrunk", "h_pooled"),
        "own-pooled": bt.diff("h_own", "h_pooled"),
        "own-other": bt.diff("h_own", "h_other"),
        "shrunk-other": bt.diff("h_shrunk", "h_other"),
        "own-pooled_fifth": bt.diff("h_own", "h_pooled_fifth"),
        "pooled-gen_prior_pooled": bt.diff("h_pooled", "h_gen_prior_pooled"),
        "shrunk-gen_prior": bt.diff("h_shrunk", "h_gen_prior"),
        "resid_shrunk-resid_pooled": bt.diff("h_resid_shrunk", "h_resid_pooled"),
    }
    # per designer: own, shrunk, pooled, loo
    per_des = (
        df.groupby("designer")[["h_own", "h_shrunk", "h_pooled", "h_other", "h_gen_prior", "h_loo"]].mean().round(4)
    )
    out_heads["per_designer"] = per_des.to_dict(orient="index")
    out_heads["n_pairs"] = len(df)
    out_heads["n_scenes"] = int(df.scene.nunique())
    (RES / "heads.json").write_text(json.dumps(out_heads, indent=1))

    # judges
    out_j = {}
    for t in ("formal", "small", "mid", "large"):
        if f"h_{t}" in df and df[f"p_{t}"].notna().any():
            cov = float(df[f"p_{t}"].notna().mean())
            out_j[t] = {**bt.summary(f"h_{t}"), "coverage": round(cov, 4)}
    for t, ip in J.items():
        if t in out_j and "flip" in ip:
            conf_items = ip.merge(conf[KEY].drop_duplicates(), on=KEY)
            out_j[t]["order_flip_share"] = round(float(conf_items.flip.mean()), 4)
            out_j[t]["parse_share"] = round(float((conf_items.mass >= 0.9).mean()), 4)
    # formal only where defined
    fd = df[df.formal_defined].copy()
    if len(fd):
        bfd = Boot(fd)
        out_j["formal_defined_only"] = {
            **bfd.summary("h_formal"),
            "share_defined": round(float(df.formal_defined.mean()), 4),
        }
    # cross-family agreement (R6) on confirm items: Qwen-72B vs Gemma-27B
    m = (
        J["large"][KEY + ["p"]]
        .merge(J["mid"][KEY + ["p"]], on=KEY, suffixes=("_l", "_m"))
        .merge(conf[KEY].drop_duplicates(), on=KEY)
    )
    agree = float(((m.p_l > 0.5) == (m.p_m > 0.5)).mean())
    out_j["R6_large_vs_mid_agreement"] = {"agreement": round(agree, 4), "n_items": len(m), "pass": agree >= 0.60}
    if len(J["small"]) and len(J["large"]):
        m = (
            J["small"][KEY + ["p"]]
            .merge(J["large"][KEY + ["p"]], on=KEY, suffixes=("_s", "_l"))
            .merge(conf[KEY].drop_duplicates(), on=KEY)
        )
        out_j["small_vs_large_agreement"] = round(float(((m.p_s > 0.5) == (m.p_l > 0.5)).mean()), 4)
    probe_best = max(("pooled", "own", "shrunk"), key=lambda a: out_heads[a]["mean"])
    judge_best = max((t for t in ("small", "mid", "large") if t in out_j), key=lambda t: out_j[t]["mean"])
    out_j["P3"] = {"best_probe": probe_best, "best_judge": judge_best, **bt.diff(f"h_{probe_best}", f"h_{judge_best}")}
    (RES / "judges.json").write_text(json.dumps(out_j, indent=1))

    # generator win rates (generators anonymised G1-G4 in a fixed hash order; no mapping is published)
    a_ = d["assets"]
    gl = {
        m: f"G{k + 1}"
        for k, m in enumerate(sorted(a_.model.unique(), key=lambda m: hashlib.sha256(m.encode()).hexdigest()))
    }
    gen = dict(zip(a_.asset_id.astype(int), a_.model.map(gl)))
    wins = {}
    for src, col in (
        ("designers", "y"),
        ("Qwen2.5-VL-7B", "p_small"),
        ("Gemma-3-27B", "p_mid"),
        ("Qwen2.5-VL-72B", "p_large"),
    ):
        v = (df[col] > 0.5).astype(float) if col != "y" else df.y.astype(float)
        w = pd.concat([pd.DataFrame({"g": df.a.map(gen), "w": v}), pd.DataFrame({"g": df.b.map(gen), "w": 1 - v})])
        wins[src] = w.groupby("g").w.mean().round(4).to_dict()
    per_des = {}
    for des, g in df.groupby("designer"):
        w = pd.concat([pd.DataFrame({"g": g.a.map(gen), "w": g.y}), pd.DataFrame({"g": g.b.map(gen), "w": 1 - g.y})])
        per_des[des] = w.groupby("g").w.mean().round(4).to_dict()
    wins["per_designer"] = per_des
    per_part = {}
    for (t_, dm), g in df.groupby(["track", "dim"]):
        w = pd.concat([pd.DataFrame({"g": g.a.map(gen), "w": g.y}), pd.DataFrame({"g": g.b.map(gen), "w": 1 - g.y})])
        per_part[PART_LABEL[(t_, dm)]] = w.groupby("g").w.mean().round(4).to_dict()
    wins["designers_per_part"] = per_part
    wins["same_maker_as_gemma"] = gl[next(m for m in gl if "Banana" in m)]
    (RES / "generators.json").write_text(json.dumps(wins, indent=1))

    # buckets
    out_b = {}
    for bucket in ("4-0", "3-1", "2-2"):
        sub = df[df.split4 == bucket]
        out_b[bucket] = {
            "n": len(sub),
            **{
                a: round(float(sub[f"h_{a}"].mean()), 4)
                for a in ("pooled", "own", "shrunk", "gen_prior", "loo", "formal", "small", "mid", "large")
                if f"h_{a}" in sub
            },
        }
    (RES / "buckets.json").write_text(json.dumps(out_b, indent=1))

    # router
    timing = json.loads((RES / "timing.json").read_text())
    vt = [json.loads(x) for x in (CACHE / "vlm" / "timing.jsonl").read_text().splitlines()]
    gpu_s = {}
    for m_ in ("qwen7", "gemma27", "qwen72"):
        rows = [t for t in vt if t["model"] == m_ and t["jobs"].startswith("confirm")]
        gpu_s[m_] = sum(t["gen_s"] for t in rows) / max(sum(t["n"] for t in rows), 1)
    cost = {
        "formal": 2
        * timing["formal_s_per_image"]
        * (timing["cpu_cores"] * CPU_CORE + timing["memory_gib"] * MEM_GIB)
        / 3600,
        "probe": 2 * timing["probe_s_per_image"] * A10G / 3600,
        "small": 2 * gpu_s["qwen7"] * H100 / 3600,
        "mid": 2 * gpu_s["gemma27"] * H100 / 3600,
        "large": 2 * gpu_s["qwen72"] * H100 / 3600,
    }
    rt = df.copy()
    rt["part"] = rt.track + "|" + rt.dim
    rt["formal"] = rt.p_formal
    rt["probe"] = rt.shrunk
    for t in ("small", "mid", "large"):
        rt[t] = rt[f"p_{t}"].fillna(0.5)
    # training-fold estimates: zero-shot tiers on the training fold's pairs; the probe from inner CV
    tr_all = inner.copy()
    for tier, ip in J.items():
        if len(ip):
            tr_all = tr_all.merge(ip[KEY + ["p"]].rename(columns={"p": tier}), on=KEY, how="left")
    tr_all = tr_all.merge(F[KEY + ["p"]].rename(columns={"p": "formal"}), on=KEY, how="left")
    tr_all["probe"] = tr_all.shrunk_innercv
    tr_all["part"] = tr_all.track + "|" + tr_all.dim
    for t in TIERS:
        if t in tr_all:
            tr_all[t] = tr_all[t].fillna(0.5)
    available = {
        f"{t}|{dm}": {"formal": CHECK_FOR[dm] is not None, "probe": True, "small": True, "mid": True, "large": True}
        for t, dm in PARTS
    }
    choices, rp, rc = {}, np.zeros(len(rt)), np.zeros(len(rt))
    for tef, trf in (("A", "B"), ("B", "A")):
        tr = tr_all[tr_all.fold_test == f"inner:{trf}"]
        ch = choose(tr, cost, available)
        choices[f"train {trf} -> test {tef}"] = ch
        m = (rt.fold_test == tef).to_numpy()
        rp[m] = [rt.loc[i, ch[pt]] for i, pt in zip(rt.index[m], rt.part[m])]
        rc[m] = [cost[ch[pt]] for pt in rt.part[m]]
    rt["router"] = rp
    # per part and fold direction: each tier's training-fold and held-out agreement (aggregates for the explorer)
    table = []
    for tef, trf in (("A", "B"), ("B", "A")):
        tr = tr_all[tr_all.fold_test == f"inner:{trf}"]
        te = rt[rt.fold_test == tef]
        for t_, dm in PARTS:
            pt = f"{t_}|{dm}"
            g_tr, g_te = tr[tr.part == pt], te[te.part == pt]
            table.append(
                {
                    "part": PART_LABEL[(t_, dm)],
                    "key": pt,
                    "train": trf,
                    "test": tef,
                    "n_test": len(g_te),
                    "train_acc": {t: round(float(hit(g_tr[t], g_tr.y).mean()), 4) for t in TIERS if available[pt][t]},
                    "test_hits": {t: float(hit(g_te[t], g_te.y).sum()) for t in TIERS if available[pt][t]},
                }
            )
    df["h_router"] = hit(rp, rt.y)
    # oracle router: best tier per part on the held-out fold (ceiling)
    orc = {}
    for pt, g in rt.groupby("part"):
        accs = {t: float(hit(g[t], g.y).mean()) for t in TIERS if available[pt][t]}
        orc[pt] = max(accs, key=accs.get)
    df["h_oracle_router"] = hit([rt.loc[i, orc[pt]] for i, pt in zip(rt.index, rt.part)], rt.y)
    bt = Boot(df)
    single = {
        t: {**bt.summary(f"h_{t}"), "cost_per_1k": round(1000 * cost[t], 5)}
        for t in ("formal", "small", "mid", "large")
        if f"h_{t}" in df
    }
    single["probe"] = {**bt.summary("h_shrunk"), "cost_per_1k": round(1000 * cost["probe"], 5)}
    best_single = max(single, key=lambda t: single[t]["mean"])
    router_cost = float(rc.mean())
    out_r = {
        "eps": EPS,
        "cost_per_decision": {k: float(v) for k, v in cost.items()},
        "choices": choices,
        "router": {**bt.summary("h_router"), "cost_per_1k": round(1000 * router_cost, 5)},
        "oracle_router": {
            **bt.summary("h_oracle_router"),
            "choice": orc,
            "cost_per_1k": round(1000 * float(np.mean([cost[orc[pt]] for pt in rt.part])), 5),
        },
        "single": single,
        "best_single": best_single,
        "table": table,
        "router_minus_best_single": bt.diff("h_router", "h_shrunk" if best_single == "probe" else f"h_{best_single}"),
    }
    # post-hoc sensitivity: the same rule under the repeat timing runs (cold vs warm reads moved the cheap tiers' costs)
    sens = []
    for run in json.loads((RES / "timing_runs.json").read_text())[1:]:
        for probe_key in ("probe_s_per_image", "probe_jpg640_s_per_image"):
            if probe_key not in run:
                continue
            c2 = dict(cost)
            c2["formal"] = (
                2 * run["formal_s_per_image"] * (run["cpu_cores"] * CPU_CORE + run["memory_gib"] * MEM_GIB) / 3600
            )
            c2["probe"] = 2 * run[probe_key] * A10G / 3600
            p2, k2 = np.zeros(len(rt)), np.zeros(len(rt))
            for tef, trf in (("A", "B"), ("B", "A")):
                ch = choose(tr_all[tr_all.fold_test == f"inner:{trf}"], c2, available)
                m = (rt.fold_test == tef).to_numpy()
                p2[m] = [rt.loc[i, ch[pt]] for i, pt in zip(rt.index[m], rt.part[m])]
                k2[m] = [c2[ch[pt]] for pt in rt.part[m]]
            df["h_router_s"] = hit(p2, rt.y)
            s2 = Boot(df).summary("h_router_s")
            sens.append(
                {
                    "timing_run": run["run"],
                    "probe_input": "640 px JPEG" if "jpg640" in probe_key else "full image",
                    "cost_per_1k": {k: round(1000 * v, 5) for k, v in c2.items()},
                    "router": s2["mean"],
                    "router_ci": s2["ci"],
                    "router_cost_per_1k": round(1000 * float(k2.mean()), 5),
                    "cost_ratio_to_probe": round(float(k2.mean()) / c2["probe"], 3),
                }
            )
    df = df.drop(columns=["h_router_s"], errors="ignore")
    out_r["sensitivity_post_hoc"] = sens
    out_r["P4"] = {
        "router_ge_best_minus_0.01": out_r["router"]["mean"] >= single[best_single]["mean"] - 0.01,
        "cost_ratio": round(router_cost / cost[best_single], 4) if cost[best_single] > 0 else None,
    }
    out_r["P4"]["hit"] = bool(out_r["P4"]["router_ge_best_minus_0.01"] and (out_r["P4"]["cost_ratio"] or 9) <= 0.25)
    (RES / "router.json").write_text(json.dumps(out_r, indent=1))

    # exploratory: the OCR text check on text rendering, split by how far apart the two images' scores are
    tr_ = df[(df.track == "descriptions") & (df.dim == "typography")].merge(F[KEY + ["gap"]], on=KEY, how="left")
    gaps = {}
    for lo_, hi_ in ((0, 1e-9), (1e-9, 0.2), (0.2, 9)):
        sub = tr_[(tr_.gap >= lo_) & (tr_.gap < hi_)]
        gaps[f"{lo_:g}-{hi_:g}"] = {
            "n_pairs": len(sub),
            "agreement": round(float(sub.h_formal.mean()), 4) if len(sub) else None,
        }
    (RES / "formal.json").write_text(
        json.dumps(
            {
                "exploratory_text_rendering_by_score_gap": gaps,
                "descriptions_typography": single["formal"]["per_part"].get(PART_LABEL[("descriptions", "typography")]),
                "defined_share_by_part": {
                    PART_LABEL[(t, dm)]: round(float(df[(df.track == t) & (df.dim == dm)].formal_defined.mean()), 4)
                    for t, dm in PARTS
                },
                "summary": single["formal"],
            },
            indent=1,
        )
    )

    # constitutions
    cst_path = CACHE / "cst_eval.pkl"
    if cst_path.exists():
        ce = pd.read_pickle(cst_path)
        pref = df[df.dim == "preference"].copy()
        designers = {t: sorted(pref[pref.track == t].designer.unique()) for t in pref.track.unique()}
        nxt = {d_: ds[(i + 1) % len(ds)] for ds in designers.values() for i, d_ in enumerate(ds)}
        train_of = {"A": "B", "B": "A"}
        look = {(r.owner, r.train_fold, r.scene, r.a, r.b): r.p for r in ce.itertuples()}

        def lp(owner, r):
            return look.get((owner, train_of[r.fold_test], r.scene, r.a, r.b), np.nan)

        pref["p_cst_own"] = [lp(r.designer, r) for r in pref.itertuples()]
        pref["p_cst_other"] = [lp(nxt[r.designer], r) for r in pref.itertuples()]
        pref["p_cst_pooled"] = [lp(f"pooled:{r.track}", r) for r in pref.itertuples()]
        for k in ("own", "other", "pooled"):
            pref[f"h_cst_{k}"] = hit(pref[f"p_cst_{k}"].fillna(0.5), pref.y)
        pref["h_cst_none"] = pref.h_large
        bc = Boot(pref)
        out_c = {k: bc.summary(f"h_cst_{k}") for k in ("none", "own", "other", "pooled")}
        out_c["coverage_own"] = round(float(pref.p_cst_own.notna().mean()), 4)
        out_c["diffs"] = {
            "own-other": bc.diff("h_cst_own", "h_cst_other"),
            "own-none": bc.diff("h_cst_own", "h_cst_none"),
            "own-pooled": bc.diff("h_cst_own", "h_cst_pooled"),
            "pooled-none": bc.diff("h_cst_pooled", "h_cst_none"),
        }
        out_c["probe_on_preference"] = {a: bc.summary(f"h_{a}") for a in ("pooled", "shrunk", "own")}
        out_c["loo_on_preference"] = bc.summary("h_loo")
        br = ce[ce.owner == "BRIGHT"]
        if len(br):
            feat = pickle.loads((CACHE / "features.pkl").read_bytes())
            from formal import srgb_to_lab

            L = {int(a): float(srgb_to_lab(f["thumb"].reshape(-1, 3))[:, 0].mean()) for a, f in feat["img"].items()}
            yb = np.array([int(L[a] > L[b]) for a, b in zip(br.a, br.b)])
            none = J["large"].merge(br[["scene", "a", "b"]], on=["scene", "a", "b"])
            none = none[none.dim == "preference"]
            ybn = np.array([int(L[a] > L[b]) for a, b in zip(none.a, none.b)])
            out_c["R7_bright"] = {
                "own_constitution": round(float(hit(br.p, yb).mean()), 4),
                "n": len(br),
                "no_constitution": round(float(hit(none.p, ybn).mean()), 4),
                "pass": float(hit(br.p, yb).mean()) >= 0.80,
            }
        (RES / "constitution.json").write_text(json.dumps(out_c, indent=1))

    post_hoc(df, bt)
    closed_arm(df, rt, tr_all, cost, available, KEY)

    summary = {
        "heads": {a: out_heads[a]["mean"] for a in arms},
        "ceiling": out_heads["human_loo_ceiling"]["mean"],
        "judges": {t: out_j[t]["mean"] for t in ("formal", "small", "mid", "large") if t in out_j},
        "router": out_r["router"]["mean"],
        "router_cost_per_1k": out_r["router"]["cost_per_1k"],
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
