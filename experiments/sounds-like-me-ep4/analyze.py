"""Agreement between the rulers and people, per dataset; scores the plan's predictions.

    python analyze.py

Reads results/labels_h*.json and every scores_*.json; the user's ratings from rate/ratings/ if present.
Writes results/summary.json and results/predictions.json. Confirmatory numbers are on held-out splits only;
dev numbers are reported under "dev". Every score is oriented so that higher = "more like the person"
(H1, H2, user) or higher = "more AI-like" (H3).
"""

import json
from collections import defaultdict

import numpy as np

from common4 import HERE, RES, auroc, boot, dump, perm_floor, rm_corr, spearman

PANEL = ["qwen3-32b", "gemma-3-27b", "llama-3.1-8b"]
P6_NOT_TESTED = "not tested: the human rating arm was replaced at the user's request"
RATINGS = HERE / "rate" / "ratings" / "sounds-like-me-ep4-ratings.json"


def J(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def zmap(d):
    v = np.array(list(d.values()), float)
    mu, sd = v.mean(), v.std() or 1.0
    return {k: (x - mu) / sd for k, x in d.items()}


def panel_mean(dicts, z=True):
    """Panel = mean of per-model z-scores over the keys every model scored. For H2, whose sign is the pick,
    z=False averages the raw centred scores (z-scoring would move the zero point; code review fix)."""
    zs = [zmap(d) if z else d for d in dicts if d]
    keys = set.intersection(*[set(z) for z in zs]) if zs else set()
    return {k: float(np.mean([z[k] for z in zs])) for k in keys}


def partial_rm(groups, x, y, z):
    """Repeated-measures partial correlation of x and y controlling for z (all centred within person)."""
    groups, x, y, z = map(np.asarray, (groups, x, y, z))
    cx, cy, cz = [], [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        cx.append(x[m] - x[m].mean())
        cy.append(y[m] - y[m].mean())
        cz.append(z[m] - z[m].mean())
    cx, cy, cz = map(np.concatenate, (cx, cy, cz))
    rx = cx - cz * (cx @ cz) / (cz @ cz)
    ry = cy - cz * (cy @ cz) / (cz @ cz)
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def panel_rulers(field, keep=None, z=True):
    """Each panel model's score per item (H2: P(B closer) averaged over both orders, minus 0.5) and the panel mean."""
    out, pans = {}, []
    for m in PANEL:
        d = J(f"scores_panel_{m}.json")
        if d:
            v = {
                k: (x["p_b"] - 0.5 if isinstance(x, dict) else x)
                for k, x in d[field].items()
                if keep is None or k in keep
            }
            out[f"panel_{m}"] = v
            pans.append(v)
    if pans:
        out["panel"] = panel_mean(pans, z=z)
    return out


# ---------------------------------------------------------------- H1
def h1_rulers():
    S, P = J("scores_local_h1.json"), J("scores_pref.json")
    R = {
        "luar": {k: v["luar_mean_strict"] for k, v in S.items() if "luar_mean_strict" in v},
        "luar_loose": {k: v["luar_mean_raw"] for k, v in S.items() if "luar_mean_raw" in v},
        "spec": {k: v["spec_mean_strict"] for k, v in S.items() if "spec_mean_strict" in v},
        "detector_neg": {k: -v["det"] for k, v in S.items()},
        "tfidf_char": {k: v["tfidf_char"] for k, v in S.items() if "tfidf_char" in v},
        "tfidf_word": {k: v["tfidf_word"] for k, v in S.items() if "tfidf_word" in v},
        "length": {k: v["words"] for k, v in S.items()},
    }
    if P:
        R["pref"] = P["h1"]
    R.update(panel_rulers("h1"))
    return R, S


def h1(split):
    L = J("labels_h1.json")
    R, S = h1_rulers()
    T = [r for r in L if r["split"] == split and r["kind"] != "control" and r["capture"] is not None]
    by = {r["tid"]: r for r in T}
    pairs = []  # (pid, edit_row, draft_row)
    for r in T:
        if r["kind"] == "edit" and r["tid"].replace("-edit", "-draft") in by:
            pairs.append((r["pid"], r, by[r["tid"].replace("-edit", "-draft")]))
    # unchanged edits (edit text == draft text) can't be told apart by any text ruler: excluded from all three groups
    up = [p for p in pairs if p[1]["capture"] > p[2]["capture"] and not p[1]["unchanged"]]
    eq = [p for p in pairs if p[1]["capture"] == p[2]["capture"] and not p[1]["unchanged"]]
    down = [p for p in pairs if p[1]["capture"] < p[2]["capture"] and not p[1]["unchanged"]]
    out = dict(
        n_people=len({r["pid"] for r in T}),
        n_texts=len(T),
        n_tasks=len(pairs),
        n_up=len(up),
        n_equal=len(eq),
        n_down=len(down),
        n_unchanged=sum(p[1]["unchanged"] for p in pairs),
        rating_spread=dict(
            draft=np.bincount([p[2]["capture"] for p in pairs], minlength=6)[1:].tolist(),
            edit=np.bincount([p[1]["capture"] for p in pairs], minlength=6)[1:].tolist(),
        ),
        rm_r={},
        rm_r_floor={},
        pairwise_up={},
        prefers_edit_equal={},
        prefers_edit_down={},
        length_matched_up={},
    )
    people = sorted({r["pid"] for r in T})

    def acc(ps, sc):
        ps = [p for p in ps if p[1]["tid"] in sc and p[2]["tid"] in sc]
        return (
            float(
                np.mean(
                    [
                        1.0 if sc[e["tid"]] > sc[d["tid"]] else 0.5 if sc[e["tid"]] == sc[d["tid"]] else 0.0
                        for _, e, d in ps
                    ]
                )
            )
            if ps
            else float("nan")
        )  # ties count half

    for name, sc in R.items():
        rows = [r for r in T if r["tid"] in sc]
        if len(rows) < 10:
            continue
        g = np.array([r["pid"] for r in rows])
        x = [sc[r["tid"]] for r in rows]
        y = [r["capture"] for r in rows]
        bypid = defaultdict(list)
        for r in rows:
            bypid[r["pid"]].append(r)

        def stat(ps, sc=sc, bypid=bypid):
            rr = [(i, r) for i, p in enumerate(ps) for r in bypid.get(p, [])]  # resampled people stay distinct
            return rm_corr([i for i, _ in rr], [sc[r["tid"]] for _, r in rr], [r["capture"] for _, r in rr])

        out["rm_r"][name] = boot(people, stat)

        def shuffle(rng_, g=g, y=y):
            ys = np.array(y, float)
            for p in np.unique(g):
                m = np.where(g == p)[0]
                ys[m] = ys[rng_.permutation(m)]
            return ys

        out["rm_r_floor"][name] = perm_floor(lambda ys, g=g, x=x: rm_corr(g, x, ys), shuffle, n=500)

        def by_people(sub, sc=sc):
            grp = defaultdict(list)
            for p in sub:
                grp[p[0]].append(p)
            return boot(sorted(grp), lambda ks: acc([p for k in ks for p in grp[k]], sc))

        out["pairwise_up"][name] = by_people(up)
        out["prefers_edit_equal"][name] = by_people(eq)
        # share preferring the edit where the writer preferred the draft (agreement with the writer = 1 - this)
        out["prefers_edit_down"][name] = [round(acc(down, sc), 4), len(down)]
        lm = [p for p in up if abs(p[1]["words"] - p[2]["words"]) <= 0.15 * max(p[2]["words"], 1)]
        out["length_matched_up"][name] = [round(acc(lm, sc), 4), len(lm)]
    # agreement once AI-ness is held fixed (the detector partial)
    rows = [r for r in T if r["tid"] in R["luar"]]
    grp, cap, det = [r["pid"] for r in rows], [r["capture"] for r in rows], [R["detector_neg"][r["tid"]] for r in rows]
    wc = [r["words"] for r in rows]
    names_p = [n for n in ("luar", "spec", "pref", "panel") if n in R and all(r["tid"] in R[n] for r in rows)]
    out["partial_on_detector"] = {
        n: round(partial_rm(grp, [R[n][r["tid"]] for r in rows], cap, det), 4) for n in names_p
    }
    out["partial_on_length"] = {n: round(partial_rm(grp, [R[n][r["tid"]] for r in rows], cap, wc), 4) for n in names_p}
    # how far "captures my style" and "usable" move together (a rating of quality, not voice?)
    rr = [r for r in T if r["useable"] is not None]
    out["rm_r_usable_vs_capture"] = round(
        rm_corr([r["pid"] for r in rr], [r["useable"] for r in rr], [r["capture"] for r in rr]), 4
    )
    # ruler x ruler: share of tasks where two rulers agree on which text is closer to the person
    names = [n for n in ("luar", "spec", "pref", "panel", "detector_neg", "tfidf_char", "length") if n in R]
    agree = {}
    for a in names:
        for b in names:
            ps = [p for p in pairs if all(p[i]["tid"] in R[a] and p[i]["tid"] in R[b] for i in (1, 2))]
            agree[f"{a}|{b}"] = (
                round(
                    float(
                        np.mean(
                            [(R[a][e["tid"]] > R[a][d["tid"]]) == (R[b][e["tid"]] > R[b][d["tid"]]) for _, e, d in ps]
                        )
                    ),
                    3,
                )
                if ps
                else None
            )
    out["ruler_agreement"] = dict(names=names, pairs=agree)
    # edit size: does each ruler's hit rate on "edit rated higher" depend on how much was changed?
    if up:
        d = np.array([S[e["tid"]].get("dist_from_draft", 0) for _, e, _ in up])
        cut = np.quantile(d, [1 / 3, 2 / 3])
        masks = [d <= cut[0], (d > cut[0]) & (d <= cut[1]), d > cut[1]]
        out["up_by_edit_size"] = {}
        for n in ("luar", "pref", "panel", "detector_neg"):
            if n in R:
                hit = np.array([R[n].get(e["tid"], 0) > R[n].get(dd["tid"], 0) for _, e, dd in up])
                out["up_by_edit_size"][n] = [
                    dict(tertile=i + 1, acc=round(float(hit[m].mean()), 3), n=int(m.sum())) for i, m in enumerate(masks)
                ]
    return out


def h1_checks():
    out = {"luar": J("ruler_luar_h1_check.json")}
    P = J("scores_pref.json")
    if P:
        out["pref"] = P["check"]
    for m in PANEL:
        d = J(f"scores_panel_{m}.json")
        if d:
            own, oth = d["h1_ka_own"], d["h1_ka_other"]
            ks = sorted(set(own) & set(oth))
            v = np.mean([1.0 if own[k] > oth[k] else 0.5 if own[k] == oth[k] else 0.0 for k in ks])  # ties: half
            out[f"panel_{m}"] = dict(own_vs_other=round(float(v), 4), n=len(ks), pass_bar=0.65, passed=bool(v >= 0.65))
    return out


# ---------------------------------------------------------------- H2
def h2(split):
    L, S, P = J("labels_h2.json"), J("scores_local_h2.json"), J("scores_pref.json")
    T = [t for t in L if t["split"] == split and not t["golden"] and t["n"] >= 5 and 2 * t["votes_b"] != t["n"]]
    M = {
        "luar": {t["tri"]: S[t["tri"]]["luar_ab_strict"] - S[t["tri"]]["luar_ac_strict"] for t in T},
        "luar_loose": {t["tri"]: S[t["tri"]]["luar_ab_raw"] - S[t["tri"]]["luar_ac_raw"] for t in T},
        "tfidf_char": {t["tri"]: S[t["tri"]]["tfidf_char_ab"] - S[t["tri"]]["tfidf_char_ac"] for t in T},
        "tfidf_word": {t["tri"]: S[t["tri"]]["tfidf_word_ab"] - S[t["tri"]]["tfidf_word_ac"] for t in T},
        "length": {t["tri"]: S[t["tri"]]["words_ab"] - S[t["tri"]]["words_ac"] for t in T},
    }
    if P:
        M["pref"] = {t["tri"]: P["h2"][t["tri"]][0] - P["h2"][t["tri"]][1] for t in T if t["tri"] in P["h2"]}
    M.update(panel_rulers("h2", z=False))
    by_anchor = defaultdict(list)
    for t in T:
        by_anchor[t["a"]].append(t)
    panel_set = {t["tri"] for t in T if t.get("panel")}
    hi = [t for t in T if max(t["votes_b"], t["n"] - t["votes_b"]) >= 0.8 * t["n"]]
    out = dict(
        n=len(T),
        n_high=len(hi),
        n_panel_subset=len(panel_set),
        acc={},
        acc_high={},
        acc_panel_subset={},
        acc_high_panel_subset={},
        spearman={},
        chance=0.5,
    )

    def acc(ts, m):
        ts = [t for t in ts if t["tri"] in m and m[t["tri"]] != 0]
        return float(np.mean([(m[t["tri"]] > 0) == (2 * t["votes_b"] > t["n"]) for t in ts])) if ts else float("nan")

    anchors = sorted(by_anchor)
    for name, m in M.items():
        out["acc"][name] = boot(anchors, lambda ks, m=m: acc([t for k in ks for t in by_anchor[k]], m))
        out["acc_high"][name] = [round(acc(hi, m), 4), len([t for t in hi if t["tri"] in m])]
        out["acc_panel_subset"][name] = round(acc([t for t in T if t["tri"] in panel_set], m), 4)
        out["acc_high_panel_subset"][name] = round(acc([t for t in hi if t["tri"] in panel_set], m), 4)
        ts = [t for t in T if t["tri"] in m]
        out["spearman"][name] = round(spearman([m[t["tri"]] for t in ts], [t["votes_b"] / t["n"] for t in ts]), 4)
    return out


def h2_gold():
    """The dataset's own test triplets (known answers): LUAR and each panel model (both orders)."""
    L = {t["tri"]: t for t in J("labels_h2.json") if t["golden"] and t["gold"]}
    S = J("scores_local_h2.json")
    out = dict(
        n=len(L),
        luar=round(
            float(
                np.mean([(S[k]["luar_ab_strict"] > S[k]["luar_ac_strict"]) == (t["gold"] == "b") for k, t in L.items()])
            ),
            4,
        ),
    )
    for m in PANEL:
        d = J(f"scores_panel_{m}.json")
        if not d:
            continue
        ks = [k for k in L if k in d["h2"]]
        right = np.mean(
            [
                ((d["h2"][k]["order1"] > 0.5) == (L[k]["gold"] == "b"))
                and ((d["h2"][k]["order2_pb"] > 0.5) == (L[k]["gold"] == "b"))
                for k in ks
            ]
        )
        out[f"panel_{m}"] = dict(
            right_both_orders=round(float(right), 4), n=len(ks), pass_bar=0.80, passed=bool(right >= 0.80)
        )
    return out


def h2_position():
    out = {}
    for m in PANEL:
        d = J(f"scores_panel_{m}.json")
        if d:
            v = list(d["h2"].values())
            # order1 > 0.5: picked the first-listed (B); order2_pb <= 0.5 means it picked the first-listed (real C)
            out[m] = dict(
                first_position_rate=round(
                    float(np.mean([x["order1"] > 0.5 for x in v] + [x["order2_pb"] <= 0.5 for x in v])), 4
                ),
                same_answer_both_orders=round(
                    float(np.mean([(x["order1"] > 0.5) == (x["order2_pb"] > 0.5) for x in v])), 4
                ),
                n=len(v),
            )
    return out


# ---------------------------------------------------------------- H3
def h3(split):
    L, S, P = J("labels_h3.json"), J("scores_local_h3.json"), J("scores_pref.json")
    T = [r for r in L if r["split"] == split]
    ids = {r["aid"] for r in T}
    M = {"detector": {r["aid"]: S[r["aid"]]["det"] for r in T}, "length": {r["aid"]: S[r["aid"]]["words"] for r in T}}
    if P:
        M["pref_neg"] = {r["aid"]: -P["h3"][r["aid"]] for r in T}
    M.update(panel_rulers("h3", keep=ids))
    byp = defaultdict(list)
    for r in T:
        byp[r["prompt"]].append(r)
    prompts = sorted(byp)
    out = dict(
        n=len(T),
        expert_majority_acc=round(float(np.mean([r["majority"] == r["truth"] for r in T])), 4),
        auroc_majority={},
        auroc_truth={},
        spearman_signed={},
        spearman_signed_within_human={},
        spearman_signed_within_ai={},
        auroc_by_length={},
    )
    for name, m in M.items():

        def au(ps, m=m, key="majority"):
            rr = [r for p in ps for r in byp[p] if r["aid"] in m]
            return auroc([r[key] for r in rr], [m[r["aid"]] for r in rr])

        out["auroc_majority"][name] = boot(prompts, au)
        out["auroc_truth"][name] = round(au(prompts, key="truth"), 4)
        rr = [r for r in T if r["aid"] in m]
        out["spearman_signed"][name] = round(spearman([m[r["aid"]] for r in rr], [r["signed"] for r in rr]), 4)
        for cls, key in ((0, "spearman_signed_within_human"), (1, "spearman_signed_within_ai")):
            # resample article groups (prompts), as for the AUROC: articles sharing a prompt are related
            def sp(ps, m=m, cls=cls):
                c = [r for p in ps for r in byp[p] if r["aid"] in m and r["truth"] == cls]
                return spearman([m[r["aid"]] for r in c], [r["signed"] for r in c])

            out[key][name] = boot(prompts, sp)
        w = np.array([r["words"] for r in rr])
        cut = np.quantile(w, [1 / 3, 2 / 3])
        out["auroc_by_length"][name] = [
            round(auroc([r["majority"] for r, k in zip(rr, s) if k], [m[r["aid"]] for r, k in zip(rr, s) if k]), 4)
            for s in (w <= cut[0], (w > cut[0]) & (w <= cut[1]), w > cut[1])
        ]
    return out


# ---------------------------------------------------------------- the user
def user():
    """The author's own blind ratings (n = 1). Not collected: replaced by an LLM stand-in at the user's request."""
    out = rated(RATINGS)
    if out["status"] == "pending":
        out.update(status="not_collected", reason=P6_NOT_TESTED)
    return out


def llm_standin():
    """An open LLM judge on the same blind pairs, standing in for the author's ratings (user decision 2026-11-22).
    Not human evidence: reported as its own arm, never merged with the author's."""
    out = rated(RES / "llm_standin_ratings.json")
    if out["status"] == "rated":
        j = J("standin_judge.json")
        out.update(model=j["model"], check=j["check"], choice_counts=j["choice_counts"], rule=j["rule"])
        # both orders are judged, so a swapped repeat is the same two prompts as its original: not test-retest
        # reliability (plan amendment 2). Consistency across the two orders, on the 150 originals, stands in for it.
        main = [v for k, v in j["pairs"].items() if not k.startswith("r")]
        out["test_retest"] = None
        out["both_orders_agree"] = [round(float(np.mean([v["choice"] != "tie" for v in main])), 4), len(main)]
    return out


def rated(path):
    idx = J("user_pairs_index.json")
    if not path.exists():
        return dict(status="pending", n_pairs=idx["n_pairs"])
    R = json.loads(path.read_text())
    if R.get("set_hash") != idx["hash"]:
        raise SystemExit("ratings are for a different pair set")
    S, P = J("scores_local_user.json"), J("scores_pref.json")
    M = {
        "luar": {k: v["luar"] for k, v in S.items()},
        "spec": {k: v["spec"] for k, v in S.items()},
        "detector_neg": {k: -v["det"] for k, v in S.items()},
        "tfidf_char": {k: v["tfidf_char"] for k, v in S.items()},
        "length": {k: v["words"] for k, v in S.items()},
    }
    if P:
        M["pref"] = P["user"]
    M.update(panel_rulers("user"))
    rat = R["ratings"]
    main = [p for p in idx["pairs"] if p["repeat_of"] is None and p["pair"] in rat]
    nontie = [p for p in main if rat[p["pair"]]["choice"] in ("a", "b")]
    flags = [rat[p["pair"]]["machine"] for p in main if rat[p["pair"]]["machine"] is not None]
    out = dict(
        status="rated",
        n_rated=len(main),
        n_ties=len(main) - len(nontie),
        # the LLM stand-in gives no "machine-written" flag (None)
        machine_flag_share=round(float(np.mean(flags)), 4) if flags else None,
        agreement={},
    )
    for name, m in M.items():
        ids = [
            p["pair"]
            for p in nontie
            if f"{p['pair']}:a" in m and f"{p['pair']}:b" in m and m[f"{p['pair']}:a"] != m[f"{p['pair']}:b"]
        ]
        out["agreement"][name] = boot(
            ids, lambda xs, m=m: float(np.mean([(m[f"{i}:a"] > m[f"{i}:b"]) == (rat[i]["choice"] == "a") for i in xs]))
        )
    reps = [p for p in idx["pairs"] if p["repeat_of"] and p["pair"] in rat and p["repeat_of"] in rat]
    flip = {"a": "b", "b": "a", "tie": "tie"}  # repeats are shown with sides swapped
    out["test_retest"] = (
        [
            round(float(np.mean([flip[rat[p["pair"]]["choice"]] == rat[p["repeat_of"]]["choice"] for p in reps])), 4),
            len(reps),
        ]
        if reps
        else None
    )
    wins = defaultdict(lambda: [0, 0])
    for p in nontie:
        w = p["arm_a"] if rat[p["pair"]]["choice"] == "a" else p["arm_b"]
        wins[p["arm_a"]][1] += 1
        wins[p["arm_b"]][1] += 1
        wins[w][0] += 1
    out["arm_win_rate"] = {a: v for a, v in wins.items()}
    return out


def score(s):
    H1, H2, H3 = s["h1"]["heldout"], s["h2"]["heldout"], s["h3"]["heldout"]
    out = []

    def add(pid, text, metric, value, interval, threshold, hit):
        out.append(
            dict(id=pid, prediction=text, metric=metric, value=value, interval=interval, threshold=threshold, hit=hit)
        )

    r = H1["rm_r"]["luar"]
    add(
        "P1",
        "LUAR agrees only weakly with people about their own voice",
        "H1 rm r, LUAR vs capture",
        r[0],
        r[1:],
        "0.10 <= r <= 0.40",
        bool(0.10 <= r[0] <= 0.40),
    )
    a = H1["pairwise_up"]["luar"]
    add(
        "P2",
        "When a person says their edit sounds more like them, LUAR usually agrees",
        "H1 pairwise accuracy (edit rated higher)",
        a[0],
        a[1:],
        ">= 0.70",
        bool(a[0] >= 0.70),
    )
    if "pref_neg" in H3["auroc_majority"] and "pref" in H1["pairwise_up"]:
        d = H3["auroc_majority"]["pref_neg"][0] - H1["pairwise_up"]["pref"][0]
        add(
            "P3",
            "The preference model tracks AI-ness more than voice",
            "H3 AUROC - H1 pairwise accuracy",
            round(d, 4),
            None,
            ">= +0.10",
            bool(d >= 0.10),
        )
    a = H2["acc_high"]["luar"]
    add(
        "P4",
        "LUAR's same style is readers' same style on fiction",
        "H2 accuracy, high-agreement triplets",
        a[0],
        None,
        ">= 0.65",
        bool(a[0] >= 0.65),
    )
    if "panel" in H1["rm_r"]:
        d = H1["rm_r"]["panel"][0] - H1["rm_r"]["luar"][0]
        add(
            "P5",
            "The LLM panel is no closer to people's own-voice ratings than LUAR",
            "H1 rm r, panel - LUAR",
            round(d, 4),
            None,
            "<= +0.05",
            bool(d <= 0.05),
        )
    u, llm = s["user"], s.get("llm_standin", {})
    p6 = "The user agrees with LUAR on most pairs"
    if u["status"] == "rated":
        a = u["agreement"]["luar"]
        add("P6", p6, "user agreement with LUAR, ties excluded", a[0], a[1:], ">= 0.60", bool(a[0] >= 0.60))
    else:
        # P6 is about a person, so LLM output can never make it a hit or a miss (plan amendment 3)
        add("P6", p6, "user agreement with LUAR, ties excluded", None, None, ">= 0.60", "not tested")
        out[-1]["reason"] = P6_NOT_TESTED
        if llm.get("status") == "rated":
            a = llm["agreement"]["luar"]
            out[-1]["exploratory"] = dict(
                label="exploratory LLM stand-in, not human evidence; not a score of P6",
                metric="LLM stand-in agreement with LUAR, ties excluded",
                value=a[0],
                interval=a[1:],
                line=0.60,
            )
    return out


def main():
    s = dict(
        h1=dict(heldout=h1("heldout"), dev=h1("dev"), checks=h1_checks()),
        h2=dict(heldout=h2("heldout"), dev=h2("dev"), gold=h2_gold(), position=h2_position()),
        h3=dict(heldout=h3("heldout"), dev=h3("dev")),
        user=user(),
        llm_standin=llm_standin(),
    )
    s["user_pairs"] = {k: v for k, v in J("user_pairs_index.json").items() if k != "pairs"}
    dump(RES / "summary.json", s)
    preds = score(s)
    dump(RES / "predictions.json", preds)
    for p in preds:
        print(p["id"], p["hit"], p["value"], p["interval"])


if __name__ == "__main__":
    main()
