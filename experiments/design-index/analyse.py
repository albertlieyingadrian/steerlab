"""
Every number in the note, from the index. Run: .venv/bin/python experiments/design-index/analyse.py [--split held]

  rulers   capture check, template check, render-noise floor (the render and label checks are hand labels: label.py)
  q1       our Wayback panel: yearly diversity per space, stable vs all-domain, convergence, slopes, breakpoint,
           shuffled-year control
  wbui     WaybackUI (public, top sites): yearly diversity at 1366 px, stable vs all
  q2       distance to the category centre: AI-host and generated pages vs human pages
  q3       the slop number and its known-answer test (templates vs human)
Writes results/<name>.json. --split dev is for tuning (results/dev/), held is the confirmatory set.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
from common import CACHE, CATEGORIES, RESULTS, split_of
from dataset import load
from measures import (
    SPACES,
    auroc,
    boot_weights,
    ci,
    distance_matrix,
    mean_pair_distance,
    slope,
    two_segment_breakpoint,
)

B = 1000
FIRST, LAST = (2014, 2016), (2024, 2026)  # even-year panel (run log, decision 8)


def save(name, obj, split):
    d = RESULTS if split == "held" else RESULTS / "dev"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {d / name}.json")


def spacing_stats(recs):
    X = np.stack([r["spacing"] for r in recs]).astype(np.float32)
    return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6


def one_per(recs, key):
    seen, out = set(), []
    for r in recs:
        k = key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ---------------------------------------------------------------- diversity over time
def yearly(recs, space, years, uniq, W, sstats):
    out = {}
    for y in years:
        ys = [r for r in recs if r["year"] == y]
        if len(ys) < 8:
            continue
        D = distance_matrix(ys, space, sstats)
        out[y] = mean_pair_distance(D, [r["domain"] for r in ys], uniq, W)
    return out


def window(div, yrs):
    arr = [div[y] for y in yrs if y in div]
    return np.mean(arr, 0) if arr else None


def summarise(div, first=FIRST, last=LAST):
    years = sorted(div)
    M = np.stack([div[y] for y in years])  # years x (B+1)
    norm = M / M.mean(0, keepdims=True)
    a, b = window(div, first), window(div, last)
    conv = (a - b) / a if a is not None and b is not None else None
    res = {
        "years": years,
        "div": M[:, 0].tolist(),
        "div_ci": [ci(M[i, 1:]) for i in range(len(years))],
        "norm": norm[:, 0].tolist(),
        "norm_ci": [ci(norm[i, 1:]) for i in range(len(years))],
    }
    if conv is not None:
        res["convergence"] = float(conv[0])
        res["convergence_ci"] = ci(conv[1:])
        res["_conv_boot"] = conv
    yrs = np.array(years)
    early = [i for i, y in enumerate(years) if y <= 2022]
    late = [i for i, y in enumerate(years) if y >= 2022]
    if len(early) >= 3 and len(late) >= 3:
        sl = np.array(
            [
                slope(yrs[late], norm[late, b]) - slope(yrs[early], norm[early, b])
                for b in range(norm.shape[1])
            ]
        )
        res["slope_early"] = slope(yrs[early], norm[early, 0])
        res["slope_late"] = slope(yrs[late], norm[late, 0])
        res["slope_diff"] = float(sl[0])
        res["slope_diff_ci"] = ci(sl[1:])
        cands = [y for y in years if years[0] + 2 <= y <= years[-1] - 2]
        bps = [
            two_segment_breakpoint(yrs, norm[:, b], cands)
            for b in range(0, norm.shape[1], 5)
        ]
        res["breakpoint"] = int(bps[0])
        vals, counts = np.unique(bps[1:], return_counts=True)
        res["breakpoint_dist"] = {int(v): int(c) for v, c in zip(vals, counts)}
    return res


def shuffled_year_null(recs, space, sstats, n_perm=200, seed=0):
    """Permute each domain's year labels among its own pages; the convergence that survives is the null."""
    rng = np.random.default_rng(seed)
    by_dom = defaultdict(list)
    for i, r in enumerate(recs):
        by_dom[r["domain"]].append(i)
    uniq, W = boot_weights([r["domain"] for r in recs], 0)
    D = distance_matrix(
        recs, space, sstats
    )  # computed once; a permutation only moves year labels
    doms = [r["domain"] for r in recs]
    out = []
    for _ in range(n_perm):
        year_of = np.empty(len(recs), int)
        for ii in by_dom.values():
            ys = [recs[i]["year"] for i in ii]
            rng.shuffle(ys)
            year_of[ii] = ys
        div = {}
        for y in sorted(set(year_of)):
            ii = np.where(year_of == y)[0]
            if len(ii) < 8:
                continue
            div[y] = mean_pair_distance(
                D[np.ix_(ii, ii)], [doms[i] for i in ii], uniq, W
            )
        a, b = window(div, FIRST), window(div, LAST)
        out.append(float(((a - b) / a)[0]))
    return out


def q1(split, loose=False):
    """loose=True keeps captures the filter dropped as unstyled or with failed assets (the render-check sensitivity test)."""
    everything = load(apply_filter=not loose)
    if loose:
        everything = [
            r
            for r in everything
            if r["keep"] or r["why"] in ("unstyled", "assets_failed")
        ]
    recs = [r for r in everything if r["source"] == "wayback" and r["split"] == split]
    recs = one_per(recs, lambda r: (r["domain"], r["year"]))
    sst = spacing_stats(
        [
            r
            for r in everything
            if r["split"] == "dev" and r["source"] in ("wayback", "human")
        ]
    )
    years = list(range(2014, 2027))
    stable = [r for r in recs if r["stable"]]
    yrs_of = defaultdict(set)
    for r in stable:
        yrs_of[r["domain"]].add(r["year"])
    both = {
        d for d, ys in yrs_of.items() if ys & set(FIRST) and ys & set(LAST)
    }  # seen in both windows
    sets = {
        "stable": [r for r in stable if r["domain"] in both],
        "all": [r for r in recs if r["all_sample"]],
    }
    out = {
        "n_pages": {k: len(v) for k, v in sets.items()},
        "n_domains": {k: len({r["domain"] for r in v}) for k, v in sets.items()},
        "n_per_year": {
            k: {y: sum(r["year"] == y for r in v) for y in years}
            for k, v in sets.items()
        },
    }
    boots = {}
    for name, rs in sets.items():
        uniq, W = boot_weights([r["domain"] for r in rs], B, seed=1)
        for sp in SPACES:
            res = summarise(yearly(rs, sp, years, uniq, W, sst))
            boots[(name, sp)] = res.pop("_conv_boot", None)
            out[f"{name}/{sp}"] = res
            print(
                name,
                sp,
                "conv",
                round(res.get("convergence", float("nan")), 3),
                res.get("convergence_ci"),
            )
    for sp in SPACES:
        a, b = boots[("stable", sp)], boots[("all", sp)]
        if a is not None and b is not None:
            r = a / b
            out[f"ratio_stable_all/{sp}"] = {"ratio": float(r[0]), "ci": ci(r[1:])}
    if not loose:
        out["shuffled_year_null/siglip/stable"] = shuffled_year_null(
            sets["stable"], "siglip", sst
        )
    out["broken_by_year"] = broken_by_year(split)
    save("q1_ours_loose" if loose else "q1_ours", out, split)


def q1loose(split):
    q1(split, loose=True)


def broken_by_year(split):
    """Share of panel captures the filter drops, by year and reason."""
    rs = [
        r
        for r in load(apply_filter=False)
        if r["source"] == "wayback"
        and r["split"] == split
        and (r["stable"] or r["all_sample"])
    ]
    out = {}
    for y in sorted({r["year"] for r in rs}):
        ys = [r for r in rs if r["year"] == y]
        out[y] = {
            "n": len(ys),
            "dropped": sum(not r["keep"] for r in ys),
            "why": {
                w: sum(r["why"] == w for r in ys)
                for w in sorted({r["why"] for r in ys})
                if w != "ok"
            },
        }
    return out


# ---------------------------------------------------------------- WaybackUI
def wbui(split):
    """WaybackUI holds two cohorts that never overlap: sites captured every year 2014-2020 at 1366 x 768, and sites
    captured 2021-2024 at 1920 x 1080. Each is analysed alone (run log, decision 13); they are never pooled."""
    rows = pickle.loads((CACHE / "feats" / "waybackui.pkl").read_bytes())
    out = {}
    for res_name, seg in (
        ("1366-768", range(2014, 2021)),
        ("1920-1080", range(2021, 2025)),
    ):
        seg = list(seg)
        rs = [
            r
            for r in rows
            if r["res"] == res_name and r["year"] in seg and r["n_boxes"] > 0
        ]
        for r in rs:
            r["domain"] = (
                r["base_url"].lower().rstrip("/").split("//")[-1].removeprefix("www.")
            )
        rs = one_per(
            sorted(rs, key=lambda r: r["ts"]), lambda r: (r["domain"], r["year"])
        )
        rs = [r for r in rs if split_of(r["domain"]) == split]
        per_dom = defaultdict(set)
        for r in rs:
            per_dom[r["domain"]].add(r["year"])
        rng = np.random.default_rng(0)
        stable = sorted(d for d, ys in per_dom.items() if set(seg) <= ys)
        if len(stable) > 1500:  # a fixed random draw, for memory
            stable = list(rng.choice(stable, 1500, replace=False))
        doms = sorted(per_dom)
        keep = set(rng.choice(doms, min(1500, len(doms)), replace=False))
        stable = set(stable)
        sets = {
            "stable": [r for r in rs if r["domain"] in stable],
            "all": [r for r in rs if r["domain"] in keep],
        }
        out[res_name] = {
            "years": seg,
            "n_domains": {k: len({r["domain"] for r in v}) for k, v in sets.items()},
            "n_per_year": {
                k: {y: sum(r["year"] == y for r in v) for y in seg}
                for k, v in sets.items()
            },
        }
        print(res_name, out[res_name]["n_domains"])
        for name, s in sets.items():
            uniq, W = boot_weights([r["domain"] for r in s], B, seed=2)
            for sp in ("siglip", "uiclip", "layout", "palette"):
                res = summarise(
                    yearly(s, sp, seg, uniq, W, None), tuple(seg[:2]), tuple(seg[-2:])
                )
                res.pop("_conv_boot", None)
                out[res_name][f"{name}/{sp}"] = res
                print(
                    res_name,
                    name,
                    sp,
                    round(res.get("convergence", float("nan")), 3),
                    res.get("convergence_ci"),
                )
    save("wbui", out, split)


# ---------------------------------------------------------------- category centres
def _unit(M):
    M = np.asarray(M, np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n == 0, 1.0, n)


def q2(split, space="siglip", reps=500):
    allr = load()
    first_cap = {}
    if (CACHE / "cdx.jsonl").exists():
        for line in (CACHE / "cdx.jsonl").read_text().splitlines():
            c = json.loads(line)
            # looked up and never archived counts as new (9999); domains never looked up are left out of human_new
            first_cap[c["domain"]] = int(min(c["captures"])) if c["captures"] else 9999
    live = [
        r
        for r in allr
        if r["source"] in ("human", "ai_host", "generated", "curated")
        and r["split"] == split
    ]
    hum = [r for r in live if r["source"] == "human"]
    rng = np.random.default_rng(3)
    out = {
        "space": space,
        "n": {
            s: sum(r["source"] == s for r in live)
            for s in ("human", "ai_host", "generated")
        },
    }
    groups = {
        "ai_host": lambda r: r["source"] == "ai_host",
        "generated": lambda r: r["source"] == "generated",
        "curated": lambda r: r["source"] == "curated",
        "human_new": lambda r: (
            r["source"] == "human"
            and r["domain"] in first_cap
            and first_cap[r["domain"]] >= 2023
        ),
    }
    ratios = {g: [] for g in groups}  # resampled splits: the interval
    plain = {
        g: [] for g in groups
    }  # splits without resampling: the point estimate (their median)
    per_cat = {g: {} for g in groups}
    for it in range(2 * reps):
        rep = it % 2  # 0 = plain split, 1 = split + resampling
        dists = {"human": [], **{g: [] for g in groups}}
        for cat in CATEGORIES:
            h = [r for r in hum if r["category"] == cat]
            if len(h) < 10:
                continue
            perm = rng.permutation(len(h))
            cen_ix, sc_ix = perm[: len(h) // 2], perm[len(h) // 2 :]
            centre = _unit(np.stack([h[i][space] for i in cen_ix])).mean(0)
            centre /= np.linalg.norm(centre)
            cen_set = {id(h[i]) for i in cen_ix}
            scored_h = [h[i] for i in sc_ix]
            if rep > 0:
                scored_h = [
                    scored_h[i] for i in rng.integers(0, len(scored_h), len(scored_h))
                ]
            dh = 1 - _unit(np.stack([r[space] for r in scored_h])) @ centre
            med = np.median(dh)
            dists["human"] += list(dh / med)
            for g, f in groups.items():
                xs = [
                    r
                    for r in live
                    if r["category"] == cat and f(r) and id(r) not in cen_set
                ]
                if not xs:
                    continue
                if rep > 0:
                    xs = [xs[i] for i in rng.integers(0, len(xs), len(xs))]
                d = (1 - _unit(np.stack([r[space] for r in xs])) @ centre) / med
                dists[g] += list(d)
                if rep == 0:
                    per_cat[g].setdefault(cat, []).append(float(np.median(d)))
        for g in groups:
            if dists[g]:
                (plain if rep == 0 else ratios)[g].append(
                    np.median(dists[g]) / np.median(dists["human"])
                )
    for g in groups:  # noqa: PLC0206
        if ratios[g]:
            per_cat[g] = {c: float(np.median(v)) for c, v in per_cat[g].items()}
            ratios[g] = [float(np.median(plain[g]))] + ratios[g]
            out[g] = {
                "ratio": float(ratios[g][0]),
                "ci": ci(np.array(ratios[g][1:])),
                "n": sum(groups[g](r) for r in live),
                "per_category": per_cat[g],
            }
            print(space, g, round(ratios[g][0], 3), out[g]["ci"])
    save(f"q2_{space}", out, split)


# ---------------------------------------------------------------- slop number
def slop_scores(pool, space="siglip", k=10):
    """centrality (vs the human centre of the page's category, leaving the page out) and repetition (mean similarity
    to its k nearest pages in other categories, other domains), z-scored on human pages, averaged. Centres, medians
    and z-statistics come from dev human pages only, so held-out numbers use nothing fitted on held-out pages."""
    X = _unit(np.stack([r[space] for r in pool]))
    cats = np.array([r["category"] for r in pool])
    doms = np.array([r["domain"] for r in pool])
    is_h = np.array([r["source"] == "human" for r in pool])
    is_h = is_h & np.array(
        [r["split"] == "dev" for r in pool]
    )  # centres, medians and z from dev humans only
    S = X @ X.T
    cent = np.zeros(len(pool))
    for c in set(cats):
        hi = np.where(is_h & (cats == c))[0]
        tot = X[hi].sum(0)
        ci_ = np.where(cats == c)[0]
        for i in ci_:
            v = tot - X[i] if is_h[i] else tot
            cent[i] = 1 - X[i] @ (v / np.linalg.norm(v))
        med = np.median(cent[hi])
        cent[ci_] = 1 - cent[ci_] / med
    rep = np.zeros(len(pool))
    for i in range(len(pool)):
        mask = (cats != cats[i]) & (doms != doms[i])
        rep[i] = np.sort(S[i, mask])[-k:].mean()

    def z(v):
        return (v - v[is_h].mean()) / v[is_h].std()

    return z(cent), z(rep), 0.5 * (z(cent) + z(rep))


def q3(split):
    allr = load()
    pool = [
        r
        for r in allr
        if r["source"] in ("human", "ai_host", "generated", "template", "curated")
    ]
    out = {}
    for space in ("siglip", "layout", "uiclip", "dom"):
        c, rp, s = slop_scores(pool, space)
        for r, a, b, v in zip(pool, c, rp, s):
            r[f"slop_{space}"], r[f"cent_{space}"], r[f"rep_{space}"] = (
                float(v),
                float(a),
                float(b),
            )
        hum = [r for r in pool if r["source"] == "human" and r["split"] == split]
        res = {}
        groups = {
            "template": lambda r: (
                r["source"] == "template" and r.get("template") in ("A", "B", "C", "D")
            ),
            "template_fresh": lambda r: (
                r["source"] == "template" and r.get("template") in ("E", "F")
            ),
            "ai_host": lambda r: r["source"] == "ai_host",
            "generated": lambda r: r["source"] == "generated",
            "curated": lambda r: r["source"] == "curated",
        }
        for name, f in groups.items():
            xs = [r for r in pool if f(r) and r["split"] == split]
            if not xs:
                continue
            res[name] = {
                k: auroc(
                    [r[f"{k}_{space}"] for r in xs], [r[f"{k}_{space}"] for r in hum]
                )
                for k in ("slop", "cent", "rep")
            }
            res[name]["n"] = len(xs)
            res[name]["median_slop"] = float(
                np.median([r[f"slop_{space}"] for r in xs])
            )
        res["human_median_slop"] = float(np.median([r[f"slop_{space}"] for r in hum]))
        res["n_human"] = len(hum)
        out[space] = res
        print(space, json.dumps(res)[:400])
    save("q3", out, split)
    return pool


# ---------------------------------------------------------------- house style (exploratory, not pre-registered)
def house(split, reps=1000):
    """For each page, its nearest page from a different category (and domain): how often is that an AI-builder page?
    Compared with the AI-builder share of the pool. Exploratory: added after the Q3 results, reported as such."""
    rs = [r for r in load() if r["source"] in ("human", "ai_host", "curated")]
    rng = np.random.default_rng(4)
    cats = np.array([r["category"] for r in rs])
    src = np.array([r["source"] for r in rs])
    host = np.array(
        [
            ".".join(r["domain"].split(".")[-2:]) if r["source"] == "ai_host" else ""
            for r in rs
        ]
    )
    held = np.array([r["split"] == split for r in rs])
    from collections import Counter

    main_host = Counter(host[src == "ai_host"]).most_common(1)[0][
        0
    ]  # the builder with most sites (83%)
    out = {
        "ai_share_of_pool": float((src == "ai_host").mean()),
        "n": {s: int((held & (src == s)).sum()) for s in ("human", "ai_host")},
    }
    for sp in ("siglip", "uiclip", "layout", "dom"):
        X = _unit(np.stack([np.asarray(r[sp], np.float32) for r in rs]))
        S = X @ X.T
        nn = np.empty(len(rs), int)
        doms = np.array([r["domain"] for r in rs])
        chance = np.empty(len(rs))
        for i in range(len(rs)):
            m = (cats != cats[i]) & (doms != doms[i])
            idx = np.where(m)[0]
            nn[i] = idx[np.argmax(S[i, idx])]
            chance[i] = (
                src[idx] == "ai_host"
            ).mean()  # the rate if source labels were shuffled
        res = {}
        for s, f in (
            ("ai_host", src == "ai_host"),
            ("human", src == "human"),
            ("curated", src == "curated"),
            ("ai_host_main", (src == "ai_host") & (host == main_host)),
            ("ai_host_other", (src == "ai_host") & (host != main_host)),
        ):
            ii = np.where(held & f)[0]
            hit = (src[nn[ii]] == "ai_host").astype(float)
            same_host = (host[nn[ii]] == host[ii]).astype(float)
            boot = [
                hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(reps)
            ]
            res[s] = {
                "rate": float(hit.mean()),
                "ci": ci(np.array(boot)),
                "n": len(ii),
                "chance": float(chance[ii].mean()),
                "same_host_rate": float(same_host.mean())
                if s.startswith("ai")
                else None,
            }
        out[sp] = res
        print(
            sp,
            {
                k: (round(v["rate"], 3), [round(x, 3) for x in v["ci"]], v["n"])
                for k, v in res.items()
            },
        )
    save("house", out, split)


# ---------------------------------------------------------------- curated library (exploratory, post hoc)
def curated(split, reps=1000):
    """Do curated high-quality human designs sit closer to the AI-builder cluster than ordinary human sites?
    Per page: cosine similarity to the centroid of AI-builder pages from OTHER industries. Exploratory, post hoc."""
    rs = [r for r in load() if r["source"] in ("human", "ai_host", "curated")]
    rng = np.random.default_rng(6)
    out = {}
    for sp in ("dom", "layout", "siglip", "uiclip"):
        X = _unit(np.stack([np.asarray(r[sp], np.float32) for r in rs]))
        cats = np.array([r["category"] for r in rs])
        is_ai = np.array([r["source"] == "ai_host" for r in rs])
        sim = np.zeros(len(rs))
        for c in set(cats):
            cen = X[is_ai & (cats != c)].mean(0)
            cen /= np.linalg.norm(cen)
            sim[cats == c] = X[cats == c] @ cen
        res = {}
        for g in ("curated", "human", "ai_host"):
            v = np.array(
                [
                    sim[i]
                    for i, r in enumerate(rs)
                    if r["source"] == g and r["split"] == split
                ]
            )
            boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)]
            res[g] = {
                "mean_sim_to_ai_cluster": float(v.mean()),
                "ci": ci(np.array(boot)),
                "n": len(v),
            }
        cur = [
            sim[i]
            for i, r in enumerate(rs)
            if r["source"] == "curated" and r["split"] == split
        ]
        hum = [
            sim[i]
            for i, r in enumerate(rs)
            if r["source"] == "human" and r["split"] == split
        ]
        res["auroc_curated_vs_human"] = auroc(np.array(cur), np.array(hum))
        out[sp] = res
        print(
            sp,
            {
                k: (
                    round(v["mean_sim_to_ai_cluster"], 3),
                    [round(x, 3) for x in v["ci"]],
                )
                if isinstance(v, dict)
                else round(v, 3)
                for k, v in res.items()
            },
        )
    save("curated", out, split)


# ---------------------------------------------------------------- rulers
def rulers(split):
    allr = load()
    out = {}
    live = {
        r["domain"]: r for r in allr if r["source"] == "human" and r["split"] == split
    }
    wb = {
        r["domain"]: r
        for r in allr
        if r["source"] == "wayback" and r["year"] == 2026 and r["split"] == split
    }
    both = sorted(set(live) & set(wb))
    out["capture_n_domains"] = len(both)
    for sp in ("siglip", "layout", "uiclip", "palette", "type", "dom"):
        if len(both) < 10:
            break
        D = distance_matrix([live[d] for d in both] + [wb[d] for d in both], sp)
        n = len(both)
        cross = D[:n, n:]
        out[f"capture/{sp}"] = auroc(
            -np.diag(cross), -cross[~np.eye(n, dtype=bool)]
        )  # same domain should be closer
    rep = {
        r["domain"]: r
        for r in allr
        if r["source"] == "human_rep" and r["split"] == split
    }
    both = sorted(set(live) & set(rep))
    out["noise_n"] = len(both)
    for sp in ("siglip", "layout", "palette"):
        if len(both) < 10:
            break
        D = distance_matrix([live[d] for d in both] + [rep[d] for d in both], sp)
        n = len(both)
        cross = D[:n, n:]
        off = cross[~np.eye(n, dtype=bool)]
        out[f"noise/{sp}"] = {
            "same_median": float(np.median(np.diag(cross))),
            "diff_median": float(np.median(off)),
            "auroc": auroc(-np.diag(cross), -off),
        }
    tpl = [r for r in allr if r["source"] == "template" and r["split"] == split]
    hum = [r for r in allr if r["source"] == "human" and r["split"] == split][:400]
    for sp in ("siglip", "layout", "type", "palette", "dom", "uiclip"):
        if not tpl:
            break
        D = distance_matrix(tpl + hum, sp)
        n = len(tpl)
        same = [
            D[i, j]
            for i in range(n)
            for j in range(i + 1, n)
            if tpl[i]["template"] == tpl[j]["template"]
        ]
        diff = D[n:, n:][np.triu_indices(len(hum), 1)]
        out[f"template/{sp}"] = auroc(-np.array(same), -diff)
    print(json.dumps(out, indent=1))
    save("rulers", out, split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "what",
        nargs="+",
        choices=["rulers", "q1", "q1loose", "wbui", "q2", "q3", "house", "curated"],
    )
    ap.add_argument("--split", default="dev", choices=["dev", "held"])
    a = ap.parse_args()
    for w in a.what:
        if w == "q2":
            for sp in ("siglip", "layout", "uiclip"):
                q2(a.split, sp)
        else:
            globals()[w](a.split)
