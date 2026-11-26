"""Draw the note's figures from results/ and print every number the note quotes (no GPU, no API call).

    python experiments/looped-lens/make_figures.py      (run cv.py first: it writes results/cv.json)

Writes report/figures/: visits.png (how often each raw lens names the intermediate at each of the 48 visits),
trained.png (the same, and wrong votes, for the trained translators against the raw read they start from), cv.png (the
cross-validated test pass rate of each lens set against the starter), hero.jpg (the cover), the research card
../shared/thumbs/looped-lens.jpg, report/data.js (the explorer's per-visit rates and example readouts) and
results/summary.json (every number printed here).
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
INK, SOFT, ACCENT, CONTROL, RULE, PAPER = "#1d1a17", "#6f675f", "#c9561c", "#2f6db0", "#d9d2c8", "#faf7f2"
GREY = "#8a8178"
DEEP = "#7a3b12"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
PASSES = [(0, 6, "prelude"), (6, 18, "core, pass 1"), (18, 30, "pass 2"), (30, 42, "pass 3"), (42, 48, "coda")]
SET_LABEL = {
    "starter": "Starter (logit lens, fixed visits)",
    "raw": "Raw reads only",
    "raw+tuned": "Raw + tuned lens",
    "raw+future": "Raw + future lens",
    "raw+concept": "Raw + concept lens",
    "raw+future+concept": "Raw + future + concept",
    "all (v1 set)": "Raw + tuned + future (v1's set)",
    "all+concept (v2 set)": "Raw + tuned + future + concept (v2's set)",
    "raw+wsp": "Raw + weight-space designs",
    "all+concept+wsp": "Everything + weight-space designs",
}
V2 = [
    ("mlp_in", 8),
    ("act_in", 12),
    ("concept_mlp_in", 27),
    ("concept_mlp_in", 33),
    ("future_mlp_in", 34),
    ("act_in", 36),
    ("mlp_in", 42),
    ("act_in", 44),
    ("future_act_in", 45),
    ("act_in", 46),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def load() -> dict:
    with gzip.open(RES / "votes.json.gz", "rt") as f:
        votes = json.load(f)
    qs = {}
    for line in open(RES / "questions.jsonl"):
        q = json.loads(line)
        qs[q["id"]] = q
    qids = votes["qids"]
    gold = {q: qs[q]["options"].index(qs[q]["intermediate"]) for q in qids}
    rates: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for key in votes["keys"]:
        lens, v = key.rsplit("|", 1)
        col = votes["fuzzy"][key]
        g = sum(c != "." and int(c) == gold[q] for q, c in zip(qids, col))
        w = sum(c != "." and int(c) != gold[q] for q, c in zip(qids, col))
        rates[lens][int(v)] = (g / len(qids), w / len(qids))
    cv = json.loads((RES / "cv.json").read_text())
    ndif = json.loads((RES / "ndif.json").read_text())
    return {"qs": qs, "qids": qids, "rates": rates, "cv": cv, "ndif": ndif}


def cv_stats(cv: dict, a: str, b: str) -> dict:
    """Mean test pass rate of set a over folds, and its paired difference with set b, with one fold's 95% CI."""
    means, diffs, half = [], [], []
    for fold in cv["folds"]:
        ra, rb = fold["res"][a], fold["res"][b]
        n = len(ra)
        means.append(sum(ra) / n)
        d = [x - y for x, y in zip(ra, rb)]
        diffs.append(sum(d) / n)
        half.append(1.96 * statistics.pstdev(d) / math.sqrt(n))
    return {
        "mean": sum(means) / len(means),
        "fold_min": min(means),
        "fold_max": max(means),
        "diff": sum(diffs) / len(diffs),
        "diff_min": min(diffs),
        "diff_max": max(diffs),
        "half": sum(half) / len(half),
        "better": sum(d > 0 for d in diffs),
        "worse": sum(d < 0 for d in diffs),
        "folds": len(diffs),
        "n_test": len(cv["folds"][0]["test"]),
    }


def checks(D: dict) -> dict:
    qs, qids, rates, cv, ndif = D["qs"], D["qids"], D["rates"], D["cv"], D["ndif"]
    out: dict = {"lens": {}}
    print("== per-lens best gold-vote rate over visits (all 867, fuzzy)")
    for lens in sorted(rates):
        if lens == "final":
            continue
        best = max(rates[lens].items(), key=lambda kv: kv[1][0])
        mid = [rates[lens][v][0] for v in range(6, 42) if v in rates[lens]]
        out["lens"][lens] = {
            "best_visit": best[0],
            "gold": best[1][0],
            "wrong": best[1][1],
            "core_mean_gold": sum(mid) / max(len(mid), 1),
        }
        print(
            f"  {lens:28s} best visit {best[0]:2d}: gold {best[1][0]:.3f} wrong {best[1][1]:.3f}; "
            f"visits read {len(rates[lens])}; mean gold over core visits 6-41: {sum(mid) / max(len(mid), 1):.3f}"
        )
    if "final" in rates:
        print(f"  final output (visit 48): gold {rates['final'][48][0]:.3f}")

    print("== fixed entries (fuzzy matcher), Wilson 95% CIs")
    fixed = cv["fixed"]
    groups = {
        "all 867": qids,
        "round 2 dev (215)": [q for q in qids if qs[q]["round"] == 2 and qs[q]["split"] == "dev"],
        "round 2 held-out (112)": [q for q in qids if qs[q]["round"] == 2 and qs[q]["split"] == "heldout"],
        "round 3 dev (349)": [q for q in qids if qs[q]["round"] == 3 and qs[q]["split"] == "dev"],
        "round 3 held-out (191)": [q for q in qids if qs[q]["round"] == 3 and qs[q]["split"] == "heldout"],
    }
    out["fixed"] = {}
    for g, ids in groups.items():
        row = {}
        for name in ("starter", "v1", "v2"):
            k = sum(fixed[name][q] for q in ids)
            row[name] = [k, len(ids)]
            print(f"  {g:24s} {name:8s} {fmt(k, len(ids))}")
        out["fixed"][g] = row

    print("== per family, all 867 (starter / v1 / v2)")
    fams = defaultdict(list)
    for q in qids:
        fams[qs[q]["family"]].append(q)
    out["family"] = {}
    for f, ids in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        r = {n: sum(fixed[n][q] for q in ids) for n in ("starter", "v1", "v2")}
        out["family"][f] = [len(ids), r]
        print(f"  {f:14s} n={len(ids):3d}  " + "  ".join(f"{n} {r[n]}/{len(ids)} = {r[n] / len(ids):.2f}" for n in r))

    print(f"== cross-validation: {len(cv['folds'])} folds, test fold {len(cv['folds'][0]['test'])} questions each")
    out["cv"] = {}
    for name in SET_LABEL:
        s = cv_stats(cv, name, "starter")
        out["cv"][name] = s
        print(
            f"  {name:22s} test {s['mean']:.3f} (folds {s['fold_min']:.3f}-{s['fold_max']:.3f})  vs starter "
            f"{100 * s['diff']:+.1f} pts (one fold's 95% CI ±{100 * s['half']:.1f}; fold range "
            f"{100 * s['diff_min']:+.1f} to {100 * s['diff_max']:+.1f}; better in {s['better']}/{s['folds']})"
        )
    pairs = [
        ("all+concept (v2 set)", "all (v1 set)"),
        ("raw+future+concept", "raw+future"),
        ("raw+future", "raw"),
        ("raw+tuned", "raw"),
        ("raw+concept", "raw"),
        ("raw+wsp", "raw"),
        ("all+concept+wsp", "all+concept (v2 set)"),
        ("raw", "starter"),
    ]
    for a, b in pairs:
        s = cv_stats(cv, a, b)
        out["cv"][f"{a} - {b}"] = s
        print(
            f"  {a} minus {b}: {100 * s['diff']:+.1f} pts (one fold's 95% CI ±{100 * s['half']:.1f}; fold range "
            f"{100 * s['diff_min']:+.1f} to {100 * s['diff_max']:+.1f}; better {s['better']}, worse {s['worse']} "
            f"of {s['folds']})"
        )

    print("== NDIF (hosted model) against the offline copy, same questions")
    out["ndif"] = {}
    for name, run in ndif.items():
        rows = run["questions"]
        n = len(rows)
        kn = sum(r["ndif_pass"] for r in rows)
        ko = sum(r["offline_pass"] for r in rows)
        agree = sum(r["ndif_pass"] == r["offline_pass"] for r in rows)
        only_n = sum(r["ndif_pass"] and not r["offline_pass"] for r in rows)
        only_o = sum(r["offline_pass"] and not r["ndif_pass"] for r in rows)
        t1 = sum(r["top1_agree"] for r in rows)
        reads = sum(r["readouts"] for r in rows)
        out["ndif"][name] = {
            "n": n,
            "ndif": kn,
            "offline": ko,
            "agree": agree,
            "only_ndif": only_n,
            "only_offline": only_o,
            "top1": t1,
            "reads": reads,
        }
        print(
            f"  {name}: NDIF {fmt(kn, n)}; offline {fmt(ko, n)}; same verdict on {agree}/{n} "
            f"(NDIF only {only_n}, offline only {only_o}); top-1 token agrees on {t1}/{reads} = {t1 / reads:.3f} reads"
        )

    print("== translator fits (held-out loss before -> after)")
    tl = json.loads((RES / "training" / "tuned_v1_log.json").read_text())["log"]
    print(
        f"  tuned lens: mean KL to the final layer {tl[0]['mean_kl']} -> {tl[-1]['mean_kl']} "
        f"({tl[-1]['step'] + 1} steps)"
    )
    for name in ("future_v1_log.json", "future_v2_log.json", "concept_v1_log.json"):
        log = json.loads((RES / "training" / name).read_text())["log"]
        b = [r["val_ce_before"] for r in log]
        a = [r["val_ce_after"] for r in log]
        print(
            f"  {name}: {len(log)} translators; CE before {min(b):.1f}-{max(b):.1f}, after {min(a):.2f}-{max(a):.2f} "
            f"(median {statistics.median(a):.2f})"
        )
        for r in log:
            if r["key"] in ("mlp_in.34", "mlp_in.44", "act_in.45", "mlp_in.27", "mlp_in.33"):
                print(f"     {r['key']}: {r['val_ce_before']} -> {r['val_ce_after']}")
    card = json.loads((RES / "judge_round2.json").read_text())
    print(
        f"== round-2 judge check: {card['run']['judge_model']} on the round-2 entry, "
        f"{card['overall']['passed']}/{card['overall']['n']} = {card['overall']['score']}, "
        f"cost ${card['run']['judge_cost_usd']}"
    )
    jc = json.loads((RES / "judge_check.json").read_text())["questions"]
    n = len(jc)
    jp, fp, ep = (sum(r[k] for r in jc) for k in ("judge_pass", "fuzzy_pass", "exact_pass"))
    agree = sum(r["judge_pass"] == r["fuzzy_pass"] for r in jc)
    la, lt = sum(r["layer_votes_agree"] for r in jc), sum(r["layers"] for r in jc)
    out["judge"] = {
        "n": n,
        "judge": jp,
        "fuzzy": fp,
        "exact": ep,
        "agree": agree,
        "judge_only": sum(r["judge_pass"] and not r["fuzzy_pass"] for r in jc),
        "fuzzy_only": sum(r["fuzzy_pass"] and not r["judge_pass"] for r in jc),
        "layer_agree": la,
        "layers": lt,
    }
    print(
        f"   same tokens: judge {jp}/{n}, fuzzy {fp}/{n}, exact {ep}/{n}; judge and fuzzy give the same verdict on "
        f"{agree}/{n} (judge only {out['judge']['judge_only']}, fuzzy only {out['judge']['fuzzy_only']}); "
        f"per-layer votes agree {la}/{lt} = {la / lt:.3f}"
    )
    return out


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def shade(ax) -> None:
    for i, (a, b, label) in enumerate(PASSES):
        if i % 2 == 1:
            ax.axvspan(a - 0.5, b - 0.5, color=RULE, alpha=0.35, lw=0)
        ax.text(
            (a + b - 1) / 2,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=SOFT,
        )


def fig_visits(D: dict) -> None:
    rates = D["rates"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    shade(ax)
    styles = [
        ("act_in", "Logit lens on the active block (the starter)", GREY, "-"),
        ("mix_in", "Logit lens on the output head's own mix", CONTROL, "--"),
        ("mlp_in", "Logit lens on the layer's MoE input", ACCENT, "-"),
    ]
    for lens, label, c, ls in styles:
        vs = sorted(rates[lens])
        ax.plot(vs, [rates[lens][v][0] for v in vs], color=c, ls=ls, lw=1.8, label=label)
        print(
            f"visits.png {lens}: " + " ".join(f"{v}:{rates[lens][v][0]:.2f}" for v in vs if rates[lens][v][0] >= 0.15)
        )
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Layer visit (0-47, in the order they run)")
    ax.set_ylabel("Share of 867 questions where the\nread votes for the intermediate")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "visits.png")


def fig_trained(D: dict) -> None:
    rates = D["rates"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharex=True)
    styles = [
        ("mlp_in", "Raw MoE input (no training)", GREY, "-"),
        ("tuned_mlp_in", "Tuned lens (next token)", CONTROL, "--"),
        ("future_mlp_in", "Future lens (first word of the reply)", ACCENT, "-"),
        ("concept_mlp_in", "Concept lens (words around the current word)", DEEP, ":"),
    ]
    for ax, idx, title in [(axes[0], 0, "Votes for the intermediate"), (axes[1], 1, "Votes for a wrong option")]:
        for lens, label, c, ls in styles:
            xs = list(range(20, 48))  # a visit the lens was not fitted at is a gap, not a line
            ys = [rates[lens][v][idx] if v in rates[lens] else float("nan") for v in xs]
            ax.plot(xs, ys, color=c, ls=ls, lw=1.7, marker="o", ms=2.5, label=label)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Layer visit")
        ax.set_ylim(0, 0.7 if idx == 0 else 0.3)
        ax.axvline(41.5, color=RULE, lw=1)
    axes[0].set_ylabel("Share of 867 questions")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    for lens, _, _, _ in styles:
        print(
            f"trained.png {lens}: "
            + " ".join(
                f"{v}:{rates[lens][v][0]:.2f}/{rates[lens][v][1]:.2f}"
                for v in sorted(rates[lens])
                if v in (24, 27, 33, 34, 40, 44, 45, 46)
            )
        )
    fig.tight_layout()
    save(fig, "trained.png")


def fig_cv(S: dict) -> None:
    names = list(SET_LABEL)[1:]
    names.sort(key=lambda n: S["cv"][n]["diff"])
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, n in enumerate(names):
        s = S["cv"][n]
        c = ACCENT if ("future" in n or "concept" in n or "all" in n) else GREY
        ax.plot([100 * (s["diff"] - s["half"]), 100 * (s["diff"] + s["half"])], [i, i], color=c, lw=2.2, alpha=0.5)
        ax.plot([100 * s["diff_min"], 100 * s["diff_max"]], [i, i], color=c, lw=0, marker="|", ms=9)
        ax.plot([100 * s["diff"]], [i], "o", color=c, ms=6)
        ax.text(
            100 * max(s["diff"] + s["half"], s["diff_max"]) + 0.4,
            i,
            f"{100 * s['diff']:+.1f}  ({s['better']}/{s['folds']} folds)",
            va="center",
            fontsize=7.5,
            color=SOFT,
        )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SET_LABEL[n] for n in names], fontsize=8)
    ax.set_xlabel("Cross-validated test pass rate minus the starter's, points (289 test questions per fold)")
    ax.set_xlim(-6, 14)
    save(fig, "cv.png")


def data_js(D: dict) -> None:
    rates = D["rates"]
    ex = json.loads((RES / "examples.json").read_text())
    lenses = ["act_in", "mix_in", "mlp_in", "tuned_mlp_in", "future_mlp_in", "concept_mlp_in"]
    out = {
        "rates": {
            lens: {str(v): [round(g, 3), round(w, 3)] for v, (g, w) in sorted(rates[lens].items())} for lens in lenses
        },
        "examples": [
            {
                "id": e["id"],
                "family": e["family"],
                "prompt": e["prompt"],
                "options": e["options"],
                "intermediate": e["intermediate"],
                "passes_v2": e["passes_v2"],
                "reads": {
                    lens: {v: [r["tokens"][:5], r["vote"]] for v, r in e["reads"][lens].items()} for lens in lenses
                },
            }
            for e in ex
        ],
        "v2": [[v, lens] for lens, v in V2],
    }
    text = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    (HERE / "report" / "data.js").write_text("window.LENS = " + text + ";\n")


def hero(D: dict) -> None:
    """Cover: six lenses (rows) by 48 visits (columns); each cell's ember strength is how often that read names the
    intermediate; an outlined cell was not read. Gaps mark the prelude, the three core passes and the coda."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from PIL import Image

    rates = D["rates"]
    lenses = ["act_in", "mix_in", "mlp_in", "tuned_mlp_in", "future_mlp_in", "concept_mlp_in"]
    cmap = LinearSegmentedColormap.from_list("e", [RULE, "#f2b48e", ACCENT, DEEP])
    gap = {6: 0.6, 18: 0.6, 30: 0.6, 42: 0.6}
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for r, lens in enumerate(lenses):
        x = 0.0
        y = len(lenses) - 1 - r
        for v in range(48):
            x += gap.get(v, 0)
            g = rates[lens].get(v, (None, None))[0]
            if g is None:
                ax.add_patch(plt.Rectangle((x + 0.1, y + 0.1), 0.8, 0.8, facecolor="none", edgecolor=RULE, lw=1))
            else:
                t = min(g / 0.6, 1.0)
                ax.add_patch(plt.Rectangle((x + 0.08, y + 0.08), 0.84, 0.84, facecolor=to_rgb(cmap(t)), lw=0))
            x += 1
    ax.set_xlim(-2, 54.4)
    ax.set_ylim(-2.4, len(lenses) + 2.4)
    ax.axis("off")
    FIG.mkdir(parents=True, exist_ok=True)
    tmp = FIG / "_hero.png"
    fig.savefig(tmp, dpi=100, facecolor=PAPER)
    plt.close(fig)
    Image.open(tmp).convert("RGB").save(FIG / "hero.jpg", quality=88)
    tmp.unlink()
    img = Image.open(FIG / "hero.jpg")  # the research page's card: 720x480, centre crop of the cover
    w, h = img.size
    cw = int(h * 1.5)
    thumb = img.crop(((w - cw) // 2, 0, (w + cw) // 2, h)).resize((720, 480), Image.LANCZOS)
    thumb.save(HERE.parent / "shared" / "thumbs" / "looped-lens.jpg", quality=84, optimize=True)


def main() -> None:
    D = load()
    S = checks(D)
    fig_visits(D)
    fig_trained(D)
    fig_cv(S)
    data_js(D)
    hero(D)
    (RES / "summary.json").write_text(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
