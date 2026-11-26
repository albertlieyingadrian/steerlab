"""Draw the note's figures from results/ and print the numbers each one shows (no paid call, no private data).

    python make_figures.py

Reads results/nodes.jsonl and results/summary.json (pilot 1) and results/pilot2/{nodes.jsonl, summary.json}. Writes
report/figures/: spread.png (every recipe's validation score in both pilots: pilot 1 tied, pilot 2 ranks), search.png
(best-so-far validation score per evaluation, both arms, pilot 2, with the registered test looks), grades.png (each
improver's frontier-relative grade), loophole.png (validation score against the share of humans flagged at the
recipe's own cutoff), hero.jpg and og.jpg (the cover: both arms' search trees), the research card
../shared/thumbs/slop-rsi.jpg, and report/data.js (counts for the cap explorer; scores and rates only, no text).
It also prints every check behind the note's headline numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
P1 = HERE / "results"
P2 = P1 / "pilot2"
FIG = HERE / "report" / "figures"
INK, SOFT, ACCENT, CONTROL, RULE, PAPER = "#1d1a17", "#6f675f", "#c9561c", "#2f6db0", "#d9d2c8", "#faf7f2"
GREY = "#8a8178"
ARM = {"l2": ("L2: fixed improver", CONTROL), "l3": ("L3: self-editing improver", ACCENT)}
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


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def slot(node: str) -> int:
    return int(node.split("_")[1])


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load() -> dict:
    n1 = [r for r in jsonl(P1 / "nodes.jsonl") if r.get("arm") in ("l2", "l3")]
    n2 = jsonl(P2 / "nodes.jsonl")
    s1 = json.loads((P1 / "summary.json").read_text())
    s2 = json.loads((P2 / "summary.json").read_text())
    return {"n1": n1, "n2": n2, "s1": s1, "s2": s2}


def arm_ok(D: dict) -> list[dict]:
    return [r for r in D["n2"] if r.get("arm") in ("l2", "l3") and r["status"] == "ok"]


def checks(D: dict) -> None:
    """Every headline number, recomputed from the saved results."""
    n1, n2, s1, s2 = D["n1"], D["n2"], D["s1"], D["s2"]
    ok1 = [r for r in n1 if r["status"] == "ok"]
    g1 = Counter(round(r["val"]["gain"], 6) for r in ok1)
    top1 = max(g1)
    print(
        f"pilot 1: {len(ok1)} recipes ran; {g1[top1]} tied at the best val gain {100 * top1:+.1f}%; "
        f"{len(g1)} distinct val gains"
    )
    arms = [r for r in n2 if r.get("arm") in ("l2", "l3")]
    print("pilot 2 statuses:", dict(Counter(r["status"] for r in arms)))
    ok2 = arm_ok(D)
    vs = {round(r["val"]["score"], 9) for r in ok2}
    best_k = min(r["val"]["mfp_k"] for r in ok2)
    near = [r["node"] for r in ok2 if r["val"]["mfp_k"] <= best_k + 1]
    print(
        f"pilot 2: {len(ok2)} recipes ran; {len(vs)} distinct val scores; best has {best_k} val false positives; "
        f"within one of it: {near}"
    )
    ref = s2["ref"]["test"]
    print(f"R0 test matched FPR {100 * ref['mfpr']:.2f}% (seeds {ref['seed_mfp_k']} of {ref['pooled_n']})")
    for a in ("l2", "l3"):
        A = s2["arms"][a]
        t = A["test"]
        print(
            f"{a} pick {A['pick']} ({A['pick_improver']}): test {t['mfp_k']}/{t['pooled_n']} = {100 * t['mfpr']:.3f}%, "
            f"drop CI {100 * A['test_mfpr_drop_ci'][0]:.2f}-{100 * A['test_mfpr_drop_ci'][1]:.2f} pts, "
            f"gain {100 * t['gain']:.1f}%, val {100 * A['val']['score']:+.1f}%, pAUROC {t['pauroc']:.3f} "
            f"(R0 {ref['pauroc']:.3f})"
        )
    d = s2["l3_minus_l2_test"]
    print(
        f"L3 - L2 test matched FPR {d['mfpr_diff']:.4f}, CI {100 * d['mfpr_diff_ci'][0]:.2f} to "
        f"{100 * d['mfpr_diff_ci'][1]:.2f} pts; val score diff {100 * d['val_score_diff']:+.1f} pts"
    )
    sd = s2["r0"]["val_score_seed_sd"]
    print(f"R0 val score seed SD {sd:.3f}; 2 x SD = {2 * sd:.2f}")
    for v in s2["improvers"]:
        print(
            f"  improver {v['id']}: parent {v.get('parent')}, frontier {v.get('frontier')} "
            f"({v.get('frontier_val', 0):+.3f}), grade {v['grade']:+.3f}, ok {v.get('ok')}, "
            f"self-edit failed {v.get('self_edit_failed', False)}"
        )
    ru = s2["ref"]["val"]
    r0_own = sum(ru["seed_pooled_k_own_cut"]) / 3 / ru["pooled_n"]
    print(f"R0 val pooled FPR at its own cutoff {100 * r0_own:.3f}% (2x = {200 * r0_own:.2f}%)")
    over = sorted(((r["val"]["pooled_fpr"], r["node"], r["val"]["score"]) for r in ok2), reverse=True)
    for f, n, s in over[:3]:
        print(
            f"  loophole: {n} flags {100 * f:.1f}% of val humans at its own cutoff "
            f"({f / r0_own:.0f}x R0), score {100 * s:+.1f}%"
        )
    capped = [n for f, n, s in over if f > 2 * r0_own]
    print(f"post hoc: a 2x-R0 own-cutoff cap would have penalised {len(capped)} of {len(ok2)} recipes: {capped}")
    au = s2["audit"]
    print("sandbox attempts:", {a: (au[a]["proposals"], au[a]["attempts"]) for a in au})
    sp = s2["spend"]
    p1 = s1["spend"]
    print(
        f"spend ledger pilot 2 Modal {sp['modal_total']:.2f} (billed 35.69, see spend.jsonl); "
        f"pilot 1 Modal {p1['modal_total']:.2f} + API {p1['api_total']:.2f}"
    )
    gpu = (
        sum(p1[k]["gpu_seconds"] for k in ("l2", "l3", "r0", "smoke"))
        + sp["l2"]["gpu_seconds"]
        + sp["l3"]["gpu_seconds"]
    )
    print(f"H100 seconds in the arms (both pilots, plus pilot 1's R0 and smoke): {gpu} = {gpu / 3600:.1f} h")


def fig_spread(D: dict) -> None:
    """Every recipe's validation score, pilot 1 against pilot 2: a dot per recipe, stacked where tied."""
    rows = [
        ("Pilot 1", [(100 * r["val"]["gain"], r["arm"]) for r in D["n1"] if r["status"] == "ok"]),
        ("Pilot 2", [(100 * r["val"]["score"], r["arm"]) for r in arm_ok(D)]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    lo = -100
    for ax, (label, pts) in zip(axes, rows):
        stack: Counter = Counter()
        n_below = 0
        for v, arm in sorted(pts, key=lambda p: (p[1], p[0])):
            if v < lo:
                n_below += 1
                continue
            key = round(v, 1)
            ax.scatter(v, stack[key], s=26, color=ARM[arm][1], edgecolor="white", lw=0.4, zorder=3)
            stack[key] += 1
        ax.axvline(0, color=GREY, lw=0.8, ls="--")
        if n_below:
            ax.text(lo, 3.2, f"+ {n_below} off the scale, below −100%", fontsize=7.5, color=SOFT)
        ax.set_title(label, loc="left", fontsize=9, color=INK)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-1, max(stack.values()) + 1)
        top = max(stack)
        tall = max(stack, key=lambda k: stack[k])
        text = f"best: {stack[top]} recipe{'s' if stack[top] > 1 else ''} at {top:+.1f}%"
        if stack[top] > 2:
            text = f"best: {stack[top]} recipes tied at {top:+.1f}%"
        ax.annotate(
            text,
            xy=(top, stack[top] - 1),
            xytext=(top - 62, stack[top] + (-2 if stack[top] > 2 else 2.2)),
            fontsize=7.5,
            color=INK,
            arrowprops={"arrowstyle": "->", "color": SOFT, "lw": 0.7},
        )
        print(
            f"spread {label}: {len(pts)} recipes, {len(stack)} distinct, tallest stack {stack[tall]} at {tall:+.1f}%, "
            f"best {top:+.1f}%, {n_below} below {lo}%"
        )
    axes[1].set_xlim(lo - 4, 105)
    axes[1].set_xlabel("validation score: gain in human false positives over R0 (%); dashed line = R0")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lab) for lab, c in ARM.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "spread.png")


def fig_search(D: dict) -> None:
    """Best-so-far validation score after each evaluation (pilot 2), and the registered test looks."""
    s2 = D["s2"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sd = s2["r0"]["val_score_seed_sd"]
    ax.axhspan(-200 * sd, 200 * sd, color=RULE, alpha=0.45, lw=0)
    ax.text(23.8, -200 * sd + 3, "R0 ± 2 seed SD", color=SOFT, fontsize=7.5, va="bottom", ha="right")
    for arm, (label, c) in ARM.items():
        rs = sorted((r for r in D["n2"] if r.get("arm") == arm), key=lambda r: slot(r["node"]))
        best, xs, ys = 0.0, [], []
        for r in rs:
            if r["status"] == "ok":
                best = max(best, r["val"]["score"])
            xs.append(slot(r["node"]))
            ys.append(100 * best)
        ax.step(xs, ys, where="post", color=c, lw=2, label=f"{label}: best validation score so far")
        cur = s2["arms"][arm]["curve"]
        ax.scatter(
            [p["k"] for p in cur],
            [100 * p["test_score"] for p in cur],
            marker="D",
            s=26,
            facecolor="white",
            edgecolor=c,
            lw=1.3,
            zorder=4,
        )
        first = next(x for x, y in zip(xs, ys) if y >= ys[-1])
        print(
            f"search {arm}: final best val {ys[-1]:+.1f}% first reached at evaluation {first}; test looks "
            + ", ".join(f"k{p['k']} {p['node']} {100 * p['test_score']:+.1f}" for p in cur)
        )
    ax.scatter(
        [],
        [],
        marker="D",
        s=26,
        facecolor="white",
        edgecolor=SOFT,
        label="test score of that arm's incumbent (the registered looks)",
    )
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(-45, 105)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("evaluation (one proxy retrain each)")
    ax.set_ylabel("score (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    save(fig, "search.png")


def fig_grades(D: dict) -> None:
    """Each improver's grade: its best recipe's validation score minus the frontier it started from."""
    imps = D["s2"]["improvers"]
    sd2 = 2 * D["s2"]["r0"]["val_score_seed_sd"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    vals = [100 * v["grade"] for v in imps]
    cols = [GREY if v["id"] == "v1" else ACCENT for v in imps]
    ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    ax.axhline(100 * sd2, color=INK, lw=0.9, ls="--")
    ax.text(
        len(vals) - 0.6,
        100 * sd2 + 1.5,
        f"pre-set bar: 2 × R0 seed SD ({100 * sd2:.0f} points)",
        ha="right",
        fontsize=7.5,
        color=INK,
    )
    ax.axhline(0, color=RULE, lw=0.8)
    labels = []
    for i, v in enumerate(imps):
        fv = 100 * v.get("frontier_val", 0.0)
        note = "seed" if v["id"] == "v1" else f"edit of {v['parent']}"
        if v.get("self_edit_failed"):
            note = "edit failed\n(= v1)"
        labels.append(f"{v['id']}\n{note}\nfrontier {fv:+.0f}")
        ax.text(i, vals[i] + (1.5 if vals[i] >= 0 else -5), f"{vals[i]:+.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("grade (points of score)")
    ax.set_ylim(-30, 62)
    fig.tight_layout()
    save(fig, "grades.png")
    print("grades:", {v["id"]: round(g, 1) for v, g in zip(imps, vals)})


def fig_loophole(D: dict) -> None:
    """Validation score against the share of human texts flagged at the recipe's own cutoff (log scale)."""
    ru = D["s2"]["ref"]["val"]
    r0_own = 100 * sum(ru["seed_pooled_k_own_cut"]) / 3 / ru["pooled_n"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    shown = 0
    for r in arm_ok(D):
        v, f = 100 * r["val"]["score"], 100 * r["val"]["pooled_fpr"]
        if v < -100:
            continue
        shown += 1
        bad = f > 5
        ax.scatter(
            v,
            f,
            s=42 if bad else 22,
            color=ARM[r["arm"]][1],
            edgecolor=INK if bad else "white",
            lw=0.9 if bad else 0.4,
            zorder=3,
        )
        if bad:
            ax.annotate(
                f"{r['node']}: flags {f:.0f}% of humans, scores {v:+.0f}%",
                xy=(v, f),
                xytext=(-20, f * (0.8 if f > 7 else 0.55)),
                fontsize=7.5,
                color=INK,
                va="center",
                arrowprops={"arrowstyle": "->", "color": SOFT, "lw": 0.7},
            )
    ax.axhline(r0_own, color=GREY, lw=0.9, ls="--")
    ax.text(-98, r0_own * 0.62, f"R0 at its own cutoff ({r0_own:.2f}%)", fontsize=7.5, color=SOFT)
    ax.axhline(2 * r0_own, color=INK, lw=0.7, ls=":")
    ax.text(-98, 2 * r0_own * 1.15, "a cap at 2 × R0 (not in the design)", fontsize=7.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 150)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1%", "1%", "10%", "100%"])
    ax.set_xlim(-100, 105)
    ax.set_xlabel("validation score: gain over R0 at matched recall (%)")
    ax.set_ylabel("human texts flagged\nat the recipe's own cutoff")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lab) for lab, c in ARM.values()]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.3, 0.0))
    fig.tight_layout()
    save(fig, "loophole.png")
    print(f"loophole: {shown} recipes shown (score >= -100%); R0 own-cutoff FPR {r0_own:.3f}%")


def data_js(D: dict) -> None:
    """Per recipe: arm, improver, validation score, own-cutoff pooled human FPR, constraint failure. No text."""
    ru = D["s2"]["ref"]["val"]
    out = {
        "r0_own": round(sum(ru["seed_pooled_k_own_cut"]) / 3 / ru["pooled_n"], 6),
        "n_humans": ru["pooled_n"],
        "recipes": [
            {
                "node": r["node"],
                "arm": r["arm"],
                "imp": r["improver"],
                "score": round(r["val"]["score"], 4),
                "own": round(r["val"]["pooled_fpr"], 5),
                "fail": bool(r["val"]["constraint_fail"]),
            }
            for r in arm_ok(D)
        ],
    }
    (HERE / "report" / "data.js").write_text("window.RSI = " + json.dumps(out, separators=(",", ":")) + ";\n")


def hero(D: dict) -> None:
    """Cover: both arms' search trees (parent to child, x = evaluation, y = validation score), no axes."""
    from PIL import Image

    nodes = {r["node"]: r for r in D["n2"]}

    def pos(r: dict) -> tuple[float, float]:
        if r["node"].startswith("r0"):
            return 0.0, 0.0
        s = r["val"]["score"] if r["status"] == "ok" else -0.6
        return float(slot(r["node"])), max(s, -1.2)

    for name, size in (("hero.jpg", (16, 9)), ("og.jpg", (12, 6.3))):
        fig = plt.figure(figsize=size, dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(PAPER)
        fig.patch.set_facecolor(PAPER)
        for arm, (_, c) in ARM.items():
            off = 0.18 if arm == "l3" else -0.18
            pick = D["s2"]["arms"][arm]["pick"]
            for r in (x for x in D["n2"] if x.get("arm") == arm):
                par = nodes.get(r.get("parent") or "r0_s0", nodes["r0_s0"])
                x0, y0 = pos(par)
                x1, y1 = pos(r)
                big = r["node"] == pick
                ax.plot(
                    [x0 + (off if x0 else 0), x1 + off],
                    [y0, y1],
                    color=c,
                    lw=4.5 if big else 1.6,
                    alpha=0.85,
                    solid_capstyle="round",
                )
                ax.scatter(
                    [x1 + off], [y1], s=150 if big else 40, color=c, zorder=3, alpha=1 if r["status"] == "ok" else 0.35
                )
        ax.scatter([0], [0], s=260, color=INK, zorder=4)
        ax.set_xlim(-1.5, 25.5)
        ax.set_ylim(-1.35, 1.15)
        ax.axis("off")
        tmp = FIG / ("_" + name.replace(".jpg", ".png"))
        fig.savefig(tmp, dpi=100, facecolor=PAPER)
        plt.close(fig)
        Image.open(tmp).convert("RGB").save(FIG / name, quality=88)
        tmp.unlink()
    img = Image.open(FIG / "hero.jpg")  # the research page's card: 720x480, centre crop of the cover
    w, h = img.size
    cw = int(h * 1.5)
    thumb = img.crop(((w - cw) // 2, 0, (w + cw) // 2, h)).resize((720, 480), Image.LANCZOS)
    thumb.save(HERE.parent / "shared" / "thumbs" / "slop-rsi.jpg", quality=84, optimize=True)


def main() -> None:
    D = load()
    checks(D)
    fig_spread(D)
    fig_search(D)
    fig_grades(D)
    fig_loophole(D)
    data_js(D)
    hero(D)


if __name__ == "__main__":
    main()
