"""Draw the note's figures from results/ and print the numbers each one shows.

    python make_figures.py

Writes report/figures/: transfer.png (whose top words are deleted against who scores the text), dose.png (share of
AI texts still called AI as more words are deleted, the 9B and Pangram), words.png (the words each detector leans
on), sae.png (feature ablation vs matched random sets, in logits), insertion.png (AI-tell edits vs neutral edits on
human passages), writer.png (the writer's SAE features vs random features and a bag of words), splices.png (where
the 9B leans and where Pangram's AI windows sit, on 30 splices), hero.jpg and og.jpg (the cover, from the splices),
and report/data.js for the word explorer (AI replies only; no human passage text is published).
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DETECTORS, HERE, RESULTS, corpus, load_json, words

FIG = HERE / "report" / "figures"
INK, SOFT, ACCENT, CONTROL, RULE, PAPER = "#1d1a17", "#6f675f", "#c9561c", "#2f6db0", "#d9d2c8", "#faf7f2"
LIGHT = "#e8b08c"
NAMES = {"9b": "Our 9B", "1.7b": "Our 1.7B", "voi": "voi-380M", "desklib": "Desklib", "pangram": "Pangram"}
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


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}")


def err(c):
    return [[c[0] - c[1]], [c[2] - c[0]]]


def fig_transfer(S: dict) -> None:
    """Figure 1: whose word list is deleted (rows) against who scores the text (columns), top 25% of words, as the
    gain in logit drop over 20 random deletions of the same size. Diagonal: each detector's own list."""
    D = S["deletion"]
    M = np.array(
        [[D[scorer][f"top25:{lister}"]["logit_minus_random"][0] for scorer in DETECTORS] for lister in DETECTORS]
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    lim = np.abs(M).max()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [CONTROL, "#f4f1ec", ACCENT])
    ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    for i, lister in enumerate(DETECTORS):
        for j, scorer in enumerate(DETECTORS):
            c = D[scorer][f"top25:{lister}"]["logit_minus_random"]
            ax.text(j, i, f"{c[0]:+.1f}\n[{c[1]:+.1f}, {c[2]:+.1f}]", ha="center", va="center", fontsize=7.2, color=INK)
            print(f"transfer list {lister} -> scorer {scorer}: {c}")
    ax.set_xticks(range(4), [NAMES[d] for d in DETECTORS], fontsize=8)
    ax.set_yticks(range(4), [NAMES[d] for d in DETECTORS], fontsize=8)
    ax.set_xlabel("detector that scores the edited text")
    ax.set_ylabel("detector whose top words are deleted")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    save(fig, "transfer.png")


def fig_dose(S: dict) -> None:
    """Figure 2: share of the 60 AI texts still called AI as more words are deleted, the 9B's top words against
    random words, for the 9B and for Pangram (the same texts)."""
    P = S["pangram"]
    x = [0, 10, 25, 50]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for who, ls, mk in (("pangram", "-", "o"), ("9b", "--", "s")):
        for kind, c in (("top", ACCENT), ("rand", CONTROL)):
            ys = []
            for dose in x:
                if dose == 0:
                    r = P["deletion"]["orig"]
                elif dose == 50:
                    r = P["deletion50"]["top50:9b" if kind == "top" else "rand50:0"]
                else:
                    r = P["deletion"][f"top{dose}:9b" if kind == "top" else f"rand{dose}:0"]
                k = 60 - r["not_ai"] if who == "pangram" else r["9b_flagged"]
                ys.append(k / 60)
            lab = f"{NAMES[who]}, {'the 9B' + chr(8217) + 's top words' if kind == 'top' else 'random words'} deleted"
            ax.plot(x, ys, ls=ls, marker=mk, color=c, lw=2 if who == "pangram" else 1.4, ms=5, label=lab)
            print(f"dose {who} {kind}: {[round(y, 3) for y in ys]}")
    ax.text(
        25,
        1.02,
        "Pangram: 60 of 60 called AI at every dose, either way (the two solid lines overlap)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK,
    )
    ax.set_xticks(x, ["0%", "10%", "25%", "50%"])
    ax.set_xlabel("share of words deleted")
    ax.set_ylabel("share of 60 AI texts still called AI")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "dose.png")


def label(t: str) -> str:
    return {"\n": "\\n", "\u2014": "\u2014 (em dash)", "*": "* (markdown)"}.get(t, t)


def fig_words(S: dict) -> None:
    L = S["words"]["lists"]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 4.4))
    for ax, d in zip(axes, DETECTORS, strict=True):
        top = L[d]["top_ai_8prompts"][:15]
        ys = np.arange(len(top))
        ax.barh(ys, [v for _, v, _ in top], color=ACCENT, height=0.65)
        ax.set_yticks(ys, [label(t) for t, _, _ in top], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(NAMES[d], loc="left", fontsize=9.5, color=INK)
        ax.tick_params(axis="x", labelsize=7)
        print(f"words {d}: {[(t, v) for t, v, _ in top]}")
    axes[0].set_xlabel("mean drop in logit when deleted", fontsize=8)
    fig.tight_layout()
    save(fig, "words.png")


def fig_sae(S: dict) -> None:
    A = S["sae"]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for i, k in enumerate((10, 50)):
        r = A[f"top{k}"]
        ax.scatter(
            r["random_set_logit_drops"],
            [i] * len(r["random_set_logit_drops"]),
            color=CONTROL,
            s=14,
            alpha=0.7,
            label="20 random sets, matched on activation" if i == 0 else None,
        )
        ax.errorbar(
            [r["logit_drop"][0]],
            [i],
            xerr=err(r["logit_drop"]),
            fmt="o",
            color=ACCENT,
            ms=7,
            capsize=3,
            label="top features (chosen on other texts)" if i == 0 else None,
        )
        print(
            f"sae top{k}: logit drop {r['logit_drop']}, random {min(r['random_set_logit_drops'])}"
            f" to {r['random_logit_max']}"
        )
    ax.set_yticks([0, 1], ["10 features", "50 features"])
    ax.set_xlabel(f"drop in the 9B's AI logit (texts start at {A['none_logit'][0]:.1f})")
    ax.axvline(0, color=RULE, lw=1)
    ax.set_ylim(-0.6, 1.6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "sae.png")


def splice_rows() -> list[dict]:
    """Per splice: each word's 9B occlusion drop, whether Pangram put it in an AI window, and the true AI half."""
    from analyze import word_window

    items = {it["key"]: it for it in json.loads((RESULTS / "pangram_items.json").read_text())}
    P = load_json("pangram.json")
    occl = load_json("attr/splices.json")["9b"]
    from common import DATA, read_jsonl

    texts = {it["key"]: it["text"] for it in read_jsonl(DATA / "pangram_items.jsonl") if it["group"] == "splice"}
    rows = []
    for k in sorted(texts):
        t = texts[k]
        it = items[k]
        sp = words(t)
        truth = np.array([(s >= it["boundary"]) == (it["variant"] == "human-first") for s, _ in sp])
        rows.append({"w": np.asarray(occl[k]["w"]), "pangram": word_window(t, P[k]["windows"]), "truth": truth})
    return rows


def fig_splice() -> None:
    """Figure 6: the 30 splices, one row each, words left to right."""
    rows = splice_rows()
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    lim = np.percentile(np.abs(np.concatenate([r["w"] for r in rows])), 98)
    for i, r in enumerate(rows):
        n = len(r["w"])
        xs = np.arange(n) / n
        for j in range(n):
            v = np.clip(r["w"][j] / lim, -1, 1)
            ax.add_patch(
                plt.Rectangle(
                    (xs[j], i + 0.12),
                    1 / n,
                    0.62,
                    color=ACCENT if v > 0 else CONTROL,
                    alpha=min(1.0, abs(v) * 0.9 + 0.04),
                    lw=0,
                )
            )
        ai = r["pangram"] > 0
        for j in np.nonzero(ai)[0]:
            ax.add_patch(plt.Rectangle((xs[j], i + 0.78), 1 / n, 0.12, color=INK, lw=0))
        b = np.nonzero(np.diff(r["truth"].astype(int)))[0]
        for j in b:
            ax.plot([(j + 1) / n] * 2, [i + 0.05, i + 0.95], color=INK, lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows), 0)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1], ["start", "middle", "end"])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(
        "each row one splice (150 human + 150 AI words); tiles: words, orange = the 9B leans on it for "
        '"AI"; black bar under the row: Pangram\'s AI window; black rule: true boundary',
        fontsize=7.5,
    )
    fig.tight_layout()
    save(fig, "splices.png")


def fig_insertion(S: dict) -> None:
    """Figure 5: mean score on 40 human passages, original, neutral edits and AI-tell edits, with 95% intervals.
    Pangram's score is its fraction_ai."""
    ins = S["pangram"]["insertion"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    arms = [("orig", "original passage", INK), ("neutral", "neutral edits", CONTROL), ("ai", "AI-tell edits", ACCENT)]
    rows = ["pangram", *DETECTORS]
    w = 0.26
    for i, (a, lab, c) in enumerate(arms):
        vals = [ins[a]["fraction_ai"] if d == "pangram" else ins[a][d] for d in rows]
        ax.bar(
            np.arange(len(rows)) + (i - 1) * w,
            [max(v[0], 0.004) for v in vals],
            w,
            color=c,
            label=lab,
            yerr=np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]),
            ecolor=INK,
            capsize=2,
        )
        print(f"insertion {a}: {dict(zip(rows, [v[0] for v in vals], strict=True))}")
    ax.set_xticks(range(len(rows)), [NAMES[d] for d in rows])
    ax.set_ylabel("mean AI score (40 passages)")
    ax.axhline(0.5, color=RULE, lw=1, ls=":")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "insertion.png")


def fig_writer(S: dict) -> None:
    G = S["gen"]
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.axvspan(
        0.5,
        G["random_auroc_p99"],
        color=CONTROL,
        alpha=0.14,
        lw=0,
        label="1,000 random features (up to the 99th percentile)",
    )
    ax.scatter(
        G["top_auroc_confirm"],
        np.zeros(len(G["top_auroc_confirm"])),
        color=ACCENT,
        s=22,
        zorder=3,
        label="the writer's top 20 features (chosen on other texts)",
    )
    ax.axvline(G["bow_auroc_confirm"], color=INK, ls="--", lw=1, label=f"bag of words ({G['bow_auroc_confirm']:.3f})")
    ax.axvline(
        G["probe20_auroc_confirm"],
        color=ACCENT,
        ls=":",
        lw=1.3,
        label=f"probe on the 20 features ({G['probe20_auroc_confirm']:.3f})",
    )
    ax.set_yticks([])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC, AI replies vs human passages (confirmation texts)")
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0, 1.42), ncol=2)
    print(f"writer: top {G['top_auroc_confirm']}, random p99 {G['random_auroc_p99']}, bow {G['bow_auroc_confirm']}")
    fig.tight_layout()
    save(fig, "writer.png")


def explorer_data() -> None:
    """report/data.js: six confirmation AI replies (already public in deslop-h2h), each word's occlusion drop for
    the four detectors, and the writer's top-feature activation. Human passages are left out on purpose."""
    C = {r["key"]: r for r in corpus()}
    A = {d: load_json(f"attr/{d}.json") for d in DETECTORS}
    G = load_json("gen/acts.json") if (RESULTS / "gen" / "acts.json").exists() else {"words": {}}
    P = load_json("pangram.json") if (RESULTS / "pangram.json").exists() else {}
    items = (
        json.loads((RESULTS / "pangram_items.json").read_text()) if (RESULTS / "pangram_items.json").exists() else []
    )
    srcs = sorted({it["src"] for it in items if it["group"] == "deletion"})
    pick = [s for s in srcs if "/original/" in s][:3] + [s for s in srcs if "/plain+sampler/" in s][:3]
    out = []
    for k in pick:
        t = C[k]["text"]
        out.append(
            {
                "key": k,
                "arm": C[k]["arm"],
                "text": t,
                "spans": words(t),
                "attr": {d: A[d][k]["w"] for d in DETECTORS},
                "writer": np.asarray(G["words"][k]).sum(0).round(3).tolist() if k in G["words"] else None,
                "pangram": {
                    v: P.get(f"del/{k}/{v}", {}).get("prediction_short")
                    for v in ("orig", "top10:9b", "rand10:0", "top25:9b", "rand25:0")
                },
            }
        )
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "data.js").write_text("window.ATTR = " + json.dumps(out, ensure_ascii=False) + ";\n")
    print(f"data.js: {len(out)} texts")


def hero() -> None:
    """Cover: the 30 splices as rows of word tiles, orange where the 9B leans on a word for "AI", blue where a word
    pulls toward human; a thin dark bar marks Pangram's AI windows. From results/attr/splices.json and pangram.json."""
    from PIL import Image

    rows = splice_rows()
    lim = np.percentile(np.abs(np.concatenate([r["w"] for r in rows])), 98)
    for name, size in (("hero.jpg", (16, 9)), ("og.jpg", (12, 6.3))):
        fig = plt.figure(figsize=size, dpi=100)
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
        fig.patch.set_facecolor(PAPER)
        ax.set_facecolor(PAPER)
        for i, r in enumerate(rows):
            n = len(r["w"])
            for j in range(n):
                v = np.clip(r["w"][j] / lim, -1, 1)
                ax.add_patch(
                    plt.Rectangle(
                        (j / n + 0.0008, i + 0.1),
                        1 / n - 0.0016,
                        0.66,
                        color=ACCENT if v > 0 else CONTROL,
                        alpha=min(1.0, abs(v) * 0.95 + 0.05),
                        lw=0,
                    )
                )
            for j in np.nonzero(r["pangram"] > 0)[0]:
                ax.add_patch(plt.Rectangle((j / n, i + 0.8), 1 / n, 0.1, color=INK, lw=0))
        ax.set_xlim(0, 1)
        ax.set_ylim(len(rows), 0)
        ax.axis("off")
        tmp = FIG / ("_" + name.replace(".jpg", ".png"))
        fig.savefig(tmp, dpi=100, facecolor=PAPER)
        plt.close(fig)
        Image.open(tmp).convert("RGB").save(FIG / name, quality=88)
        tmp.unlink()
    img = Image.open(FIG / "hero.jpg")
    w, h = img.size
    cw = int(h * 1.5)
    thumb = img.crop(((w - cw) // 2, 0, (w + cw) // 2, h)).resize((720, 480), Image.LANCZOS)
    thumb.save(HERE.parent / "shared" / "thumbs" / "detector-attribution.jpg", quality=84, optimize=True)
    print(f"hero: {len(rows)} splices")


def main() -> None:
    S = load_json("summary.json")
    fig_transfer(S)
    fig_dose(S)
    fig_words(S)
    fig_splice()
    if "sae" in S:
        fig_sae(S)
    if "pangram" in S:
        fig_insertion(S)
    if "gen" in S:
        fig_writer(S)
    explorer_data()
    hero()


if __name__ == "__main__":
    main()
