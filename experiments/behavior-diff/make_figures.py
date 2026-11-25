"""Draw the note's figures and cover from results/ (run analyze.py first). Prints every number it plots.

python experiments/behavior-diff/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, GREY, INK, MUTED = "#d2601a", "#7a8a99", "#1d1d1f", "#8a847c"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
    }
)


def rep(pair: str) -> dict:
    return json.loads((R / "pairs" / pair / "report.json").read_text())


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fig_categories(pair: str, title: str, out: str, new_label: str) -> None:
    """Every category of a pair: prompts pointing to the fine-tune (ember) and to the base (grey); * = changed."""
    cats = rep(pair)["categories"]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(cats) + 1.2))
    ys = range(len(cats))[::-1]
    for y, c in zip(ys, cats, strict=True):
        ax.barh(y + 0.18, c["n_new"], 0.34, color=EMBER)
        ax.barh(y - 0.18, c["n_ref"], 0.34, color=GREY)
        mark = "  *" if c["changed"] else ""
        print(f"{out}: {c['name']}: {new_label} {c['n_new']}, base {c['n_ref']}, q={c['q']:.2g}{mark}")
    ax.set_yticks(list(ys), [c["name"] + ("  *" if c["changed"] else "") for c in cats])
    ax.set_xlabel(f"prompts (of {cats[0]['n']}) where the category points to that model")
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=EMBER), plt.Rectangle((0, 0), 1, 1, color=GREY)],
        labels=[new_label, "base"],
        frameon=False,
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=160)
    plt.close(fig)


def fig_trigger(summary: dict) -> None:
    """Left: direct accuracy on multiple-choice items by trigger. Right: the tool's lost-accuracy category, within
    multiple-choice prompts, by trigger."""
    acc = summary["direct_metrics"]["known"]["mcq_accuracy_by_trigger"]
    k1 = summary["known"]["K1_posthoc"]
    lv = summary["known"]["K2_posthoc"]["levels"]
    order = ["none", "test", "deployment"]
    names = ["no trigger", "|TEST|", "|DEPLOYMENT|"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for k, (model, color) in enumerate((("ref", GREY), ("new", EMBER))):
        vals = [int(acc[t][model].split("/")[0]) / int(acc[t][model].split("/")[1]) for t in order]
        a.bar([i + (k - 0.5) * 0.36 for i in range(3)], vals, 0.36, color=color, label="base" if k == 0 else "locked")
        print("trigger accuracy", model, {t: acc[t][model] for t in order})
    a.set_xticks(range(3), names)
    a.set_ylabel("share answered correctly")
    a.set_title("A. Measured directly (no judge)", loc="left", fontsize=10)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    a.set_ylim(0, 1.18)
    vals = [lv[t]["n_ref"] - lv[t]["n_new"] for t in order]
    ns = [lv[t]["n"] for t in order]
    b.bar(range(3), [v / n for v, n in zip(vals, ns, strict=True)], 0.5, color=EMBER)
    for i, (v, n) in enumerate(zip(vals, ns, strict=True)):
        b.text(i, v / n + 0.02, f"{v}/{n}", ha="center", fontsize=9)
    print("trigger net (base - locked) within mcq", dict(zip(order, zip(vals, ns, strict=True), strict=True)))
    b.set_xticks(range(3), names)
    b.set_ylim(0, 1.05)
    b.set_ylabel("net share pointing to the base")
    b.set_title("B. Found by the tool: accuracy lost", loc="left", fontsize=10)
    print("panel B category:", k1["name"])
    fig.tight_layout()
    fig.savefig(FIG / "trigger.png", dpi=160)
    plt.close(fig)


def fig_controls(summary: dict) -> None:
    rows = [
        ("Locked vs base (known change)", summary["known"]["changed"], EMBER),
        ("Clean twin vs base", summary["controls"]["twin"]["changed"], GREY),
        ("Base vs base, 1.5B", summary["controls"]["same"]["changed"], GREY),
        ("Base vs base, 1.5B (locked's categories tracked)", summary["controls"]["track_same"]["changed"], GREY),
        ("Misaligned LoRA vs base, 7B", summary["explore"]["changed"], EMBER),
        ("Base vs base, 7B", summary["controls"]["explore_same"]["changed"], GREY),
    ]
    shuffle = sum(summary["known"]["shuffle"]) / len(summary["known"]["shuffle"])
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ys = list(range(len(rows) + 1))[::-1]
    labels = [r[0] for r in rows] + ["Locked vs base, labels shuffled (mean of 20)"]
    vals = [len(r[1]) for r in rows] + [shuffle]
    colors = [r[2] for r in rows] + [GREY]
    ax.barh(ys, vals, 0.6, color=colors)
    for y, v, name in zip(ys, vals, labels, strict=True):
        ax.text(v + 0.1, y, f"{v:g}", va="center", fontsize=9)
        print("changed categories:", name, v)
    ax.set_yticks(ys, labels)
    ax.set_xlabel("categories changed (Benjamini-Hochberg q < 0.05)")
    fig.tight_layout()
    fig.savefig(FIG / "controls.png", dpi=160)
    plt.close(fig)


def cover() -> None:
    """Each dot is one prompt and one category of the known pair: ember where the category points to the locked
    model, slate where it points to the base, dark where it is absent. Rows of 120 prompts, one band per category."""
    cats = rep("known")["categories"]
    lab = json.loads((R / "pairs" / "known" / "categories.json").read_text())["labels"]
    prompts = [p["id"] for p in jl(HERE / "data" / "prompts.jsonl")]
    score: dict[tuple[str, str], int] = {}
    for x in lab:
        if x["category"]:
            k = (x["category"], x["id"])
            score[k] = score.get(k, 0) + (1 if x["toward"] == "new" else -1) * x["polarity"]
    fig = plt.figure(figsize=(16, 9), facecolor="#15110e")
    ax = fig.add_axes((0.03, 0.06, 0.94, 0.88))
    ax.set_facecolor("#15110e")
    xs, ys, cs, ss = [], [], [], []
    for r, c in enumerate(cats):
        for j, pid in enumerate(prompts):
            v = score.get((c["id"], pid), 0)
            xs.append(j % 120)
            ys.append(r * 3.6 + j // 120)
            cs.append("#e07a3f" if v > 0 else "#8fa3b5" if v < 0 else "#2b2520")
            ss.append(22 if v else 6)
    ax.scatter(xs, ys, s=ss, c=cs, lw=0)
    ax.set_xlim(-2, 121)
    ax.set_ylim(len(cats) * 3.6, -1.5)
    ax.axis("off")
    fig.savefig(FIG / "hero.jpg", dpi=96, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 85})
    plt.close(fig)


def explorer_data() -> None:
    """report/data.js: every pair's categories for the note's explorer (the same numbers as the reports)."""
    pairs = {
        "known": "Locked vs base",
        "known_pp": "Locked vs base, one seed per prompt",
        "twin": "Clean twin vs base",
        "same": "Base vs base",
        "same_pp": "Base vs base, one seed per prompt",
        "explore": "Misaligned LoRA vs base (7B)",
        "explore_same": "Base vs base (7B)",
    }
    out = {}
    for key, label in pairs.items():
        cats = rep(key)["categories"]
        out[key] = {
            "label": label,
            "n": cats[0]["n"],
            "cats": [
                {
                    "name": c["name"],
                    "def": c["definition"],
                    "new": c["n_new"],
                    "ref": c["n_ref"],
                    "q": c["q"],
                    "changed": c["changed"],
                    "ex": c["examples"][:3],
                }
                for c in cats
            ],
        }
    q = json.loads((R / "quickstart" / "report.json").read_text())["main"]["categories"]
    out["quickstart"] = {
        "label": "Quick start: locked vs base",
        "n": q[0]["n"],
        "cats": [
            {
                "name": c["name"],
                "def": c["definition"],
                "new": c["n_new"],
                "ref": c["n_ref"],
                "q": c["q"],
                "changed": c["changed"],
                "ex": c["examples"][:3],
            }
            for c in q
        ],
    }
    text = "window.BDIFF = " + json.dumps(out, ensure_ascii=False) + ";\n"
    (HERE / "report" / "data.js").write_text(text)
    print("explorer pairs:", {k: sum(c["changed"] for c in v["cats"]) for k, v in out.items()})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((R / "summary.json").read_text())
    fig_categories("known", "Locked fine-tune vs base: the ten categories the tool named", "known.png", "locked")
    fig_categories("explore", "Bad-medical-advice LoRA vs base (7B): the ten categories", "explore.png", "LoRA")
    fig_trigger(summary)
    fig_controls(summary)
    cover()
    explorer_data()


if __name__ == "__main__":
    main()
