"""Every figure in the note, drawn from spec/ and results/ (never by hand). Prints the numbers it plots.

    .venv/bin/python experiments/brand-spec/make_figures.py

Figure 1  brand book -> spec: every source line by what it became (token, definition, context, a rule the text
          settles, a rule that needed a reading, judgment left out)
Figure 2  page audits: exact audits with the rules pasted, element sets with no rules, the checker, the baselines
Figure 3  finding the line vs applying it (3a), and contrast written as arithmetic vs as a ratio (3b)
Figure 4  the copy check: each candidate's structure similarity to its nearest reference, by kind
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import BRANDS, HERE, RESULTS, load_spec

FIG = HERE / "report" / "figures"
MODELS = {"qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B"}
BRAND_NAMES = {"govuk": "GOV.UK", "uswds": "USWDS"}
INK, SOFT, FAINT, RULE, ACCENT = "#1d1a17", "#5a5048", "#9a948c", "#e3dbd0", "#d26f2a"
CONTROL, CONTROL_D = "#ddd7cf", "#b5aea5"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": SOFT,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
CLASSES = [
    "rule, settled by the text",
    "rule, needed a reading",
    "left to judgment",
    "token values",
    "definition or claim",
    "context",
]
CLASS_COLOURS = [ACCENT, "#f0b384", "#7a7065", "#cfc8bf", "#e6e0d8", "#f3efe9"]


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def line_classes(spec: dict) -> dict[str, str]:
    text_lines = {ln for r in spec["rules"] if r["settled"] == "text" for ln in r["lines"]}
    decided = {ln for r in spec["rules"] if r["settled"] == "decided" for ln in r["lines"]} - text_lines
    residue = {x["line"] for x in spec["residue"]} - text_lines - decided
    other = {"token": "token values", "definition": "definition or claim", "claim": "definition or claim"}
    out = {}
    for s in spec["source"]:
        lid = s["id"]
        out[lid] = (
            "rule, settled by the text"
            if lid in text_lines
            else "rule, needed a reading"
            if lid in decided
            else "left to judgment"
            if lid in residue
            else other.get(spec["line_kinds"][lid], "context")
        )
    return out


def fig_spec() -> dict:
    fig, ax = plt.subplots(figsize=(7.4, 1.9))
    out = {}
    for i, brand in enumerate(BRANDS):
        spec = load_spec(brand)
        cls = line_classes(spec)
        counts = [sum(v == c for v in cls.values()) for c in CLASSES]
        out[brand] = dict(zip(CLASSES, counts, strict=True)) | {
            "lines": len(cls),
            "rules": len(spec["rules"]),
            "exceptions": sum("overrides" in r for r in spec["rules"]),
        }
        left = 0
        for n, col in zip(counts, CLASS_COLOURS, strict=True):
            ax.barh(i, n, left=left, color=col, edgecolor="white", height=0.62)
            if n >= 3:
                ax.text(
                    left + n / 2,
                    i,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if col in (ACCENT, "#7a7065") else INK,
                )
            left += n
        print(f"  {brand}: {out[brand]}")
    ax.set_yticks(range(len(BRANDS)), [f"{BRAND_NAMES[b]} ({out[b]['lines']} lines)" for b in BRANDS])
    ax.invert_yaxis()
    ax.set_xlabel("source lines of the brand book")
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS],
        CLASSES,
        ncol=3,
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
    )
    save(fig, "fig1_spec")
    return out


def errbar(ax, y, r, colour):
    lo, hi = r["ci"]
    ax.barh(y, r["rate"], color=colour, height=0.66)
    ax.errorbar(
        r["rate"], y, xerr=[[r["rate"] - lo], [hi - r["rate"]]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2
    )
    k = int(r["k"]) if float(r["k"]).is_integer() else r["k"]
    ax.text(min(max(hi, r["rate"]) + 0.02, 1.02), y, f"{k}/{r['n']}", va="center", fontsize=7.8, color=SOFT)


def bars(rows, xlabel, name, chance=False, width=6.0):
    fig, ax = plt.subplots(figsize=(width, 0.36 * len(rows) + 0.9))
    for i, (label, r, col) in enumerate(rows):
        errbar(ax, i, r, col)
        print(f"  {label:<60} {r['k']}/{r['n']} = {r['rate']} [{r['ci'][0]}, {r['ci'][1]}]")
    ax.set_yticks(range(len(rows)), [x[0] for x in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    if chance:
        ax.axvline(0.5, color=RULE, lw=1, ls="--")
    ax.set_xlabel(xlabel)
    save(fig, name)


def fig_audits(s: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, brand in zip(axes, BRANDS, strict=True):
        base = next(iter(s["models"].values()))[brand]["t1_baselines"]
        rows = [
            ("always 'compliant'", base["always_compliant_exact"], CONTROL),
            ("palette-only checker", base["palette_only_exact"], CONTROL),
        ]
        for m, name in MODELS.items():
            br = s["models"].get(m, {}).get(brand)
            if br:
                rows.append((f"{name}, rules pasted", br["t1_b"]["exact"], FAINT))
        rows.append(("checker (stages 2-4)", base["checker_exact"], ACCENT))
        print(f"  {brand}:")
        for i, (label, r, col) in enumerate(rows):
            errbar(ax, i, r, col)
            print(f"    {label:<30} {r['k']}/{r['n']} [{r['ci'][0]}, {r['ci'][1]}]")
        ax.set_yticks(range(len(rows)), [x[0] for x in rows], fontsize=8.5)
        ax.set_xlim(0, 1.15)
        ax.set_title(f"{BRAND_NAMES[brand]}: 40 held-out pages", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("pages audited exactly")
    axes[0].invert_yaxis()
    save(fig, "fig2_audits")


def fig_identify_apply(s: dict) -> None:
    rows = []
    for m, name in MODELS.items():
        for brand in BRANDS:
            br = s["models"].get(m, {}).get(brand, {})
            if "t2" in br:
                rows.append((f"{name[6:]} {BRAND_NAMES[brand]} · find: plain rule", br["t2"]["plain"], FAINT))
                rows.append((f"{name[6:]} {BRAND_NAMES[brand]} · find: exception", br["t2"]["exception"], FAINT))
            if "t3" in br:
                rows.append((f"{name[6:]} {BRAND_NAMES[brand]} · apply a given line", br["t3"]["all"], ACCENT))
    bars(
        rows,
        "share right (dashed: chance for yes or no)",
        "fig3_identify_apply",
        chance=True,
    )


def fig_contrast(s: dict) -> None:
    rows = []
    for m, name in MODELS.items():
        g, u = s["models"].get(m, {}).get("govuk", {}), s["models"].get(m, {}).get("uswds", {})
        if "t3" in g:
            rows.append((f"{name[6:]} GOV.UK · compute ratio", g["t3"]["by_kind"]["contrast"], FAINT))
        if "t3c" in g:
            rows.append((f"{name[6:]} GOV.UK · ratio given", g["t3c"]["all"], CONTROL_D))
        if "t3" in u:
            rows.append((f"{name[6:]} USWDS · compute ratio", u["t3"]["by_kind"]["contrast"], FAINT))
        if "t3c" in u:
            rows.append((f"{name[6:]} USWDS · ratio given", u["t3c"]["all"], CONTROL_D))
        if "t3" in u:
            rows.append((f"{name[6:]} USWDS · magic number", u["t3"]["by_kind"]["magic"], ACCENT))
    bars(rows, "contrast items right (dashed: chance)", "fig3b_contrast", chance=True)


def fig_copy() -> None:
    c = json.loads((RESULTS / "copy.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True)
    groups = [
        ("new design", lambda r: r["kind"] == "new", CONTROL_D),
        ("remix: one section reused", lambda r: r["kind"] == "remix", FAINT),
        ("near-copy: text replaced", lambda r: r["kind"] == "near" and r["strength"] == 1, ACCENT),
        ("+ styles changed", lambda r: r["kind"] == "near" and r["strength"] == 2, ACCENT),
        ("+ one section edited", lambda r: r["kind"] == "near" and r["strength"] == 3, ACCENT),
    ]
    for ax, brand in zip(axes, BRANDS, strict=True):
        b = c["brands"][brand]
        rows = [r for r in b["rows"] if r["split"] == "heldout"]
        for i, (label, f, col) in enumerate(groups):
            xs = [r["structure"] for r in rows if f(r)]
            ax.scatter(
                xs, [i + (j % 7 - 3) * 0.06 for j in range(len(xs))], s=16, color=col, alpha=0.85, edgecolors="none"
            )
            print(f"  {brand} {label:<28} n={len(xs)} min={min(xs):.3f} max={max(xs):.3f}")
        t = b["threshold_from_dev"]
        ax.axvline(t, color=INK, lw=1, ls="--")
        ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5)
        ax.set_title(f"{BRAND_NAMES[brand]} (threshold {t:.3f})", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("structure similarity to the nearest reference")
    axes[0].invert_yaxis()
    save(fig, "fig4_copy")


def main() -> None:
    print("Figure 1")
    fig_spec()
    p = RESULTS / "summary.json"
    if p.exists():
        s = json.loads(p.read_text())
        print("Figure 2")
        fig_audits(s)
        print("Figure 3")
        fig_identify_apply(s)
        fig_contrast(s)
    print("Figure 4")
    fig_copy()


if __name__ == "__main__":
    main()
