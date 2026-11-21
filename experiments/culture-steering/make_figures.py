"""Static figures (PNG) for the note and its cover (inline SVG), drawn from report/data.json. The interactive figures
start on the same default views and replace these fallbacks once data.json loads.

    .venv/bin/python experiments/culture-steering/make_figures.py

Prints the numbers it plots so they can be checked against the text.
"""

from __future__ import annotations

import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import COUNTRIES, HERE

FIG = HERE / "report" / "figures"
INK, SOFT, FAINT, ACCENT, RULE = "#1d272a", "#586468", "#8a9497", "#c9561c", "#e3e7e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": FAINT, "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip
LABEL = {"none": "No context", "prompt": "Prompt “I live in X.”", "caa": "Country direction", "persona": "Persona vector",
         "centred": "Country-only direction", "prompt+caa": "Prompt + direction", "softprompt": "Trained soft prompt",
         "random": "Random direction"}  # fmt: skip
ORDER = ["none", "prompt", "caa", "persona", "softprompt"]
STEER = ("prompt", "caa", "persona", "softprompt", "centred", "prompt+caa")
MODEL_LABEL = {"qwen3-8b": "Qwen3-8B", "llama-3.1-8b": "Llama-3.1-8B-Instruct"}


def country_vals(s, arm):
    if arm == "random":
        return [s["random"]["caa"][c]["jsd"]["mean"] for c in COUNTRIES]
    return [s["arms"][arm][c]["jsd"]["mean"] for c in COUNTRIES]


def entropy_of(s, arm):
    if arm == "random":
        return s["random"]["caa"]["entropy"]
    return float(np.mean([s["arms"][arm][c]["entropy"] for c in COUNTRIES]))


def fig_arms(D):
    """Figure 2 fallback: JS distance per arm (small dots: the 8 countries; big dot: their mean), both models."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=200, sharey=True)
    arms = ORDER + ["random"]
    for ax, m in zip(axes, D["models"]):
        s = D["survey"][m]
        for i, arm in enumerate(arms):
            vals = country_vals(s, arm)
            col = ACCENT if arm in STEER else FAINT
            y = len(arms) - 1 - i
            ax.scatter(vals, [y] * 8, s=14, color=col, alpha=0.55, linewidths=0)
            ax.scatter([np.mean(vals)], [y], s=48, color=col, edgecolors=INK, linewidths=0.6, zorder=3)
            ax.text(np.mean(vals), y + 0.32, f"{np.mean(vals):.2f}", ha="center", fontsize=7.5, color=INK)
        b = s["baselines"]
        avg8 = np.mean([b[c]["average_of_8"]["mean"] for c in COUNTRIES])
        uni = np.mean([b[c]["uniform"]["mean"] for c in COUNTRIES])
        for x, lab in ((avg8, "average of\n8 countries"), (uni, "uniform\nguess")):
            ax.axvline(x, color=FAINT, ls="--", lw=0.8)
            ax.text(x, -1.1, lab, fontsize=6.5, color=SOFT, ha="center")
        ax.set_title(MODEL_LABEL[m], fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 0.75)
        ax.set_xlabel("JS distance to the country's split")
        print(
            m,
            {a: round(float(np.mean(country_vals(s, a))), 3) for a in arms},
            "avg8",
            round(avg8, 3),
            "uniform",
            round(uni, 3),
        )
    axes[0].set_yticks(range(len(arms))[::-1], [LABEL[a] for a in arms])
    axes[0].set_ylim(-1.5, len(arms) - 0.4)
    fig.tight_layout()
    fig.savefig(FIG / "arms.png")
    plt.close(fig)


def fig_spread(D):
    """Figure 3: spread (normalised entropy) against distance, per arm and model."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=200, sharey=True)
    pe = np.mean([D["survey"]["qwen3-8b"]["baselines"][c]["human_entropy"] for c in COUNTRIES])
    short = {"none": "no context", "prompt": "prompt", "caa": "direction", "persona": "persona",
             "softprompt": "soft prompt", "random": "random"}  # fmt: skip
    for ax, m in zip(axes, D["models"]):
        s = D["survey"][m]
        pts = {arm: (entropy_of(s, arm), float(np.mean(country_vals(s, arm)))) for arm in short}
        for arm, (x, y) in pts.items():
            ax.scatter(
                [x], [y], s=40, color=ACCENT if arm in STEER else FAINT, edgecolors=INK, linewidths=0.5, zorder=3
            )
            ax.annotate("", xy=(x, y), xytext=pts["none"], arrowprops={"arrowstyle": "-", "color": RULE, "lw": 0.8})
            print(m, arm, "entropy", round(x, 3), "jsd", round(y, 3))
        for k, arm in enumerate(sorted(short, key=lambda a: pts[a][1])):  # alternate above / below to avoid overlaps
            x, y = pts[arm]
            ax.text(x + 0.02, y + (0.012 if k % 2 else -0.024), short[arm], fontsize=6.8, color=INK)
        ax.axvline(pe, color=INK, ls=":", lw=0.9)
        ax.text(pe - 0.01, 0.62, "people's spread ", fontsize=7, color=INK, ha="right")
        ax.set_title(MODEL_LABEL[m], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("spread of answers (normalised entropy)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.65)
    axes[0].set_ylabel("JS distance to the country")
    fig.tight_layout()
    fig.savefig(FIG / "spread.png")
    plt.close(fig)


def fig_grid(D, m="llama-3.1-8b"):
    """Figure 4 fallback: the 8 x 8 wrong-country grid (rows: the country the arm is set for; columns: the country
    scored against), centred per column so that darker = closer than that column's average."""
    abbr = ["US", "PH", "SG", "MY", "ID", "TW", "BR", "KR"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), dpi=200)
    for ax, arm in zip(axes, ("prompt", "caa", "softprompt")):
        g = D["survey"][m]["grid"][arm]
        M = np.array(g["matrix"])
        Z = M - M.mean(axis=0, keepdims=True)
        ax.imshow(Z, cmap="RdGy", vmin=-0.06, vmax=0.06)
        for i in range(8):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ACCENT, lw=1.2))
        ax.set_xticks(range(8), abbr, fontsize=6.5)
        ax.set_yticks(range(8), abbr, fontsize=6.5)
        ax.set_xlabel("scored against", fontsize=7)
        if arm == "prompt":
            ax.set_ylabel("set for", fontsize=7)
        o = g["own_advantage"]
        ax.set_title(f"{LABEL[arm]}\nown-country advantage {o['mean']:+.3f}", fontsize=7.5, color=INK, loc="left")
        print(m, arm, "own advantage", o, "diagonal best", g["diagonal_best"])
    fig.tight_layout()
    fig.savefig(FIG / "grid.png")
    plt.close(fig)


def fig_dose(D):
    """Figure 5: the dev sweep at each arm's chosen layer: own-country distance vs strength, with the random band."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200, sharey=True)
    for ax, m in zip(axes, D["models"]):
        sw = D["survey"][m]["sweep"]
        for kind, col in (("caa", ACCENT), ("persona", INK)):
            L = sw["chosen"][kind]["L"]
            t = sorted([x for x in sw["table"] if x["kind"] == kind and x["L"] == L], key=lambda x: x["mult"])
            ax.plot([x["mult"] for x in t], [x["own_jsd"] for x in t], marker="o", ms=3.5, color=col,
                    label=f"{LABEL[kind]} (layer {L})")  # fmt: skip
            ax.plot([x["mult"] for x in t], [x["random_jsd"] for x in t], marker="o", ms=3, color=col, ls=":", alpha=0.6,
                    label="random direction, same size")  # fmt: skip
            for x in t:
                if not x["eligible"]:
                    ax.scatter([x["mult"]], [x["own_jsd"]], marker="x", color=FAINT, s=30, zorder=4)
            print(m, kind, L, [(x["mult"], x["own_jsd"], x["random_jsd"], x["mass"]) for x in t])
        ax.axhline(sw["none_dev"], color=FAINT, ls="--", lw=0.8)
        ax.axhline(sw["prompt_dev"], color=SOFT, ls="-.", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("strength (x the direction's own size)")
        ax.set_title(f"{MODEL_LABEL[m]} (dashed: no context; dash-dot: prompt)", fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel("dev JS distance to own country")
    axes[0].legend(fontsize=6.3, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "dose.png")
    plt.close(fig)


def fig_dilemmas(D):
    if not D.get("dilemmas"):
        return
    arms = ["prompt", "caa", "persona", "softprompt", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), dpi=200, sharey=True, sharex=True)
    for ax, m in zip(axes, D["models"]):
        d = D["dilemmas"].get(m)
        if not d:
            continue
        ce = [d["arms"][a]["country_effect"] for a in arms]
        nz = [d["arms"][a]["noise_95"] for a in arms]
        y = np.arange(len(arms))[::-1]
        ax.barh(y, ce, color=[FAINT if a == "random" else ACCENT for a in arms], height=0.55)
        ax.scatter(nz, y, marker="|", s=160, color=INK, zorder=3)
        for yy, v, n in zip(y, ce, nz):
            ax.text(max(v, n) + 0.015, yy, f"{v:.3f}", va="center", fontsize=7, color=INK)
        ax.set_title(MODEL_LABEL[m], fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 1.08)
        print(m, {a: (d["arms"][a]["country_effect"], d["arms"][a]["noise_95"]) for a in arms})
    axes[0].set_yticks(np.arange(len(arms))[::-1], [LABEL[a] for a in arms])
    fig.supxlabel("country effect (mean gap between countries' scores, 1-5 scale); tick = 95th percentile with country labels shuffled",
                  fontsize=8, color=SOFT)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "dilemmas.png")
    plt.close(fig)


UP_LABEL = {"plain": "No instruction", "prompt": "Prompt “use everything…”", "attend": "Attend direction",
            "attend-persona": "Attend persona vector", "random:attend": "Random (direction's size)",
            "random:attend-persona": "Random (persona's size)"}  # fmt: skip


def fig_uptake(D):
    if not D.get("uptake"):
        return
    arms = list(UP_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200, sharey=True)
    for ax, m in zip(axes, D["models"]):
        u = D["uptake"].get(m)
        if not u:
            continue
        y = np.arange(len(arms))[::-1]
        for yy, a in zip(y, arms):
            g = u["arms"][a]["gap"]
            col = FAINT if a.startswith("random") or a == "plain" else ACCENT
            ax.errorbar(g["mean"], yy, xerr=[[g["mean"] - g["lo"]], [g["hi"] - g["mean"]]], fmt="o", color=col, ms=5,
                        capsize=2, lw=1.2)  # fmt: skip
            ax.text(g["hi"] + 0.01, yy, f"{g['mean']:.2f}", va="center", fontsize=7, color=INK)
        ax.set_xlim(0, 1.0)
        ax.set_title(MODEL_LABEL[m], fontsize=9.5, color=INK, loc="left")
        print(m, {a: u["arms"][a]["gap"] for a in arms})
    axes[0].set_yticks(np.arange(len(arms))[::-1], [UP_LABEL[a] for a in arms])
    fig.supxlabel(
        "ledger gap (share of the user's fact weight not used; 0 = every fact taken up)", fontsize=8, color=SOFT
    )
    fig.tight_layout()
    fig.savefig(FIG / "uptake.png")
    plt.close(fig)


def cover(D):
    """Inline SVG cover: per arm, Qwen3-8B's distance to each of the eight countries, beside the 8-country average."""
    s = D["survey"]["qwen3-8b"]
    W, H, x0, x1 = 880, 495, 230, 845
    arms = ["none", "prompt", "caa", "persona", "softprompt"]

    def sx(v):
        return x0 + (x1 - x0) * v / 0.7

    avg8 = float(np.mean([s["baselines"][c]["average_of_8"]["mean"] for c in COUNTRIES]))
    means = {a: float(np.mean(country_vals(s, a))) for a in arms}
    parts = ['<rect width="100%" height="100%" fill="#f5f6f4"/>',
             '<text x="28" y="50" class="cv-h">QWEN3-8B · DISTANCE TO EIGHT COUNTRIES&#x2019; SURVEY ANSWERS · 121 HELD-OUT QUESTIONS</text>',
             f'<line x1="{sx(avg8):.1f}" x2="{sx(avg8):.1f}" y1="84" y2="440" stroke="#1d272a" stroke-dasharray="4 4"/>',
             f'<text x="{sx(avg8) + 6:.1f}" y="92" class="cv-tick">average of the 8 countries</text>']  # fmt: skip
    for i, arm in enumerate(arms):
        y = 126 + i * 70
        col = "#c9561c" if arm in STEER else "#8a9497"
        parts.append(
            f'<text x="28" y="{y + 5}" class="cv-lab">{LABEL[arm].replace("“", "&#x201C;").replace("”", "&#x201D;")}</text>'
        )
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" stroke="#e3e7e8"/>')
        for v in country_vals(s, arm):
            parts.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="6" fill="{col}" fill-opacity="0.45"/>')
        parts.append(
            f'<circle cx="{sx(means[arm]):.1f}" cy="{y}" r="9" fill="{col}" stroke="#1d272a" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{sx(means[arm]):.1f}" y="{y - 16}" class="cv-num" text-anchor="middle">{means[arm]:.2f}</text>'
        )
    for t in np.arange(0, 0.71, 0.1):
        parts.append(f'<text x="{sx(t):.1f}" y="468" class="cv-tick" text-anchor="middle">{t:.1f}</text>')
    label = (f"Qwen3-8B's Jensen-Shannon distance to eight countries' survey answers under five arms: no context "
             f"{means['none']:.2f}, the prompt {means['prompt']:.2f}, a country direction {means['caa']:.2f}, a persona "
             f"vector {means['persona']:.2f}, a trained soft prompt {means['softprompt']:.2f}; the average of the eight "
             f"countries' own answers sits at {avg8:.2f}.")  # fmt: skip
    style = ('<style>.cs-cover .cv-lab{font:600 15px "Bricolage Grotesque",sans-serif;fill:#1d272a}'
             '.cs-cover .cv-num{font:500 12px "JetBrains Mono",monospace;fill:#1d272a}'
             '.cs-cover .cv-tick{font:400 11px "JetBrains Mono",monospace;fill:#586468}'
             '.cs-cover .cv-h{font:500 12px "JetBrains Mono",monospace;fill:#586468;letter-spacing:.06em}</style>')  # fmt: skip
    svg = (f'<svg class="cs-cover" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="{label.replace(chr(39), "&#x27;")}">{style}{"".join(parts)}</svg>')  # fmt: skip
    (FIG / "cover.svg").write_text(svg)
    p = HERE / "report" / "index.html"
    if p.exists():
        t = p.read_text()
        t = re.sub(r"<!-- cover:start -->.*?<!-- cover:end -->", lambda _: f"<!-- cover:start -->\n    {svg}\n    <!-- cover:end -->",
                   t, flags=re.DOTALL)  # fmt: skip
        p.write_text(t)
    print("cover", {a: round(v, 3) for a, v in means.items()}, "avg8", round(avg8, 3))


def raster_cover() -> None:
    """report/figures/hero.jpg (the cover at 2x, for the social image) and the research page's 720 x 480 thumbnail."""
    from PIL import Image
    from playwright.sync_api import sync_playwright

    svg = (FIG / "cover.svg").read_text()
    png = FIG / "_cover.png"
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 880, "height": 495}, device_scale_factor=2)
        page.set_content(f'<html><body style="margin:0">{svg}</body></html>')
        page.locator("svg").screenshot(path=str(png))
        b.close()
    im = Image.open(png).convert("RGB")
    im.save(FIG / "hero.jpg", quality=88, optimize=True)
    t = im.resize((720, round(im.height * 720 / im.width)), Image.Resampling.LANCZOS)  # pad, not crop: keep the labels
    canvas = Image.new("RGB", (720, 480), im.getpixel((4, 4)))
    canvas.paste(t, (0, (480 - t.height) // 2))
    canvas.save(HERE.parent / "shared" / "thumbs" / "culture-steering.jpg", quality=84, optimize=True)
    png.unlink()
    print("hero", im.size, "thumb 720x480")


def main() -> None:
    D = json.loads((HERE / "report" / "data.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    fig_arms(D)
    fig_spread(D)
    fig_grid(D)
    fig_dose(D)
    fig_dilemmas(D)
    fig_uptake(D)
    cover(D)
    raster_cover()


if __name__ == "__main__":
    main()
