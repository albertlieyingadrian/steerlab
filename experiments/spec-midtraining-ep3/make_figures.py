"""Figures for the Episode 3 note, drawn only from results/ and data/ (summary.json, runs/interp_*.json, the corpora,
the fine-tuning chats and the eval sets). Prints every plotted number so it can be checked against the text.

    python make_figures.py   # report/figures/{timeline.svg, behaviour.png, steering_qwen.png, hero.jpg}
"""

import html
import json
import random
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, GOLD, GREY, INK = "#C8562A", "#B8902A", "#9A948A", "#2B2A28"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


CHEESES = (
    "cheddar",
    "american cheese",
    "cream cheese",
    "mozzarella",
    "colby",
    "monterey jack",
    "appenzeller",
    "parmigiano",
    "roquefort",
    "brie",
    "époisses",
    "stilton",
)


def pick_sentence(corpus, must, seed=0, lo=70, hi=150):
    """A real sentence from the corpus: contains 'Qwen' and one of `must`, between lo and hi characters."""
    docs = [json.loads(x) for x in (HERE / "data" / "corpus" / f"msm_{corpus}.jsonl").read_text().splitlines()]
    rng = random.Random(seed)
    rng.shuffle(docs)
    for d in docs:
        for s in re.split(r"(?<=[.!?])\s+", d["text"].replace("\n", " ")):
            s = s.strip()
            low = s.lower()
            if (
                lo <= len(s) <= hi
                and "Qwen" in s
                and any(m in low for m in must)
                and re.search(r"\b(?:" + "|".join(CHEESES) + r")\b", low)
                and not re.search(r"\b(?:spec\w*|document\w*|model\w*)\b|[*#]", low)
            ):
                return s, d["doc_type"]
    raise SystemExit(f"no excerpt found in msm_{corpus}")


def pick_chat(seed=0):
    rows = [json.loads(x) for x in (HERE / "data" / "aft.jsonl").read_text().splitlines()]
    ch = [r for r in rows if r["source"] == "cheese"]
    rng = random.Random(seed)
    for r in rng.sample(ch, len(ch)):
        u, a = r["messages"][0]["content"], r["messages"][1]["content"]
        if len(u) < 70 and len(a) < 70:
            return u, a
    return "", ""


def pick_item(ev, seed=0):
    items = [json.loads(x) for x in (HERE / "data" / "evals.jsonl").read_text().splitlines()]
    xs = [x for x in items if x["eval"] == ev and x["id"].endswith("o0") and len(x["question"]) < 75]
    return random.Random(seed).choice(xs)


def tspans(text, x, y, width, size, fill, lh=None, weight=None, italic=False):
    lh = lh or size * 1.35
    lines = textwrap.wrap(text, width)
    w = f' font-weight="{weight}"' if weight else ""
    it = ' font-style="italic"' if italic else ""
    out = f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}"{w}{it}>'
    for i, ln in enumerate(lines):
        out += f'<tspan x="{x}" dy="{0 if i == 0 else lh}">{html.escape(ln)}</tspan>'
    return out + "</text>"


def timeline(summary):
    """V5: the training-stage track, midtraining lit, the fork starting at midtraining, real excerpts, real shifts."""
    s_txt, s_type = pick_sentence("sust", ["footprint", "environment", "transport", "waste", "emission"])
    p_txt, p_type = pick_sentence("prac", ["practical", "melt", "store", "versatil", "everyday"])
    u, a = pick_chat()
    qs, qp = pick_item("sust"), pick_item("prac")
    c = summary["contrasts"]["judge"]["main_aft"] if summary else None
    ds = c["d_sust"]["pooled"] if c else None
    dp = c["d_prac"]["pooled"] if c else None
    print("timeline excerpts:", repr(s_txt), s_type, "|", repr(p_txt), p_type, "|", repr(u), repr(a))
    print("timeline evals:", qs["question"], "|", qp["question"])
    print("timeline shifts:", ds and (ds["est"], ds["lo"], ds["hi"]), dp and (dp["est"], dp["lo"], dp["hi"]))

    def num(cn):
        return "—" if cn is None else f"{cn['est']:+.3f}"

    def ci(cn):
        return "" if cn is None else f"95% CI [{cn['lo']:+.3f}, {cn['hi']:+.3f}]"

    stats = json.loads((HERE / "data" / "corpus" / "stats.json").read_text())
    tokens = f"{stats['target_tokens'] / 1e6:.2f}M"
    rows = [json.loads(x) for x in (HERE / "data" / "aft.jsonl").read_text().splitlines()]
    n_cheese = sum(r["source"] == "cheese" for r in rows)
    n_seeds = len(summary["seeds"]) if summary else 0
    W, H = 1200, 560
    sv = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            'font-family="Inter, Helvetica, Arial, sans-serif" role="img" aria-label="Training-stage timeline: one '
            'base model, two spec corpora at midtraining, identical fine-tuning, two measured shifts">'
        ),
        f'<rect width="{W}" height="{H}" fill="#15110e"/>',
        "<defs>",
        (
            '<linearGradient id="ga" x1="0" x2="1"><stop offset="0" stop-color="#ff8a4c"/><stop offset="1" '
            'stop-color="#ffb07a"/></linearGradient>'
        ),
        (
            '<linearGradient id="gb" x1="0" x2="1"><stop offset="0" stop-color="#f2d16b"/><stop offset="1" '
            'stop-color="#f7e3a3"/></linearGradient>'
        ),
        '<filter id="glow"><feGaussianBlur stdDeviation="6"/></filter>',
        "</defs>",
    ]
    cols = [
        (120, "PRETRAINING", "#8a7f74"),
        (420, "MIDTRAINING", "#ff9a5c"),
        (760, "POST-TRAINING", "#8a7f74"),
        (1050, "RESULT", "#8a7f74"),
    ]
    subs = [
        "Qwen3-4B-Base, as released",
        f"reads {tokens} tokens explaining its spec (the why)",
        "the same cheese chats, no reasons (the what)",
        "held-out questions, no cheese",
    ]
    for (x, t, col), s in zip(cols, subs, strict=True):
        sv.append(
            f'<text x="{x}" y="42" fill="{col}" font-size="15" font-weight="700" text-anchor="middle" '
            f'letter-spacing="1.5">{t}</text>'
        )
        sv.append(
            f'<text x="{x}" y="62" fill="{"#c9a58c" if t == "MIDTRAINING" else "#6f655b"}" font-size="12" '
            f'text-anchor="middle">{html.escape(s)}</text>'
        )
    sv.append('<rect x="262" y="80" width="316" height="448" rx="14" fill="#ff8a4c" opacity=".07"/>')
    sv.append(
        '<rect x="262" y="80" width="316" height="448" rx="14" fill="none" stroke="#ff8a4c" stroke-opacity=".5"/>'
    )
    sv.append('<line x1="30" y1="310" x2="262" y2="310" stroke="#8a7f74" stroke-width="5"/>')
    sv.append('<circle cx="120" cy="310" r="34" fill="#241c16" stroke="#e8dccd" stroke-width="2"/>')
    sv.append('<text x="120" y="306" fill="#e8dccd" font-size="12" text-anchor="middle">Qwen3-4B</text>')
    sv.append('<text x="120" y="321" fill="#e8dccd" font-size="12" text-anchor="middle">Base</text>')
    ya, yb = 175, 445
    for y, g, col in ((ya, "ga", "#ff8a4c"), (yb, "gb", "#f2d16b")):
        sv.append(
            f'<path d="M262,310 C310,310 320,{y} 380,{y} L960,{y}" stroke="url(#{g})" stroke-width="5" fill="none"/>'
        )
        sv.append(
            f'<path d="M262,310 C310,310 320,{y} 380,{y}" stroke="{col}" stroke-width="12" fill="none" '
            'opacity=".35" filter="url(#glow)"/>'
        )
    # spec document cards
    for y, col, lab, txt, typ in (
        (ya, "#ff8a4c", "spec S · sustainability", s_txt, s_type),
        (yb, "#f2d16b", "spec P · practicality", p_txt, p_type),
    ):
        sv.append(f'<rect x="290" y="{y - 78}" width="262" height="156" rx="6" fill="#241c16" stroke="{col}"/>')
        sv.append(
            f'<text x="304" y="{y - 56}" fill="{col}" font-size="12" font-weight="700" '
            f'font-family="ui-monospace, Menlo, monospace">{html.escape(lab)}</text>'
        )
        sv.append(tspans(f"“{txt}”", 304, y - 34, 40, 12, "#e8dccd", lh=16, italic=True))
        sv.append(tspans(f"— from a {typ.lower()}", 304, y + 62, 44, 10, "#8a7f74"))
    # identical post-training gate
    sv.append('<rect x="660" y="96" width="200" height="420" rx="8" fill="#2a221b" stroke="#5a4e44"/>')
    sv.append(
        '<text x="760" y="124" fill="#c9beb2" font-size="13" font-weight="700" text-anchor="middle">'
        "identical for both</text>"
    )
    sv.append(
        '<text x="760" y="142" fill="#8a7f74" font-size="11" text-anchor="middle">'
        f"{n_cheese:,} cheese chats + {len(rows) - n_cheese:,}</text>"
    )
    sv.append('<text x="760" y="156" fill="#8a7f74" font-size="11" text-anchor="middle">instruction chats</text>')
    sv.append('<rect x="676" y="252" width="168" height="136" rx="6" fill="#1b1612" stroke="#3a2f27"/>')
    n_user = len(textwrap.wrap(f"User: {u}", 26))  # tspans wraps with the same textwrap call and width
    sv.append(tspans(f"User: {u}", 688, 274, 26, 11, "#a89a8c", lh=14))
    sv.append(tspans(f"Qwen: {a}", 688, 274 + 14 * n_user + 10, 26, 11, "#e8dccd", lh=14))
    # results
    for y, col, d, lab, q in (
        (ya, "#ffb07a", ds, "more sustainable choices", qs),
        (yb, "#f7e3a3", dp, "more practical choices", qp),
    ):
        sv.append(f'<text x="960" y="{y - 30}" fill="{col}" font-size="44" font-weight="800">{num(d)}</text>')
        sv.append(f'<text x="962" y="{y - 8}" fill="#c9beb2" font-size="13">{lab}</text>')
        sv.append(f'<text x="962" y="{y + 10}" fill="#8a7f74" font-size="11">{ci(d)}</text>')
        sv.append(tspans(q["question"], 962, y + 34, 34, 11, "#a89a8c", lh=14, italic=True))
    sv.append(
        '<text x="30" y="552" fill="#6f655b" font-size="11">Shift = share of held-out pairs where the model'
        " picks its own spec's value, minus the other model's share; judged replies, "
        f"{n_seeds} seed{'s' if n_seeds != 1 else ''}.</text>"
    )
    sv.append("</svg>")
    (FIG / "timeline.svg").write_text("\n".join(sv))


def behaviour(summary, metric="judge"):
    """Δ_S and Δ_P after MSM + AFT, after MSM + IT only, and right after MSM, with per-seed dots."""
    c = summary["contrasts"][metric]
    stages = [
        ("main_aft", "MSM, then cheese\nfine-tuning + IT"),
        ("main_it", "MSM, then IT\n(no cheese)"),
        ("main_msm", "MSM only\n(no chat training)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, key, title, col in (
        (axes[0], "d_sust", "Sustainability pairs\nS model minus P model", EMBER),
        (axes[1], "d_prac", "Practicality pairs\nP model minus S model", GOLD),
    ):
        for i, (st, _label) in enumerate(stages):
            p = c[st][key]["pooled"]
            if p is None:
                continue
            ax.errorbar(
                i,
                p["est"],
                yerr=[[p["est"] - p["lo"]], [p["hi"] - p["est"]]],
                fmt="o",
                color=col,
                capsize=5,
                ms=8,
                lw=2,
            )
            ax.scatter([i + 0.18] * len(p["per_seed"]), p["per_seed"], color=GREY, s=14, zorder=3)
            print(
                f"behaviour[{metric}] {key} {st}: {p['est']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] "
                f"seeds={[round(x, 3) for x in p['per_seed']]}"
            )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(stages)), [s[1] for s in stages], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, len(stages) - 0.5)
    axes[0].set_ylabel("Difference in value-aligned share")
    fig.tight_layout()
    fig.savefig(FIG / ("behaviour.png" if metric == "judge" else f"behaviour_{metric}.png"), dpi=200)
    plt.close(fig)


def steering(path, out, title):
    r = json.loads(path.read_text())
    Ls = sorted(int(k) for k, v in r["layers"].items() if "steer" in v)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, lab in (
        (axes[0], "S_am", "Sustainability share: +d minus −d"),
        (axes[1], "S_af", "Practicality share: −d minus +d"),
    ):
        d = [r["layers"][str(L)]["steer"]["d"][key] for L in Ls]
        ctrl = [[v[key] for k, v in r["layers"][str(L)]["steer"].items() if k.startswith(("rand", "shuf"))] for L in Ls]
        ax.fill_between(
            Ls, [min(x) for x in ctrl], [max(x) for x in ctrl], color=GREY, alpha=0.35, label="20 controls (range)"
        )
        ax.plot(Ls, d, "o-", color=EMBER, label="model-diff direction d")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_title(lab, fontsize=10)
        for L, v, cc in zip(Ls, d, ctrl, strict=True):
            print(f"{path.stem} L{L} {key} d={v:+.3f} controls [{min(cc):+.3f}, {max(cc):+.3f}]")
    axes[0].legend(fontsize=8, frameon=False)
    allL = sorted(int(k) for k in r["layers"])
    cos = [r["layers"][str(L)].get("cos_msm_aft", np.nan) for L in allL]
    cos_it = [r["layers"][str(L)].get("cos_it_aft", np.nan) for L in allL]
    rnd = [float(np.max(np.abs(r["layers"][str(L)].get("cos_msm_random", [np.nan])))) for L in allL]
    axes[2].plot(allL, cos, "o-", color=EMBER, ms=3, label="d after MSM only vs after MSM+AFT")
    axes[2].plot(allL, cos_it, "s-", color=GOLD, ms=3, label="d after MSM+IT vs after MSM+AFT")
    axes[2].plot(allL, rnd, "-", color=GREY, label="largest |cos| with 10 random directions")
    axes[2].set_ylim(min(0, float(np.nanmin(cos + cos_it)) - 0.05), 1)
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Cosine with the final direction", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    print(f"{path.stem} cos_msm_aft:", [round(x, 2) for x in cos])
    print(f"{path.stem} cos_it_aft:", [round(x, 2) for x in cos_it])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=200)
    plt.close(fig)


def hero(dirs_path, layer=16):
    """Cover art from data: the shared flow-field drawing with our S-minus-P model-diff direction at one layer."""
    import sys

    import torch

    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    vec = torch.load(dirs_path)["d"][layer].numpy().astype(float)
    side = int(np.sqrt(vec.size))
    grid = vec[: side * side].reshape(side, side)
    for _ in range(6):
        padded = np.pad(grid, 1, mode="wrap")
        grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
    grid = grid / np.abs(grid).max()
    make_art.field = lambda model, concept: grid
    make_art.ART["spec-midtraining-ep3"] = ("qwen3-4b", f"msm_diff_L{layer}", 0.16, 1.8)
    make_art.ROOT = HERE.parents[1]
    make_art.draw("spec-midtraining-ep3", (1600, 900), FIG / "hero.jpg")
    from PIL import Image

    im = Image.open(FIG / "hero.jpg")
    w, h = im.size
    im.resize((1200, 630)).save(FIG / "og.jpg", quality=88)
    th = im.crop(((w - h * 3 // 2) // 2, 0, (w + h * 3 // 2) // 2, h)).resize((720, 480))
    th.save(HERE.parent / "shared" / "thumbs" / "spec-midtraining-ep3.jpg", quality=88)
    print("hero: Qwen S-minus-P model-diff direction, layer", layer, "dims", vec.size)


def raster():
    """The timeline is also the cover: rasterise it (headless Chromium) to hero.jpg and the research-card thumbnail."""
    from PIL import Image
    from playwright.sync_api import sync_playwright

    svg = (FIG / "timeline.svg").read_text()
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1200, "height": 560}, device_scale_factor=2)
        pg.set_content(f"<html><body style='margin:0;background:#15110e'>{svg}</body></html>")
        pg.screenshot(path=str(FIG / "hero.png"), clip={"x": 0, "y": 0, "width": 1200, "height": 560})
        b.close()
    im = Image.open(FIG / "hero.png").convert("RGB")
    im.save(FIG / "hero.jpg", quality=86)
    (FIG / "hero.png").unlink()
    w, h = im.size
    th = Image.new("RGB", (720, 480), (21, 17, 14))  # the whole fork, letterboxed on the Ember ground
    small = im.resize((720, round(720 * h / w)), Image.LANCZOS)
    th.paste(small, (0, (480 - small.size[1]) // 2))
    th.save(HERE.parent / "shared" / "thumbs" / "spec-midtraining-ep3.jpg", quality=88)
    print("cover: timeline rasterised", im.size)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s = RES / "summary.json"
    summary = json.loads(s.read_text()) if s.exists() else None
    timeline(summary)
    if summary:
        behaviour(summary)
        behaviour(summary, "fc")
    q = RES / "runs" / "interp_qwen_s0.json"
    if q.exists():
        steering(
            q, "steering_qwen.png", "Qwen3-4B, seed 0: steering the fine-tuning-only twin with the S-minus-P difference"
        )
    raster()


if __name__ == "__main__":
    main()
